#!/usr/bin/env python3
"""比較評価: (A) textlint / (B) 表層規則群 / (C) 汎用LLM / (D) Deference / (E) 量子化後。

評価軸（6つ）:
    (1) 誤り種別ごとの検出率 — 特に敬意の向きの誤り（規則では捕まらない類）
    (2) 揺れの範囲を誤検出しない率 — 実用性の要（過剰指摘は嫌われる）
    (3) 修正候補の妥当性 — 規則で生成可能な正しい形と一致するか
    (4) 誤り種別分類の正解率 — 混同行列も出す
    (5) CPU での応答時間 — 中央値と p95
    (6) 業種別アダプタの効果

**利用不能な条件の数字は捏造しない。** API キーが無い (C) や、ONNX が無い (E) は、
理由を結果 JSON に明記して skip する。

使い方::

    python scripts/run_benchmarks.py --data data/hf --output benchmarks/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deference.baselines import LLMBaseline, RuleBaseline, TextlintBaseline  # noqa: E402
from deference.pipeline import Deference  # noqa: E402
from deference.types import (  # noqa: E402
    Audience,
    CheckResult,
    ErrorType,
    Finding,
    MailContext,
    Party,
    Person,
    Span,
    Verdict,
    ERROR_TYPE_JA,
)

CONDITIONS = ("A", "B", "C", "D", "E")
CONDITION_LABEL = {
    "A": "(A) textlint",
    "B": "(B) 表層規則群",
    "C": "(C) 汎用LLM",
    "D": "(D) Deference（量子化前）",
    "E": "(E) Deference（量子化後）",
}


# ---------------------------------------------------------------------------
def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open(encoding="utf-8")]


def row_context(row: Dict[str, Any]) -> MailContext:
    persons = tuple(
        Person(p["name"], Party(p["side"]), p.get("title", ""))
        for p in json.loads(row.get("persons") or "[]")
    )
    return MailContext(
        audience=Audience(row["audience"]),
        writer_org=row.get("writer_org", "弊社"),
        recipient_org=row.get("recipient_org", "貴社"),
        persons=persons,
    )


def row_gold(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return json.loads(row.get("errors") or "[]")


# ---------------------------------------------------------------------------
class OnnxDeference:
    """条件 (E)。ONNX（int8 量子化）で推論する Deference。

    実証する主張: 「速度」。量子化前後で**同じ本文集合**を同じ手順で測り、
    精度の変化と速度の変化を両方報告する。
    """

    name = "deference_onnx"

    def __init__(self, onnx_dir: Path, *, quantized: bool = True) -> None:
        import numpy as np
        import onnxruntime as ort

        from deference.model import ErrorSpanClassifier

        path = onnx_dir / ("model.int8.onnx" if quantized else "model.onnx")
        if not path.exists():
            raise FileNotFoundError(path)
        self._np = np
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # CPU 単スレッドで公平に測る
        self.sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        # 符号化と復号だけ再利用する（PyTorch の重みは読まない）。
        # 量子化後の条件で PyTorch モデルを読み込んでしまうと、
        # 「量子化して軽くした」という主張と実測が食い違う。
        self._helper = ErrorSpanClassifier.tokenizer_only(onnx_dir, max_length=192)
        self.tokenizer = self._helper.tokenizer
        self.labels = json.loads((onnx_dir / "labels.json").read_text())

    def predict(
        self, text: str, context: MailContext, *, threshold: float = 0.5
    ) -> List[Finding]:
        np = self._np
        enc = self._helper.encode(text, context)
        ids = np.array([enc["input_ids"]], dtype=np.int64)
        mask = np.array([enc["attention_mask"]], dtype=np.int64)
        logits = self.sess.run(
            ["logits"], {"input_ids": ids, "attention_mask": mask}
        )[0][0]
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        preds = probs.argmax(axis=-1).tolist()
        confs = probs.max(axis=-1).tolist()
        return self._helper._decode_findings(text, enc, preds, confs, threshold)


def build_onnx_condition(onnx_dir: Path) -> Optional[Deference]:
    try:
        engine = OnnxDeference(onnx_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"  (E) を作れません: {exc}", file=sys.stderr)
        return None
    df = Deference(engine="norm")
    df.engine = "neural"
    df._model = engine  # type: ignore[attr-defined]
    return df


# ---------------------------------------------------------------------------
def reportable(result: CheckResult) -> List[Finding]:
    return [f for f in result.findings if f.verdict is Verdict.NORM_DIVERGENCE]


def run_condition(
    key: str,
    runner: Callable[[Sequence[str], Sequence[MailContext]], List[CheckResult]],
    rows: Sequence[Dict[str, Any]],
) -> List[CheckResult]:
    texts = [r["text"] for r in rows]
    ctxs = [row_context(r) for r in rows]
    return runner(texts, ctxs)


def evaluate(
    key: str,
    err_rows: Sequence[Dict[str, Any]],
    err_results: Sequence[CheckResult],
    ok_rows: Sequence[Dict[str, Any]],
    ok_results: Sequence[CheckResult],
    var_rows: Sequence[Dict[str, Any]],
    var_results: Sequence[CheckResult],
) -> Dict[str, Any]:
    """6つの評価軸をまとめて算出する。"""
    # --- (1) 誤り種別ごとの検出率 ---------------------------------------
    total: Counter = Counter()
    hit_loc: Counter = Counter()
    hit_type: Counter = Counter()
    confusion: Dict[str, Counter] = defaultdict(Counter)
    sugg_total = 0
    sugg_exact = 0

    for row, res in zip(err_rows, err_results):
        found = reportable(res)
        for g in row_gold(row):
            gt = g["type"]
            gspan = Span(int(g["start"]), int(g["end"]))
            total[gt] += 1
            overlap = [f for f in found if f.span.overlaps(gspan)]
            if overlap:
                hit_loc[gt] += 1
                best = max(overlap, key=lambda f: f.span.iou(gspan))
                confusion[gt][best.error_type.value] += 1
                if any(f.error_type.value == gt for f in overlap):
                    hit_type[gt] += 1
                # --- (3) 修正候補の妥当性 -----------------------------
                #
                # 生の文字列比較だけでは検出スパンの粒度差に左右される。
                # たとえば「願わさせていただきます」に対し、検出スパンが
                # 「願わさせ」なら候補は「願わせ」になり、gold の
                # 「願わせていただきます」とは文字列として一致しない。
                # そこで **候補を本文に当てはめた結果**で判定する。
                # 直された本文が注入前の正しい本文に戻れば妥当な修正である。
                gold_forms = set(g.get("gold", ()))
                source = row.get("source_text") or ""
                if gold_forms:
                    sugg_total += 1
                    ok = False
                    for f in overlap:
                        for sg in f.suggestions:
                            if sg.text in gold_forms:
                                ok = True
                                break
                            fixed = (
                                row["text"][: f.span.start]
                                + sg.text
                                + row["text"][f.span.end :]
                            )
                            if source and fixed == source:
                                ok = True
                                break
                            if any(gf in fixed for gf in gold_forms):
                                ok = True
                                break
                        if ok:
                            break
                    if ok:
                        sugg_exact += 1
            else:
                confusion[gt]["__missed__"] += 1

    per_type = {
        t: {
            "n": total[t],
            "location_recall": hit_loc[t] / total[t] if total[t] else 0.0,
            "type_recall": hit_type[t] / total[t] if total[t] else 0.0,
            "is_direction": ErrorType(t).is_direction_error,
            "ja": ERROR_TYPE_JA.get(ErrorType(t), t),
        }
        for t in sorted(total)
    }
    n_all = sum(total.values())
    dir_types = [t for t in total if ErrorType(t).is_direction_error]
    n_dir = sum(total[t] for t in dir_types)

    # --- (2) 揺れを誤検出しない率 / 正例への偽陽性 -------------------------
    var_ok = 0
    var_flagged: List[Dict[str, Any]] = []
    for row, res in zip(var_rows, var_results):
        focus = Span(int(row.get("focus_start", 0)), int(row.get("focus_end", 0)))
        hits = [f for f in reportable(res) if f.span.overlaps(focus)]
        if hits:
            var_flagged.append(
                {
                    "text": row["text"],
                    "focus": row.get("focus_text", ""),
                    "flagged_as": [f.error_type.value for f in hits],
                }
            )
        else:
            var_ok += 1

    fp_docs = sum(1 for r in ok_results if reportable(r))
    fp_count = sum(len(reportable(r)) for r in ok_results)

    # --- (5) 応答時間 -----------------------------------------------------
    lat = sorted(r.elapsed_ms for r in list(err_results) + list(ok_results))
    latency = {
        "median_ms": statistics.median(lat) if lat else 0.0,
        "p95_ms": lat[int(len(lat) * 0.95)] if lat else 0.0,
        "mean_ms": statistics.fmean(lat) if lat else 0.0,
        "n": len(lat),
    }

    return {
        "condition": key,
        "label": CONDITION_LABEL[key],
        "axis1_detection_by_type": per_type,
        "axis1_overall": {
            "n": n_all,
            "location_recall": sum(hit_loc.values()) / n_all if n_all else 0.0,
            "type_recall": sum(hit_type.values()) / n_all if n_all else 0.0,
        },
        "axis1_direction_only": {
            "n": n_dir,
            "location_recall": sum(hit_loc[t] for t in dir_types) / n_dir if n_dir else 0.0,
            "type_recall": sum(hit_type[t] for t in dir_types) / n_dir if n_dir else 0.0,
        },
        "axis2_variation": {
            "n": len(var_rows),
            "not_flagged": var_ok,
            "no_overflag_rate": var_ok / len(var_rows) if var_rows else 0.0,
            "flagged_examples": var_flagged[:10],
        },
        "axis2_false_positive": {
            "n_docs": len(ok_rows),
            "docs_with_fp": fp_docs,
            "fp_doc_rate": fp_docs / len(ok_rows) if ok_rows else 0.0,
            "fp_count": fp_count,
        },
        "axis3_suggestions": {
            "n": sugg_total,
            "matched_gold": sugg_exact,
            "rate": sugg_exact / sugg_total if sugg_total else 0.0,
        },
        "axis4_confusion": {k: dict(v) for k, v in confusion.items()},
        "axis4_type_accuracy": (
            sum(hit_type.values()) / sum(hit_loc.values()) if sum(hit_loc.values()) else 0.0
        ),
        "axis5_latency": latency,
    }


# ---------------------------------------------------------------------------
def sweep_threshold(
    df: Deference,
    err_rows: Sequence[Dict[str, Any]],
    var_rows: Sequence[Dict[str, Any]],
    ok_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, float]]:
    """検出率 vs 過剰指摘率の曲線（図2 用）。

    実証する主張: 「過剰指摘の少なさ」。閾値を動かしたとき、検出率を上げると
    どれだけ過剰指摘が増えるのかを曲線で示す。左上に寄っているほど良い。
    """
    curve: List[Dict[str, float]] = []
    for th in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        det = 0
        n = 0
        for row in err_rows:
            res = df.check(row["text"], row_context(row))
            found = [
                f
                for f in res.findings
                if f.verdict is Verdict.NORM_DIVERGENCE and f.confidence >= th
            ]
            for g in row_gold(row):
                n += 1
                gs = Span(int(g["start"]), int(g["end"]))
                if any(f.span.overlaps(gs) for f in found):
                    det += 1
        over = 0
        for row in var_rows:
            res = df.check(row["text"], row_context(row))
            focus = Span(int(row.get("focus_start", 0)), int(row.get("focus_end", 0)))
            if any(
                f.verdict is Verdict.NORM_DIVERGENCE
                and f.confidence >= th
                and f.span.overlaps(focus)
                for f in res.findings
            ):
                over += 1
        fp = 0
        for row in ok_rows:
            res = df.check(row["text"], row_context(row))
            if any(
                f.verdict is Verdict.NORM_DIVERGENCE and f.confidence >= th
                for f in res.findings
            ):
                fp += 1
        denom = len(var_rows) + len(ok_rows)
        curve.append(
            {
                "threshold": th,
                "detection_rate": det / n if n else 0.0,
                "overflag_rate": (over + fp) / denom if denom else 0.0,
            }
        )
    return curve


# ---------------------------------------------------------------------------
def industry_effect(
    data_dir: Path, base_model: Optional[Path], adapters: Dict[str, Path]
) -> Dict[str, Any]:
    """(6) 業種別アダプタの効果。

    アダプタが未学習なら「測定不可」と明記して skip する。数字を作らない。
    """
    if not adapters:
        return {
            "measured": False,
            "reason": (
                "業種別アダプタが学習されていないため測定不可。"
                "scripts/train.py --industry-adapter <業種> で学習してください。"
            ),
        }
    out: Dict[str, Any] = {"measured": True, "by_industry": {}}
    test = load_jsonl(data_dir / "test.jsonl")
    for name, path in adapters.items():
        rows = [r for r in test if r.get("industry") == name and r["n_errors"] > 0]
        if not rows:
            continue
        base = Deference(engine="norm")
        adapted = Deference(engine="neural", model_dir=path)
        res = {}
        for tag, df in (("base", base), ("adapter", adapted)):
            hit = n = 0
            for row in rows:
                r = df.check(row["text"], row_context(row))
                found = reportable(r)
                for g in row_gold(row):
                    n += 1
                    gs = Span(int(g["start"]), int(g["end"]))
                    if any(f.span.overlaps(gs) for f in found):
                        hit += 1
            res[tag] = hit / n if n else 0.0
        out["by_industry"][name] = res
    return out


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/hf")
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--output", default="benchmarks/results.json")
    ap.add_argument("--model-dir", default="checkpoints/deference-base")
    ap.add_argument("--onnx-dir", default="exports/deference-base/onnx")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    data_dir = Path(args.data)
    test = load_jsonl(data_dir / "test.jsonl")
    direction = load_jsonl(data_dir / "direction.jsonl")
    variation = load_jsonl(data_dir / "variation.jsonl")

    # 誤りを含む評価集合（test + direction を混ぜる。向き系を十分に含めるため）
    err_rows = [r for r in test if r["n_errors"] > 0][: args.limit]
    seen = {r["id"] for r in err_rows}
    err_rows += [r for r in direction if r["id"] not in seen][: args.limit]
    ok_rows = [r for r in test if r["n_errors"] == 0][: args.limit]

    print(
        f"評価集合: 誤りあり {len(err_rows)} / 誤りなし {len(ok_rows)} / 揺れ {len(variation)}"
    )

    want = (
        list(CONDITIONS)
        if args.conditions == "all"
        else [c.strip().upper() for c in args.conditions.split(",")]
    )
    results: Dict[str, Any] = {
        "meta": {
            "n_error_docs": len(err_rows),
            "n_correct_docs": len(ok_rows),
            "n_variation_docs": len(variation),
            "data_dir": str(data_dir),
            "limit": args.limit,
        },
        "conditions": {},
        "skipped": {},
    }

    runners: Dict[str, Any] = {}
    if "A" in want:
        tl = TextlintBaseline()
        if tl.available():
            runners["A"] = tl.check_batch
        else:
            results["skipped"]["A"] = tl.unavailable_reason()
    if "B" in want:
        runners["B"] = RuleBaseline().check_batch
    if "C" in want:
        llm = LLMBaseline()
        if llm.available():
            runners["C"] = llm.check_batch
        else:
            results["skipped"]["C"] = llm.unavailable_reason()
    if "D" in want:
        model_dir = Path(args.model_dir)
        df = Deference(
            engine="hybrid" if model_dir.exists() else "norm",
            model_dir=model_dir if model_dir.exists() else None,
        )
        results["meta"]["D_engine"] = df.engine
        if not model_dir.exists():
            results["meta"]["D_note"] = (
                "学習済みチェックポイントが無いため、規範エンジンのみで測定した"
            )
        runners["D"] = df.check_batch
    if "E" in want:
        onnx_dir = Path(args.onnx_dir)
        dfe = build_onnx_condition(onnx_dir) if onnx_dir.exists() else None
        if dfe is None:
            results["skipped"]["E"] = (
                f"ONNX（量子化）が見つかりません: {onnx_dir}。"
                "scripts/export.py --quantize で作成してください。測定不可。"
            )
        else:
            runners["E"] = dfe.check_batch

    for key, runner in runners.items():
        print(f"\n── {CONDITION_LABEL[key]} を実行中…")
        t0 = time.time()
        err_res = run_condition(key, runner, err_rows)
        ok_res = run_condition(key, runner, ok_rows)
        var_res = run_condition(key, runner, variation)
        ev = evaluate(key, err_rows, err_res, ok_rows, ok_res, variation, var_res)
        results["conditions"][key] = ev
        a1, a1d, a2 = ev["axis1_overall"], ev["axis1_direction_only"], ev["axis2_variation"]
        print(
            f"   検出率(位置) {a1['location_recall']*100:5.1f}%  "
            f"種別 {a1['type_recall']*100:5.1f}%  "
            f"向き系 {a1d['type_recall']*100:5.1f}%  "
            f"非過剰指摘 {a2['no_overflag_rate']*100:5.1f}%  "
            f"{ev['axis5_latency']['median_ms']:.2f}ms  ({time.time()-t0:.0f}s)"
        )

    # --- 図2 用の曲線 -----------------------------------------------------
    if "D" in runners:
        print("\n── 検出率 vs 過剰指摘率 の曲線を計算中…")
        model_dir = Path(args.model_dir)
        df = Deference(
            engine="hybrid" if model_dir.exists() else "norm",
            model_dir=model_dir if model_dir.exists() else None,
        )
        results["curve_D"] = sweep_threshold(
            df, err_rows[:200], variation, ok_rows[:200]
        )

    # --- (6) 業種別アダプタ ------------------------------------------------
    adapters = {}
    for p in Path("checkpoints").glob("deference-*"):
        name = p.name.replace("deference-", "")
        if name not in ("base",) and p.is_dir():
            adapters[name] = p
    results["axis6_industry"] = industry_effect(
        data_dir, Path(args.model_dir), adapters
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果: {out}")
    write_markdown(results, out.with_suffix(".md"))
    print(f"表  : {out.with_suffix('.md')}")
    return 0


def write_markdown(results: Dict[str, Any], path: Path) -> None:
    """人間が読める表を書く。"""
    lines = ["# ベンチマーク結果", ""]
    meta = results["meta"]
    lines.append(
        f"評価集合: 誤りあり {meta['n_error_docs']} 件 / 誤りなし "
        f"{meta['n_correct_docs']} 件 / 揺れ {meta['n_variation_docs']} 件"
    )
    lines.append("")
    lines.append("## 総括")
    lines.append("")
    lines.append(
        "| 条件 | 検出率(位置) | 検出率(種別) | **向きの誤り(種別)** | "
        "非過剰指摘率 | 正例への偽陽性 | 修正候補の妥当性 | 中央値 | p95 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key in CONDITIONS:
        ev = results["conditions"].get(key)
        if not ev:
            reason = results["skipped"].get(key, "未実行")
            lines.append(f"| {CONDITION_LABEL[key]} | 測定不可 — {reason} | | | | | | | |")
            continue
        a1 = ev["axis1_overall"]
        a1d = ev["axis1_direction_only"]
        a2 = ev["axis2_variation"]
        fp = ev["axis2_false_positive"]
        a3 = ev["axis3_suggestions"]
        lat = ev["axis5_latency"]
        lines.append(
            f"| {ev['label']} | {a1['location_recall']*100:.1f}% | "
            f"{a1['type_recall']*100:.1f}% | **{a1d['type_recall']*100:.1f}%** | "
            f"{a2['no_overflag_rate']*100:.1f}% | {fp['fp_doc_rate']*100:.1f}% | "
            f"{a3['rate']*100:.1f}% | {lat['median_ms']:.2f}ms | {lat['p95_ms']:.2f}ms |"
        )
    lines.append("")

    lines.append("## (1) 誤り種別ごとの検出率（種別まで一致）")
    lines.append("")
    types: List[str] = []
    for ev in results["conditions"].values():
        for t in ev["axis1_detection_by_type"]:
            if t not in types:
                types.append(t)
    types.sort(key=lambda t: (not ErrorType(t).is_direction_error, t))
    header = "| 誤り種別 | 向き | " + " | ".join(
        CONDITION_LABEL[k] for k in CONDITIONS if k in results["conditions"]
    ) + " |"
    lines.append(header)
    lines.append("|---" * (2 + len(results["conditions"])) + "|")
    for t in types:
        ja = ERROR_TYPE_JA.get(ErrorType(t), t)
        mark = "★" if ErrorType(t).is_direction_error else ""
        cells = []
        for k in CONDITIONS:
            ev = results["conditions"].get(k)
            if not ev:
                continue
            d = ev["axis1_detection_by_type"].get(t)
            cells.append(f"{d['type_recall']*100:.1f}%" if d else "—")
        lines.append(f"| {ja} | {mark} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "★ は敬意の向きに関する誤り。〈誰の行為か〉〈誰に向かうか〉が分からないと"
        "判定できないため、表層パターンでは原理的に扱えない類型。"
    )
    lines.append("")

    if results.get("skipped"):
        lines.append("## 測定できなかった条件")
        lines.append("")
        for k, why in results["skipped"].items():
            lines.append(f"- **{CONDITION_LABEL[k]}**: {why}")
        lines.append("")

    ind = results.get("axis6_industry", {})
    lines.append("## (6) 業種別アダプタの効果")
    lines.append("")
    if not ind.get("measured"):
        lines.append(f"測定不可 — {ind.get('reason','')}")
    else:
        lines.append("| 業種 | 汎用 | アダプタ |")
        lines.append("|---|---|---|")
        for name, v in ind.get("by_industry", {}).items():
            lines.append(
                f"| {name} | {v['base']*100:.1f}% | {v['adapter']*100:.1f}% |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
