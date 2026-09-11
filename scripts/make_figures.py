#!/usr/bin/env python3
"""benchmarks/results.json から図を作る。

目玉は2枚:
    図1  誤り種別ごとの検出率。**規則ベースが「向き」の誤りをほぼ拾えない**一方で
         Deference が拾うことを示す。文脈が要る種別と表層で拾える種別を
         視覚的に区切り、どこで差が出ているのかを誤魔化さずに見せる。
    図2  検出率 vs 過剰指摘率の曲線。左上に寄っているほど良い。

実証する主張との対応:
    - 図1 → 「向きの誤り検出」
    - 図2 → 「過剰指摘の少なさ」
    - 図3 → 「速度」
    - 図4 → 誤り種別分類の正解率（評価軸4）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deference.types import ErrorType, ERROR_TYPE_JA  # noqa: E402

#: 日本語フォントの候補。見つからなければ英語ラベルに落とす。
_JP_FONTS = (
    "Hiragino Sans",
    "Hiragino Kaku Gothic Pro",
    "Noto Sans CJK JP",
    "IPAexGothic",
    "YuGothic",
    "Arial Unicode MS",
    "Osaka",
    "AppleGothic",
)

#: 日本語フォントが無い環境向けの英語ラベル
_EN_LABEL: Dict[ErrorType, str] = {
    ErrorType.DIRECTION_SWAP: "Direction swap",
    ErrorType.UCHI_SONKEIGO: "Sonkeigo to insider",
    ErrorType.SELF_SONKEIGO: "Sonkeigo to self",
    ErrorType.DEFERENCE_INVERSION: "Deference inversion",
    ErrorType.BAD_KEIGO_LINK: "Bad keigo link",
    ErrorType.DOUBLE_KEIGO: "Double keigo",
    ErrorType.SA_INSERTION: "Sa-insertion",
    ErrorType.GO_SARERU: "'go-sareru' form",
    ErrorType.OGO_DEKIRU: "'o/go-dekiru' form",
    ErrorType.SASETE_ITADAKU_OVERUSE: "'sasete itadaku' overuse",
    ErrorType.STYLE_MIXING: "Style mixing",
    ErrorType.NONE: "none",
}

_EN_CONDITION = {
    "A": "(A) textlint",
    "B": "(B) Surface rules",
    "C": "(C) General LLM",
    "D": "(D) Deference",
    "E": "(E) Deference (int8)",
}

COLORS = {
    "A": "#9aa5b1",
    "B": "#f2a65a",
    "C": "#7fb3d5",
    "D": "#2d6a4f",
    "E": "#74c69d",
}


def setup_font() -> bool:
    """日本語フォントを設定する。見つかれば True。

    豆腐（□）のまま出力しないことがこの関数の目的である。
    """
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in _JP_FONTS:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"font: {name}")
            return True
    print(
        "日本語フォントが見つかりません。英語ラベルで出力します。\n"
        "  （日本語で出したい場合は Noto Sans CJK JP などを入れてください）",
        file=sys.stderr,
    )
    return False


def label_for(et: ErrorType, jp: bool) -> str:
    return ERROR_TYPE_JA.get(et, et.value) if jp else _EN_LABEL.get(et, et.value)


def condition_label(key: str, results: Dict[str, Any], jp: bool) -> str:
    if jp:
        ev = results["conditions"].get(key)
        return ev["label"] if ev else key
    return _EN_CONDITION.get(key, key)


# ---------------------------------------------------------------------------
def fig1(results: Dict[str, Any], out: Path, jp: bool) -> None:
    """図1（目玉）: 誤り種別ごとの検出率。

    左のブロックが「文脈が無いと判定できない」種別、右が「表層で拾える」種別。
    規則ベースが左でほぼ 0 に張り付くことが一目で分かるようにする。
    """
    conds = [k for k in ("A", "B", "C", "D", "E") if k in results["conditions"]]
    types: List[str] = []
    for k in conds:
        for t in results["conditions"][k]["axis1_detection_by_type"]:
            if t not in types:
                types.append(t)
    if not types:
        print("図1: データがありません", file=sys.stderr)
        return

    ctx_types = sorted(
        [t for t in types if ErrorType(t).requires_context],
        key=lambda t: -results["conditions"][conds[-1]]["axis1_detection_by_type"]
        .get(t, {})
        .get("type_recall", 0),
    )
    sur_types = sorted(
        [t for t in types if not ErrorType(t).requires_context],
    )
    ordered = ctx_types + sur_types

    x = np.arange(len(ordered), dtype=float)
    # 2ブロックの間に隙間を空ける
    x[len(ctx_types):] += 0.9
    width = 0.8 / max(1, len(conds))

    fig, ax = plt.subplots(figsize=(13, 6.2))
    for i, k in enumerate(conds):
        per = results["conditions"][k]["axis1_detection_by_type"]
        vals = [per.get(t, {}).get("type_recall", 0.0) * 100 for t in ordered]
        pos = x + (i - (len(conds) - 1) / 2) * width
        bars = ax.bar(
            pos,
            vals,
            width,
            label=condition_label(k, results, jp),
            color=COLORS.get(k, "#888"),
            edgecolor="white",
            linewidth=0.6,
        )
        for b, v in zip(bars, vals):
            if v < 3:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    1.2,
                    "0" if v == 0 else f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#b23b3b" if v == 0 else "#555",
                    fontweight="bold" if v == 0 else "normal",
                )

    for k in results.get("skipped", {}):
        pass

    ax.set_xticks(x)
    ax.set_xticklabels(
        [label_for(ErrorType(t), jp) for t in ordered],
        rotation=22,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("検出率（種別まで一致）%" if jp else "Detection rate (type-exact) %", fontsize=11)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    # ブロックの区切りと見出し
    if ctx_types and sur_types:
        sep = (x[len(ctx_types) - 1] + x[len(ctx_types)]) / 2
        ax.axvline(sep, color="#444", linestyle="--", linewidth=1.1, alpha=0.7)
        ax.text(
            x[: len(ctx_types)].mean(),
            103,
            "文脈がなければ判定できない誤り\n（誰の行為か・宛先は誰か）"
            if jp
            else "Requires context\n(whose act? who is addressed?)",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#b23b3b",
        )
        ax.text(
            x[len(ctx_types):].mean(),
            103,
            "表層パターンでも拾える誤り" if jp else "Detectable from surface patterns",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#555",
        )
    ax.legend(loc="center left", fontsize=9.5, framealpha=0.95)
    title = (
        "誤り種別ごとの検出率 — 規則ベースは「敬意の向き」をほぼ拾えない"
        if jp
        else "Detection rate by error type — surface rules miss deference direction"
    )
    ax.set_title(title, fontsize=13.5, pad=42, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"図1: {out.with_suffix('.png')}")


# ---------------------------------------------------------------------------
def fig2(results: Dict[str, Any], out: Path, jp: bool) -> None:
    """図2（目玉）: 検出率 vs 過剰指摘率。左上が良い。"""
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    curve = results.get("curve_D") or []
    if curve:
        xs = [p["overflag_rate"] * 100 for p in curve]
        ys = [p["detection_rate"] * 100 for p in curve]
        order = np.argsort(xs)
        ax.plot(
            np.array(xs)[order],
            np.array(ys)[order],
            "-o",
            color=COLORS["D"],
            linewidth=2.2,
            markersize=5,
            label=("(D) Deference（閾値を変化）" if jp else "(D) Deference (threshold sweep)"),
            zorder=3,
        )
        # 主図では注記を省き、拡大図の側に閾値を書く（重なりを避けるため）

    for k in ("A", "B", "C", "E"):
        ev = results["conditions"].get(k)
        if not ev:
            continue
        over = 1 - ev["axis2_variation"]["no_overflag_rate"]
        fp = ev["axis2_false_positive"]["fp_doc_rate"]
        n_var = ev["axis2_variation"]["n"]
        n_ok = ev["axis2_false_positive"]["n_docs"]
        combined = (
            (over * n_var + fp * n_ok) / (n_var + n_ok) if (n_var + n_ok) else 0.0
        )
        ax.scatter(
            combined * 100,
            ev["axis1_overall"]["location_recall"] * 100,
            s=190,
            marker="D" if k != "E" else "^",
            color=COLORS.get(k, "#888"),
            edgecolor="white",
            linewidth=1.6,
            zorder=4,
            label=condition_label(k, results, jp),
        )

    ax.set_xlabel(
        "過剰指摘率（揺れ・正例を誤って指摘した割合）%"
        if jp
        else "Over-flagging rate (variations + correct text) %",
        fontsize=11,
    )
    ax.set_ylabel("検出率 %" if jp else "Detection rate %", fontsize=11)
    ax.set_title(
        "検出率 vs 過剰指摘率 — 左上ほど実用的"
        if jp
        else "Detection vs over-flagging — upper-left is better",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.set_ylim(-4, 104)
    left, right = ax.get_xlim()
    # 凡例が点を隠さないよう、右に余白を作ってから凡例を枠外に置く。
    ax.set_xlim(-max(1.0, right * 0.06), max(right, 6) * 1.08)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9.5,
        framealpha=0.95,
        borderaxespad=0.0,
    )
    ax.annotate(
        "← 左上ほど実用的" if jp else "← upper-left is better",
        xy=(0.30, 0.045),
        xycoords="axes fraction",
        fontsize=10,
        color="#2d6a4f",
        fontweight="bold",
    )

    # --- 左上を拡大した差し込み図 ---------------------------------------
    # Deference の点は左上のごく狭い範囲に固まるため、主図だけでは
    # 閾値を動かしたときの振る舞いが読み取れない。拡大図を添える。
    if curve:
        inset = ax.inset_axes([0.42, 0.42, 0.55, 0.52])
        xs = [p["overflag_rate"] * 100 for p in curve]
        ys = [p["detection_rate"] * 100 for p in curve]
        order = np.argsort(xs)
        inset.plot(np.array(xs)[order], np.array(ys)[order], "-o",
                   color=COLORS["D"], linewidth=2, markersize=5, zorder=3)
        for p in curve:
            if p["threshold"] in (0.5, 0.8, 0.9, 0.95):
                inset.annotate(
                    f"{p['threshold']:.2f}",
                    (p["overflag_rate"] * 100, p["detection_rate"] * 100),
                    textcoords="offset points", xytext=(7, -3),
                    fontsize=8, color=COLORS["D"],
                )
        ev_e = results["conditions"].get("E")
        if ev_e:
            over = 1 - ev_e["axis2_variation"]["no_overflag_rate"]
            fp = ev_e["axis2_false_positive"]["fp_doc_rate"]
            nv, no = ev_e["axis2_variation"]["n"], ev_e["axis2_false_positive"]["n_docs"]
            comb = (over * nv + fp * no) / (nv + no) if (nv + no) else 0.0
            inset.scatter(comb * 100, ev_e["axis1_overall"]["location_recall"] * 100,
                          s=120, marker="^", color=COLORS["E"],
                          edgecolor="white", linewidth=1.3, zorder=4)
        xmax = max(max(xs), 2.2)
        inset.set_xlim(-0.12, xmax * 1.35)
        inset.set_ylim(min(ys) - 4, 101.5)
        inset.grid(alpha=0.25, linestyle=":")
        inset.set_axisbelow(True)
        inset.tick_params(labelsize=8)
        inset.set_title("左上の拡大（数字は閾値）" if jp else "zoom (labels = threshold)",
                        fontsize=9)
        for spine in inset.spines.values():
            spine.set_edgecolor("#bbb")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"図2: {out.with_suffix('.png')}")


# ---------------------------------------------------------------------------
def fig3(results: Dict[str, Any], out: Path, jp: bool) -> None:
    """図3: CPU 応答時間（対数軸）。"""
    conds = [k for k in ("A", "B", "C", "D", "E") if k in results["conditions"]]
    if not conds:
        return
    med = [results["conditions"][k]["axis5_latency"]["median_ms"] for k in conds]
    p95 = [results["conditions"][k]["axis5_latency"]["p95_ms"] for k in conds]
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, med, 0.4, label="中央値" if jp else "median",
           color=[COLORS.get(k, "#888") for k in conds])
    ax.bar(x + 0.2, p95, 0.4, label="p95", color=[COLORS.get(k, "#888") for k in conds],
           alpha=0.55)
    for xi, v in zip(x - 0.2, med):
        ax.text(xi, v * 1.15, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([condition_label(k, results, jp) for k in conds],
                       rotation=14, ha="right", fontsize=9.5)
    ax.set_ylabel("1通あたりの処理時間 (ms, 対数軸)" if jp else "ms per document (log)",
                  fontsize=11)
    ax.set_title("CPU での応答時間" if jp else "CPU latency", fontsize=13,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"図3: {out.with_suffix('.png')}")


# ---------------------------------------------------------------------------
def fig4(results: Dict[str, Any], out: Path, jp: bool) -> None:
    """図4: 誤り種別分類の混同行列（Deference）。"""
    ev = results["conditions"].get("D")
    if not ev:
        return
    conf = ev["axis4_confusion"]
    golds = sorted(conf)
    preds: List[str] = []
    for g in golds:
        for p in conf[g]:
            if p not in preds:
                preds.append(p)
    preds = sorted(p for p in preds if p != "__missed__") + (
        ["__missed__"] if any("__missed__" in conf[g] for g in golds) else []
    )
    if not golds or not preds:
        return
    mat = np.zeros((len(golds), len(preds)))
    for i, g in enumerate(golds):
        tot = sum(conf[g].values()) or 1
        for j, p in enumerate(preds):
            mat[i, j] = conf[g].get(p, 0) / tot * 100

    fig, ax = plt.subplots(figsize=(9.5, 7))
    im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(preds)))
    ax.set_yticks(np.arange(len(golds)))

    def plabel(p: str) -> str:
        if p == "__missed__":
            return "検出なし" if jp else "missed"
        try:
            return label_for(ErrorType(p), jp)
        except ValueError:
            return p

    ax.set_xticklabels([plabel(p) for p in preds], rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels([label_for(ErrorType(g), jp) for g in golds], fontsize=9)
    for i in range(len(golds)):
        for j in range(len(preds)):
            if mat[i, j] > 0.5:
                ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                        fontsize=8.5, color="white" if mat[i, j] > 55 else "#333")
    ax.set_xlabel("予測" if jp else "predicted", fontsize=11)
    ax.set_ylabel("正解" if jp else "gold", fontsize=11)
    ax.set_title("誤り種別分類の混同行列（Deference, %）" if jp
                 else "Error-type confusion matrix (Deference, %)",
                 fontsize=12.5, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"図4: {out.with_suffix('.png')}")


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="benchmarks/results.json")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument(
        "--lang",
        default="en",
        choices=["en", "ja"],
        help="figure label language (default: en, matching the published docs)",
    )
    args = ap.parse_args(argv)

    path = Path(args.results)
    if not path.exists():
        print(f"結果がありません: {path}\n  scripts/run_benchmarks.py を先に実行してください。",
              file=sys.stderr)
        return 2
    results = json.loads(path.read_text(encoding="utf-8"))
    # 日本語フォントは、日本語ラベルを出すときだけ必要になる。
    # 公開ドキュメントが英語なので、既定は英語ラベル。
    jp = setup_font() and args.lang == "ja"
    if args.lang == "ja" and not jp:
        print("日本語フォントが無いため英語ラベルで出力します。", file=sys.stderr)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig1(results, outdir / "fig1_detection_by_error_type", jp)
    fig2(results, outdir / "fig2_detection_vs_overflag", jp)
    fig3(results, outdir / "fig3_latency", jp)
    fig4(results, outdir / "fig4_confusion", jp)

    for p in sorted(outdir.glob("fig*.png")):
        print(f"  {p}  {p.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
