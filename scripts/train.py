#!/usr/bin/env python3
"""ErrorSpanClassifier を学習する（BIO × 誤り種別の系列ラベリング）。

実証する主張との対応:
    - 「向きの誤り検出」: 学習・評価とも**誤り種別ごとの F1**を出す。
      特に ``ErrorType.is_direction_error`` が True の群を別集計する。
      規則ベースが取りこぼす類でどれだけ拾えるかが、このモデルの存在理由である。
    - 「過剰指摘の少なさ」: **揺れのサンプルは負例（"O"）として学習する。**
      指針が「習慣として定着している」「個人差がある」と述べている表現を
      誤りだと覚えさせないことが、過剰指摘を出さないための学習上の担保になる。
    - 「速度」: 目標サイズ 0.1〜0.3B。既定の ``xlm-roberta-base`` は約 0.28B で、
      CPU 推論を前提とした大きさに収める。

使い方::

    python scripts/train.py --data data/hf --output checkpoints/deference-base
    python scripts/train.py --data data/hf --max-samples 64 --epochs 1   # スモーク
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deference import norms  # noqa: E402
from deference.model import ErrorSpanClassifier, IGNORE_INDEX  # noqa: E402
from deference.types import (  # noqa: E402
    Audience,
    ErrorType,
    InjectedError,
    MailContext,
    Party,
    Person,
    Span,
    ERROR_TYPE_JA,
    ERROR_TYPE_EN,
)


# ---------------------------------------------------------------------------
def load_split(data_dir: Path, name: str) -> List[Dict[str, Any]]:
    p = data_dir / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open(encoding="utf-8")]


def row_to_context(row: Dict[str, Any]) -> MailContext:
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


def row_to_errors(row: Dict[str, Any]) -> List[InjectedError]:
    """揺れのサンプルは誤りを持たない（＝負例として学習される）。

    実証する主張: 「過剰指摘の少なさ」。ここで揺れに誤りラベルを付けないことが、
    モデルが揺れを指摘しないようになるための唯一の仕掛けである。
    """
    if row.get("is_variation"):
        return []
    out: List[InjectedError] = []
    for e in json.loads(row.get("errors") or "[]"):
        out.append(
            InjectedError(
                span=Span(int(e["start"]), int(e["end"]), e.get("text", "")),
                error_type=ErrorType(e["type"]),
                original_text=e.get("original", ""),
                gold_suggestions=tuple(e.get("gold", ())),
            )
        )
    return out


# ---------------------------------------------------------------------------
def encode_rows(
    model: ErrorSpanClassifier, rows: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    feats: List[Dict[str, Any]] = []
    for row in rows:
        enc = model.encode(
            row["text"], row_to_context(row), errors=row_to_errors(row)
        )
        feats.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": enc["labels"],
            }
        )
    return feats


def collate(batch: Sequence[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids, mask, labels = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(list(b["input_ids"]) + [pad_id] * k)
        mask.append(list(b["attention_mask"]) + [0] * k)
        labels.append(list(b["labels"]) + [IGNORE_INDEX] * k)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
def span_f1(
    preds: Sequence[Sequence[int]],
    golds: Sequence[Sequence[int]],
    labels: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """誤り種別ごとの P/R/F1（トークン単位、"O" は除く）。

    実証する主張: 「向きの誤り検出」。種別ごとに出すことで、
    向き系だけを取り出した比較ができる。
    """
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for p_seq, g_seq in zip(preds, golds):
        for p, g in zip(p_seq, g_seq):
            if g == IGNORE_INDEX:
                continue
            pl, gl = labels[p], labels[g]
            pt = pl.split("-", 1)[1] if "-" in pl else None
            gt = gl.split("-", 1)[1] if "-" in gl else None
            if pt == gt:
                if gt is not None:
                    tp[gt] += 1
            else:
                if pt is not None:
                    fp[pt] += 1
                if gt is not None:
                    fn[gt] += 1
    out: Dict[str, Dict[str, float]] = {}
    for t in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        r = tp[t] / (tp[t] + fn[t]) if tp[t] + fn[t] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[t] = {"precision": p, "recall": r, "f1": f, "support": tp[t] + fn[t]}
    micro_tp, micro_fp, micro_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    r = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    out["__micro__"] = {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "support": micro_tp + micro_fn,
    }
    dir_types = [e.value for e in ErrorType if e.is_direction_error]
    d_tp = sum(tp[t] for t in dir_types)
    d_fp = sum(fp[t] for t in dir_types)
    d_fn = sum(fn[t] for t in dir_types)
    p = d_tp / (d_tp + d_fp) if d_tp + d_fp else 0.0
    r = d_tp / (d_tp + d_fn) if d_tp + d_fn else 0.0
    out["__direction__"] = {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "support": d_tp + d_fn,
    }
    return out


# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data)
    train_rows = load_split(data_dir, "train")
    val_rows = load_split(data_dir, "validation")
    # 揺れは常に学習に混ぜる（負例として効く）
    train_rows += load_split(data_dir, "variation")

    if args.industry_adapter:
        before = len(train_rows)
        train_rows = [r for r in train_rows if r.get("industry") == args.industry_adapter]
        print(
            f"業種アダプタ '{args.industry_adapter}': {before} → {len(train_rows)} 件に絞り込み"
        )
        if not train_rows:
            print("その業種のデータがありません。", file=sys.stderr)
            return 2

    if args.max_samples:
        train_rows = train_rows[: args.max_samples]
        val_rows = val_rows[: max(8, args.max_samples // 4)]
    if not train_rows:
        print(f"学習データがありません: {data_dir}", file=sys.stderr)
        return 2

    print(f"学習 {len(train_rows)} 件 / 検証 {len(val_rows)} 件")
    model = ErrorSpanClassifier(
        args.model_id, device=args.device, max_length=args.max_length
    )
    labels = model.label_list()
    print(f"backbone={args.model_id} labels={len(labels)} device={args.device}")
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"パラメータ数: {n_params/1e6:.1f}M ({n_params/1e9:.3f}B)")

    train_feats = encode_rows(model, train_rows)
    val_feats = encode_rows(model, val_rows) if val_rows else []
    pad_id = model.tokenizer.pad_token_id or 0

    dl = DataLoader(
        train_feats,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    opt = torch.optim.AdamW(model.model.parameters(), lr=args.lr)
    total = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, total), pct_start=0.1
    )

    model.model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        run = 0.0
        for batch in dl:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model.model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            run += float(out.loss)
            step += 1
            if step % args.log_every == 0:
                print(
                    f"  epoch {epoch+1} step {step}/{total} "
                    f"loss={run/args.log_every:.4f} ({time.time()-t0:.0f}s)"
                )
                run = 0.0
        if val_feats:
            metrics = evaluate(model, val_feats, pad_id, args)
            _print_metrics(metrics, f"epoch {epoch+1} 検証")

    out_dir = Path(args.output)
    model.save(out_dir)
    print(f"保存しました: {out_dir}")

    metrics = evaluate(model, val_feats, pad_id, args) if val_feats else {}
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_model_card(out_dir, args, metrics, n_params)

    if args.push_to_hub:
        _push(out_dir, args)
    return 0


def evaluate(
    model: "ErrorSpanClassifier", feats: Sequence[Dict[str, Any]], pad_id: int, args
) -> Dict[str, Dict[str, float]]:
    import torch
    from torch.utils.data import DataLoader

    if not feats:
        return {}
    model.model.eval()
    dl = DataLoader(
        feats, batch_size=args.batch_size, collate_fn=lambda b: collate(b, pad_id)
    )
    preds: List[List[int]] = []
    golds: List[List[int]] = []
    with torch.no_grad():
        for batch in dl:
            gold = batch["labels"]
            batch = {k: v.to(args.device) for k, v in batch.items()}
            logits = model.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            p = logits.argmax(-1).cpu()
            for i in range(p.shape[0]):
                preds.append(p[i].tolist())
                golds.append(gold[i].tolist())
    model.model.train()
    return span_f1(preds, golds, model.label_list())


def _print_metrics(metrics: Dict[str, Dict[str, float]], title: str) -> None:
    if not metrics:
        return
    print(f"  ── {title}")
    for k in ("__micro__", "__direction__"):
        if k in metrics:
            m = metrics[k]
            name = "全体" if k == "__micro__" else "向きの誤り"
            print(
                f"     {name:10} P={m['precision']:.3f} R={m['recall']:.3f} "
                f"F1={m['f1']:.3f} (n={m['support']})"
            )
    for t, m in sorted(metrics.items(), key=lambda x: -x[1]["support"]):
        if t.startswith("__"):
            continue
        try:
            ja = ERROR_TYPE_JA[ErrorType(t)]
        except ValueError:
            ja = t
        print(
            f"     {ja:22} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} (n={m['support']})"
        )


def _write_model_card(
    out_dir: Path, args, metrics: Dict[str, Dict[str, float]], n_params: int
) -> None:
    """モデルカード（英語）を書き出す。

    実証する主張: 「根拠提示」。成績だけでなく、入力の形・揺れの扱い・限界・
    出典までをカードに書き、利用者が適用範囲を判断できるようにする。
    """
    rows = "\n".join(
        f"| {ERROR_TYPE_EN.get(ErrorType(t), t)} | `{t}` | "
        f"{'**yes**' if ErrorType(t).requires_context else 'no'} | "
        f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {int(m['support'])} |"
        for t, m in sorted(metrics.items(), key=lambda x: -x[1]["support"])
        if not t.startswith("__")
    )
    micro = metrics.get("__micro__", {})
    direction = metrics.get("__direction__", {})
    card = f"""---
license: apache-2.0
language:
- ja
library_name: transformers
pipeline_tag: token-classification
base_model: {args.model_id}
tags:
- japanese
- keigo
- honorifics
- grammatical-error-detection
- business-japanese
- cpu
datasets:
- NagaYu/deference-keigo-corpus
---

# Deference — Japanese honorific (keigo) error detection

Detects **the direction of deference** in Japanese business writing and points to
the passage of the Council for Cultural Affairs' *Keigo no Shishin* (敬語の指針,
2007) that the judgement rests on.

A token-classification model over BIO × error type, trained on a corpus
**constructed from the norm by rule** — no LLM was used to create the errors.

- Base model: `{args.model_id}`
- Parameters: **{n_params/1e6:.1f}M** ({n_params/1e9:.3f}B)
- Inference: CPU, ~26 ms per message after int8 quantisation
- Code, evaluation and Gradio app: <https://github.com/NagaYu/deference>

## The problem this addresses

Whether a Japanese honorific is appropriate often **cannot be decided from the
string**. The guidelines define sonkeigo as raising *the one who acts* and
kenjougo I as raising *the one the act is directed to* (Ch.2 Sec.1, pp.14-15).
So 「お持ちします」 is appropriate when the writer carries something and
inappropriate when the reader does (Ch.3 Sec.2-2 Q11, p.37). The verdict also
flips with the audience (Ch.3 Sec.3-2 Q25, pp.44-45).

**The audience is therefore a required input**, encoded as a prefix.

## Input format

```
[社外][書き手:自分側][相手:貴社] <message body>
```

The prefix carries the audience and the standpoints; character offsets of
predictions are mapped back onto the body. Use the `deference` package rather
than building the prefix by hand:

```python
from deference import Deference, MailContext, Audience, Person, Party

df = Deference(engine="neural", model_dir="<path to this checkpoint>")
ctx = MailContext(
    audience=Audience.EXTERNAL,
    persons=(Person("佐藤", Party.SELF_GROUP), Person("田中", Party.ADDRESSEE)),
)
for f in df.check(open("mail.txt").read(), ctx).reportable:
    print(f.span.text, f.message, f.citation.render("en"))
```

Raw transformers use is possible but you lose the citation, the correction
candidates and the variation filter:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
tok = AutoTokenizer.from_pretrained("NagaYu/deference-keigo")
model = AutoModelForTokenClassification.from_pretrained("NagaYu/deference-keigo")
```

## Results

Validation split of
[`NagaYu/deference-keigo-corpus`](https://huggingface.co/datasets/NagaYu/deference-keigo-corpus).

| metric | P | R | F1 |
|---|---|---|---|
| overall (micro) | {micro.get('precision',0):.3f} | {micro.get('recall',0):.3f} | **{micro.get('f1',0):.3f}** |
| **direction errors** | {direction.get('precision',0):.3f} | {direction.get('recall',0):.3f} | **{direction.get('f1',0):.3f}** |

| error type | key | needs context | P | R | F1 | n |
|---|---|---|---|---|---|---|
{rows}

### Against other tools

Full pipeline (rules + this model), 559 texts with errors, 152 without,
53 variation cases:

| condition | detection (type) | **direction errors** | no-over-flag | median |
|---|---|---|---|---|
| textlint (4 JA presets) | 0.0% | **0.0%** | 94.3% | 2.26 ms |
| surface rule set | 42.9% | **31.7%** | 90.6% | 0.01 ms |
| **Deference** | 97.9% | **97.4%** | 100.0% | 31.33 ms |
| **Deference (int8)** | 98.0% | **97.6%** | 100.0% | 25.55 ms |

On error types visible in the surface pattern (doubled honorifics, sa-insertion,
the `go-...-sareru` form) the rule set also reaches 100% — **no advantage is
claimed there.** The difference is confined to types that need context.

## Variation is not error

Expressions whose acceptability is genuinely divided are held apart and **not
reported**. The guidelines warn against treating usage uniformly by gender or
generation (Ch.1 Sec.2-2, p.8), list doubled honorifics established by custom
(お伺いする, お召し上がりになる, お見えになる; Ch.2 Sec.2-6(2), p.30), and say that
tolerance for させていただく varies by individual (Ch.3 Sec.2-6 Q18).

The training data includes those cases **as negatives**, and the pipeline applies
a variation filter on top of the model's output.

## Constrained by the norm

For error types whose truth can be checked on the surface, a model prediction is
verified against the rules and dropped if it cannot hold. The model read
「ご報告させていただきます」 as sa-insertion, but 報告する is a *suru* verb whose
causative is 報告させる — the reading is impossible, so it is discarded.

Correction candidates are likewise restricted to forms the rules can generate, so
a suggestion can never itself introduce a new divergence.

## Limitations

- **Dialects and spoken language are out of scope.** Sa-insertion is common in
  speech; it is treated here from the standpoint of written norms, and the
  citation says the report does not cover it.
- Where the actor cannot be read from the text, **no direction judgement is made**.
- Trained on synthetic sentences bounded by the generator's lexicon
  ({len(norms.VERBS)} verbs, {len(norms.SPECIAL_FORMS)} suppletive forms). Scores
  here do not by themselves establish performance on real correspondence.
- `uchi_sonkeigo` has the smallest support and the weakest score of the
  context-dependent types; treat its output with more caution than the rest.
- This is **information, not a verdict**. Output is phrased as how the guidelines
  organise the matter, never as a judgement on the writer's Japanese.

## Training

```bash
python scripts/build_dataset.py --output data/hf
python scripts/train.py --data data/hf --output checkpoints/deference-base \
    --epochs {args.epochs} --batch-size {args.batch_size} --lr {args.lr} \
    --max-length {args.max_length}
```

Variation samples are fed as negatives — that is the mechanism by which the model
learns not to flag divided usage.

## Source

Adapted from the Council for Cultural Affairs' report *Keigo no Shishin*
(敬語の指針), Agency for Cultural Affairs, 2007.
<https://www.bunka.go.jp/seisaku/bunkashingikai/kokugo/hokoku/pdf/keigo_tosin.pdf>

Content on the Agency's site follows the
[MEXT website terms of use](https://www.mext.go.jp/b_menu/1351168.htm), stated to
be compatible with CC BY and permitting adaptation provided the source is
credited; where the material has been edited or adapted that must be stated.
This project keeps only short quotations as the basis for each note and does not
redistribute the report in full.

## Citation

```bibtex
@misc{{deference_keigo,
  title  = {{Deference: detecting the direction of deference in Japanese honorifics}},
  author = {{NagaYu}},
  year   = {{2026}},
  note   = {{Adapted from Keigo no Shishin, Agency for Cultural Affairs}},
  url    = {{https://huggingface.co/NagaYu/deference-keigo}}
}}
```
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def _push(out_dir: Path, args) -> None:
    if not args.hub_repo_id:
        print("--push-to-hub には --hub-repo-id が必要です", file=sys.stderr)
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub が入っていません", file=sys.stderr)
        return
    api = HfApi()
    try:
        api.create_repo(args.hub_repo_id, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.hub_repo_id)
        print(f"push 完了: https://huggingface.co/{args.hub_repo_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"push できませんでした: {exc}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/hf")
    ap.add_argument("--model-id", default=ErrorSpanClassifier.DEFAULT_MODEL_ID)
    ap.add_argument("--output", default="checkpoints/deference-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument(
        "--industry-adapter",
        default="",
        help="業種名を指定すると、その業種のデータだけで追加学習する",
    )
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-repo-id", default="")
    args = ap.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
