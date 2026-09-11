#!/usr/bin/env python3
"""規範から構成した誤り付きコーパスを組み立て、Hugging Face Dataset へ push する。

**誤りの生成に LLM を使わない。** すべて `deference.generate` と `deference.inject`
の規則で作る。ラベルが厳密になり、教師バイアスも入らないためである。

実証する主張との対応:
    - 「向きの誤り検出」: ``direction`` split は向き系の誤りだけを集めた評価セット。
      規則ベースが取りこぼす類だけを取り出して比較できる。
    - 「過剰指摘の少なさ」: ``variation`` split は揺れ専用の評価セット。
      ここに指摘が出れば過剰指摘である。
    - 「根拠提示」: 各レコードに引用キーを残し、カードに出典の扱いを明記する。

使い方::

    python scripts/build_dataset.py --output data/hf
    python scripts/build_dataset.py --output data/hf --push-to-hub --hub-repo-id user/deference-keigo
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deference import norms  # noqa: E402
from deference.generate import Generator, default_context  # noqa: E402
from deference.inject import ErrorInjector  # noqa: E402
from deference.model import ErrorSpanClassifier  # noqa: E402
from deference.types import (  # noqa: E402
    Audience,
    ErrorType,
    GeneratedSentence,
    InjectedSample,
    Party,
    Span,
    ERROR_TYPE_JA,
    ERROR_TYPE_EN,
)
from deference.variation import VariationSet  # noqa: E402

#: 業種タグ。業種別アダプタの学習・評価に使う。
INDUSTRIES: Tuple[str, ...] = ("general", "it", "finance", "manufacturing", "medical")

#: 業種ごとの語彙の差し替え（本文の名詞を入れ替えるだけの軽い層）
_INDUSTRY_TERMS: Dict[str, Tuple[str, ...]] = {
    "general": ("資料", "書類", "議事録", "案件"),
    "it": ("仕様書", "リリースノート", "障害報告", "検証環境"),
    "finance": ("約款", "運用報告書", "与信審査", "決算資料"),
    "manufacturing": ("図面", "検査成績書", "部品表", "工程表"),
    "medical": ("診療情報提供書", "同意書", "検査結果", "処方内容"),
}


# ---------------------------------------------------------------------------
def _bio_labels(text: str, errors: Sequence[Any], label_list: Sequence[str]) -> List[str]:
    """文字単位の BIO 列。トークナイザに依存しない中間表現として持つ。"""
    tags = ["O"] * len(text)
    for e in errors:
        name = e.error_type.value
        start, end = e.span.start, min(e.span.end, len(text))
        if start >= end:
            continue
        tags[start] = f"B-{name}"
        for i in range(start + 1, end):
            tags[i] = f"I-{name}"
    return tags


def _record(
    rid: str,
    text: str,
    context: Any,
    errors: Sequence[Any],
    *,
    split: str,
    industry: str,
    source_text: str = "",
    is_variation: bool = False,
    variation_reason: str = "",
    function: str = "",
    kind: str = "sentence",
) -> Dict[str, Any]:
    return {
        "id": rid,
        "text": text,
        "source_text": source_text,
        "audience": context.audience.value,
        "writer_org": context.writer_org,
        "recipient_org": context.recipient_org,
        "persons": json.dumps(
            [
                {"name": p.name, "side": p.side.value, "title": p.title}
                for p in context.persons
            ],
            ensure_ascii=False,
        ),
        "errors": json.dumps(
            [
                {
                    "start": e.span.start,
                    "end": e.span.end,
                    "text": e.span.text,
                    "type": e.error_type.value,
                    "original": e.original_text,
                    "gold": list(e.gold_suggestions),
                    "citation_key": e.meta.get("citation_key", ""),
                }
                for e in errors
            ],
            ensure_ascii=False,
        ),
        "n_errors": len(errors),
        "error_types": sorted({e.error_type.value for e in errors}),
        "has_direction_error": any(e.error_type.is_direction_error for e in errors),
        "char_labels": json.dumps(
            _bio_labels(text, errors, ErrorSpanClassifier.label_list()),
            ensure_ascii=False,
        ),
        "is_variation": is_variation,
        "variation_reason": variation_reason,
        "function": function,
        "industry": industry,
        "kind": kind,
        "split": split,
    }


def _industry_flavor(text: str, industry: str, rng: random.Random) -> str:
    """本文の一般名詞を業種語に差し替える（軽い業種適応）。"""
    if industry == "general":
        return text
    terms = _INDUSTRY_TERMS.get(industry, ())
    if not terms:
        return text
    out = text
    for generic in ("資料", "書類", "議事録"):
        if generic in out:
            out = out.replace(generic, rng.choice(terms))
            break
    return out


# ---------------------------------------------------------------------------
def build(
    *,
    n_correct: int,
    n_errors: int,
    n_mails: int,
    seed: int,
    industries: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """全 split を組み立てる。"""
    g = Generator(seed=seed)
    inj = ErrorInjector(seed=seed, generator=g)
    vs = VariationSet()
    rng = random.Random(seed)

    corpus = g.generate_corpus(max(n_correct, n_errors))
    records: List[Dict[str, Any]] = []

    # --- 正例（誤りなし） --------------------------------------------------
    for i, s in enumerate(corpus[:n_correct]):
        ind = industries[i % len(industries)]
        records.append(
            _record(
                f"correct-{i:06d}",
                _industry_flavor(s.text, ind, rng),
                s.context,
                (),
                split="",
                industry=ind,
                source_text=s.text,
                function=s.function.value,
            )
        )

    # --- 誤り注入 ----------------------------------------------------------
    made = 0
    for i, s in enumerate(corpus):
        if made >= n_errors:
            break
        for et in inj.available(s):
            if made >= n_errors:
                break
            smp = inj.inject(s, et)
            if smp is None:
                continue
            ind = industries[made % len(industries)]
            records.append(
                _record(
                    f"err-{made:06d}",
                    smp.text,
                    smp.context,
                    smp.errors,
                    split="",
                    industry=ind,
                    source_text=smp.source_text,
                    function=s.function.value,
                )
            )
            made += 1

    # --- メール本文（複数文・文脈依存の誤りを含む） --------------------------
    mail_records: List[Dict[str, Any]] = []
    for i in range(n_mails):
        aud = (Audience.EXTERNAL, Audience.INTERNAL, Audience.PUBLIC)[i % 3]
        ctx = default_context(aud)
        mail = g.generate_mail(context=ctx, n_sentences=5 + (i % 3))
        ind = industries[i % len(industries)]
        if i % 3 == 0:
            mail_records.append(
                _record(
                    f"mail-ok-{i:05d}",
                    _industry_flavor(mail.text, ind, rng),
                    mail.context,
                    (),
                    split="",
                    industry=ind,
                    source_text=mail.text,
                    kind="mail",
                )
            )
            continue
        smp = inj.inject_into_mail(mail, n_errors=1 + (i % 2))
        if smp is None:
            continue
        mail_records.append(
            _record(
                f"mail-err-{i:05d}",
                smp.text,
                smp.context,
                smp.errors,
                split="",
                industry=ind,
                source_text=smp.source_text,
                kind="mail",
            )
        )
    records.extend(mail_records)

    # --- 敬意の逆転（本文単位でしか作れない） -------------------------------
    for i in range(max(20, n_mails // 10)):
        ctx = default_context(Audience.EXTERNAL)
        smp = inj.inject_deference_inversion(ctx)
        if smp is None:
            continue
        records.append(
            _record(
                f"inv-{i:05d}",
                smp.text,
                smp.context,
                smp.errors,
                split="",
                industry=industries[i % len(industries)],
                source_text=smp.source_text,
                kind="mail",
            )
        )
        break  # 同じ本文しか作れないので1件だけ入れる

    # --- 分割 --------------------------------------------------------------
    rng.shuffle(records)
    n = len(records)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for i, r in enumerate(records):
        name = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")
        r["split"] = name
        splits[name].append(r)

    # --- 揺れ専用の評価セット ---------------------------------------------
    variation: List[Dict[str, Any]] = []
    for i, c in enumerate(vs.cases()):
        variation.append(
            _record(
                f"var-{i:05d}",
                c.text,
                c.context,
                (),
                split="variation",
                industry="general",
                is_variation=True,
                variation_reason=c.reason,
                kind="variation",
            )
            | {
                "focus_start": c.focus.start,
                "focus_end": c.focus.end,
                "focus_text": c.focus.text,
                "acceptability": c.acceptability,
                "related_error_type": c.related_error_type.value,
            }
        )
    splits["variation"] = variation

    # --- 向きの誤り専用の評価セット ----------------------------------------
    direction = [
        dict(r, split="direction")
        for r in records
        if r["has_direction_error"]
    ]
    # 指針の教科書例も必ず入れる
    for i, smp in enumerate(inj.canonical_examples()):
        direction.append(
            _record(
                f"canon-{i:04d}",
                smp.text,
                smp.context,
                smp.errors,
                split="direction",
                industry="general",
                source_text=smp.source_text,
                kind="canonical",
            )
        )
    splits["direction"] = direction
    return splits


# ---------------------------------------------------------------------------
def dataset_card(splits: Dict[str, List[Dict[str, Any]]], repo_id: str) -> str:
    """データセットカード（HF frontmatter 付き・英語）を生成する。

    実証する主張: 「根拠提示」。誤りが規則で作られていること、揺れを別扱いに
    していること、出典の扱いをカード本文に明記する。
    """
    type_counts: Counter = Counter()
    for rows in splits.values():
        for r in rows:
            for t in r["error_types"]:
                type_counts[t] += 1

    rows = "\n".join(
        f"| {ERROR_TYPE_EN.get(ErrorType(t), t)} | `{t}` | {c} | "
        f"{'**yes**' if ErrorType(t).requires_context else 'no'} |"
        for t, c in type_counts.most_common()
    )
    split_rows = "\n".join(f"| `{k}` | {len(v)} |" for k, v in splits.items())

    return f"""---
license: cc-by-4.0
language:
- ja
task_categories:
- token-classification
tags:
- japanese
- keigo
- honorifics
- grammatical-error-detection
- business-japanese
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
  - split: variation
    path: variation.jsonl
  - split: direction
    path: direction.jsonl
---

# Deference — Japanese honorific (keigo) error corpus

A corpus for detecting and correcting errors in Japanese honorifics, **constructed
mechanically from the norm** rather than collected or generated by a model.

The classes, forms and conditions set out in the Council for Cultural Affairs'
report *Keigo no Shishin* (敬語の指針, 2007) are implemented as rules; correct
sentences are generated from those rules, and documented error types are then
injected — also by rule.

## No LLM was used to create the errors

Errors come from `deference/inject.py`. Two reasons:

1. **The labels are exact.** Span boundaries and error types follow from the
   definition, so there is no annotation ambiguity.
2. **No teacher bias.** If an LLM produced the errors, a model trained here would
   be learning that LLM's notion of what counts as an error, and the evaluation
   would be circular.

## Why this corpus exists

Whether a Japanese honorific form is appropriate often **cannot be decided from
the string alone**. The guidelines define sonkeigo as raising *the one who acts*
and kenjougo I as raising *the one the act is directed to* (Ch.2 Sec.1, pp.14-15).
So the very same 「お持ちします」 is appropriate when the writer carries something
and inappropriate when the reader does (Ch.3 Sec.2-2 Q11, p.37). The judgement
also flips with the audience: inside the company 「社長からごあいさつを頂きます」,
outside it 「社長からごあいさつを申し上げます」 (Ch.3 Sec.3-2 Q25, pp.44-45).

Every record therefore carries the **audience** and the **standpoint of each person
named in the text**.

## Splits

| split | rows |
|---|---|
{split_rows}

- `train` / `validation` / `test` — ordinary training and evaluation
- **`variation`** — expressions whose acceptability is genuinely divided.
  **A flag raised here is a false positive.**
- **`direction`** — only errors in the direction of deference, including the
  worked examples the guidelines themselves discuss (`canon-*`)

## Error types

| type | key | rows | needs context |
|---|---|---|---|
{rows}

"Needs context" marks the types that cannot be decided without knowing whose
action it is and who the message is addressed to — the ones a surface-pattern
linter has no route to.

## The variation label

`variation` rows and the `is_variation` field mark **divided usage, not errors**.
Treating such expressions as mistakes makes a checker unusable in practice, and
the guidelines say so themselves:

- Ch.1 Sec.2-2 (p.8): 「男女の違いや世代の違いなどによって画一的に考える態度は
  避けるべきである。」 (one should avoid treating usage uniformly by gender or
  generation)
- Ch.2 Sec.2-6(2) (p.30): on doubled honorifics, 「語によっては，習慣として定着
  しているものもある」 — listing お召し上がりになる / お見えになる / お伺いする /
  お伺いいたす / お伺い申し上げる
- Ch.3 Sec.2-5 Q17 (p.40): ご利用いただく and ご利用くださる are
  「どちらもほぼ同じように使える敬語」
- Ch.3 Sec.2-6 Q18 (pp.40-41): tolerance for させていただく varies by individual

## Fields

| name | meaning |
|---|---|
| `text` | the message body (may contain injected errors) |
| `source_text` | the same body before injection |
| `audience` | `external` / `internal` / `public` — **the same text can be judged differently** |
| `writer_org` / `recipient_org` | organisation names for each side |
| `persons` | people named in the text and their standpoint (JSON) |
| `errors` | list of `start` / `end` / `type` / `original` / `gold` / `citation_key` |
| `char_labels` | character-level BIO tags (JSON array) |
| `has_direction_error` | whether the row contains an error in the direction of deference |
| `is_variation` | whether the row is a variation case |
| `industry` | industry tag (`general` / `it` / `finance` / `manufacturing` / `medical`) |
| `kind` | `sentence` / `mail` / `variation` / `canonical` |

## Limitations

- **Dialects are out of scope.** The guidelines discuss regional variety
  (Ch.1 Sec.2-1), but this corpus targets standard written business Japanese.
- **Spoken language is out of scope.** Forms such as sa-ire kotoba are widespread
  in speech; here they are treated from the standpoint of written norms.
- **Industry flavour is shallow** — a light noun substitution layer only.
- The vocabulary is bounded by the generator's lexicon
  ({len(norms.VERBS)} verbs, {len(norms.SPECIAL_FORMS)} suppletive forms), so the
  distribution differs from naturally occurring text.
- Sentences are synthetic. High scores on this corpus do not by themselves
  establish performance on real correspondence.

## Source and how it is quoted

Adapted from the Council for Cultural Affairs' report *Keigo no Shishin*
(敬語の指針), Agency for Cultural Affairs, 2007.
<https://www.bunka.go.jp/seisaku/bunkashingikai/kokugo/hokoku/pdf/keigo_tosin.pdf>

Content on the Agency's site follows the
[MEXT website terms of use](https://www.mext.go.jp/b_menu/1351168.htm), which
state conformance with the Public Data License (v1.0), are described as
compatible with CC BY, and permit reproduction, public transmission, translation
and adaptation provided the source is credited — including commercial use. Where
the material has been edited or adapted, that fact must be stated.

This project keeps **only the short quotations needed as the basis for each
judgement** (definitions, listed forms, key sentences) in `deference/norms.py`,
each with chapter, section, question number and page. **The report is not
redistributed in full, and the PDF is not included.**

The sentences in this corpus are constructed mechanically from rules implementing
the report's classes, forms and conditions; they are not text from the report.

## Citation

```bibtex
@misc{{deference_keigo_corpus,
  title  = {{Deference: a rule-constructed corpus for Japanese honorific error detection}},
  author = {{NagaYu}},
  year   = {{2026}},
  note   = {{Adapted from Keigo no Shishin, Agency for Cultural Affairs}},
  url    = {{https://huggingface.co/datasets/{repo_id}}}
}}
```
"""


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-correct", type=int, default=1500)
    ap.add_argument("--n-errors", type=int, default=2500)
    ap.add_argument("--n-mails", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/hf")
    ap.add_argument("--industries", nargs="*", default=list(INDUSTRIES))
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-repo-id", default="")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)

    splits = build(
        n_correct=args.n_correct,
        n_errors=args.n_errors,
        n_mails=args.n_mails,
        seed=args.seed,
        industries=tuple(args.industries),
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        p = out / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name:12} {len(rows):6d} 件  → {p}")

    repo_id = args.hub_repo_id or "<your-account>/deference-keigo"
    card = dataset_card(splits, repo_id)
    (out / "README.md").write_text(card, encoding="utf-8")
    print(f"  データセットカード → {out / 'README.md'}")

    if args.push_to_hub:
        if not args.hub_repo_id:
            print("--push-to-hub には --hub-repo-id が必要です", file=sys.stderr)
            return 2
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("huggingface_hub が入っていません（pip install huggingface_hub）", file=sys.stderr)
            return 2
        api = HfApi()
        try:
            api.create_repo(
                args.hub_repo_id, repo_type="dataset", private=args.private, exist_ok=True
            )
            api.upload_folder(
                folder_path=str(out), repo_id=args.hub_repo_id, repo_type="dataset"
            )
            print(f"push 完了: https://huggingface.co/datasets/{args.hub_repo_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"push できませんでした: {exc}", file=sys.stderr)
            print("`huggingface-cli login` でトークンを設定してください。", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
