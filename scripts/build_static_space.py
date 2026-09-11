#!/usr/bin/env python3
"""静的 Space（無料枠）用のページを組み立てる。

Gradio Space の無料ホスティングには PRO が要るが、静的 Space は無料である。
ただの紹介ページにすると「宛先で判定が変わる」という中心的な主張が伝わらないので、
**実際に Deference を走らせた結果を事前計算して埋め込み**、ブラウザ側で
例と宛先を切り替えられるようにする。出力は本物で、推論だけが事前に済んでいる。

実証する主張との対応:
    - 「向きの誤り検出」: 同じ本文を社外／社内で並べ、指摘が入れ替わることを
      その場で確かめられる。
    - 「過剰指摘の少なさ」: 揺れだけを含む例を用意し、既定で何も出ないことを示す。
    - 「根拠提示」: 各指摘に指針の章・ページ・リンクをそのまま載せる。

使い方::

    python scripts/build_static_space.py --out build/space-static
    python scripts/build_static_space.py --out build/space-static --push --repo-id NagaYu/deference
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deference.baselines import TextlintBaseline  # noqa: E402
from deference.pipeline import Deference  # noqa: E402
from deference.variation import VariationSet  # noqa: E402
from deference.types import (  # noqa: E402
    Audience,
    MailContext,
    Party,
    Person,
    Verdict,
    error_type_name,
)

# ---------------------------------------------------------------------------
# 例文。いずれも実在の文面ではなく、指針が論じている型を含む作例。
EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "insider",
        "title": "In-group honorifics",
        "blurb": "The same message is judged differently inside and outside the company.",
        "text": (
            "田中様\n\n"
            "いつもお世話になっております。株式会社アルファの山田です。\n\n"
            "弊社の佐藤社長が、大変参考になったとおっしゃっておりました。\n"
            "つきましては、来週の打ち合わせについてご相談したく存じます。\n\n"
            "よろしくお願いいたします。\n"
        ),
        "persons": [("佐藤", Party.SELF_GROUP), ("田中", Party.ADDRESSEE)],
    },
    {
        "id": "direction",
        "title": "Direction of deference",
        "blurb": "Forms that a surface-pattern linter reads as perfectly ordinary.",
        "text": (
            "田中様\n\n"
            "いつもお世話になっております。\n"
            "なお、当日は私が資料をお持ちになります。\n"
            "恐れ入りますが、担当者に伺ってください。\n\n"
            "よろしくお願いいたします。\n"
        ),
        "persons": [("田中", Party.ADDRESSEE)],
    },
    {
        "id": "surface",
        "title": "Surface-visible forms",
        "blurb": "Doubled honorifics, sa-insertion, the 'go-...-sareru' pattern — a rule set finds these too.",
        "text": (
            "佐藤様\n\n"
            "資料はもうお読みになられましたか。\n"
            "こちらの機能はご利用されますでしょうか。\n"
            "明日は都合により休まさせていただきます。\n"
        ),
        "persons": [("佐藤", Party.ADDRESSEE)],
    },
    {
        "id": "variation",
        "title": "Variation only",
        "blurb": "Every form here is one the guidelines accept or call divided. Nothing is reported.",
        "text": (
            "佐藤様\n\n"
            "本日はお忙しいところ、お伺いいたします。\n"
            "資料はご持参くださいますようお願い申し上げます。\n"
            "いつもご利用いただきまして、ありがとうございます。\n"
        ),
        "persons": [("佐藤", Party.ADDRESSEE)],
    },
    {
        "id": "clean",
        "title": "In line with the guidelines",
        "blurb": "The first example rewritten. Nothing is reported.",
        "text": (
            "田中様\n\n"
            "いつもお世話になっております。株式会社アルファの山田です。\n\n"
            "弊社の佐藤が、大変参考になったと申しておりました。\n"
            "つきましては、来週の打ち合わせについてご相談したく存じます。\n\n"
            "よろしくお願いいたします。\n"
        ),
        "persons": [("佐藤", Party.SELF_GROUP), ("田中", Party.ADDRESSEE)],
    },
]

AUDIENCES = [("external", Audience.EXTERNAL), ("internal", Audience.INTERNAL)]


def _segments(text: str, findings: Sequence[Any]) -> List[Dict[str, Any]]:
    """本文を〈素の断片／指摘の断片〉に切り分ける（ブラウザ側で色を付ける）。"""
    out: List[Dict[str, Any]] = []
    pos = 0
    for f in sorted(findings, key=lambda x: (x.span.start, x.span.end)):
        if f.span.start < pos:
            continue
        if f.span.start > pos:
            out.append({"t": text[pos : f.span.start]})
        out.append(
            {
                "t": f.span.text,
                "kind": "variation"
                if f.verdict is Verdict.VARIATION
                else ("direction" if f.error_type.requires_context else "form"),
            }
        )
        pos = f.span.end
    if pos < len(text):
        out.append({"t": text[pos:]})
    return out


def run_examples(model_dir: Optional[str]) -> Dict[str, Any]:
    """各例 × 宛先で実際に検査し、結果を JSON 化する。"""
    engine = "hybrid" if model_dir else "norm"
    df = Deference(engine=engine, model_dir=model_dir, report_variations=True, lang="en")
    textlint = TextlintBaseline()
    variations = VariationSet()

    # ウォームアップ。1通目にモデル読み込みの数秒が乗ると、ページに出る
    # 「CPU で何ミリ秒」が実態とかけ離れてしまう。
    df.check("お世話になっております。", MailContext(audience=Audience.EXTERNAL))
    df.check("お世話になっております。", MailContext(audience=Audience.INTERNAL))
    payload: Dict[str, Any] = {"engine": df.engine, "examples": []}

    for ex in EXAMPLES:
        entry = {
            "id": ex["id"],
            "title": ex["title"],
            "blurb": ex["blurb"],
            "text": ex["text"],
            "runs": {},
        }
        for key, aud in AUDIENCES:
            ctx = MailContext(
                audience=aud,
                writer_org="株式会社アルファ",
                recipient_org="株式会社ベータ",
                persons=tuple(Person(n, s) for n, s in ex["persons"]),
            )
            res = df.check(ex["text"], ctx)
            tl = textlint.check(ex["text"], ctx)

            # 検出器が反応しなかった揺れも、登録簿から拾って表示する。
            # 「何も出ない」ことと「揺れとして外している」ことは違うので、
            # 外した根拠が見えるようにしておく。
            # related_error_type が NONE のものは、検出器が上げようのない項目
            # （「お世話になっております」の定型性など、指針の射程外の揺れ）。
            # 画面に出すと本文が印だらけになるだけなので、
            # 「検出器が上げえたが揺れとして外した」ものに絞る。
            from deference.types import ErrorType as _ErrorType

            known = [
                {"span": sp, "case": case}
                for sp, case in variations.match(ex["text"])
                if case.related_error_type is not _ErrorType.NONE
                and not any(sp.overlaps(f.span) for f in res.findings)
            ]

            class _Pseudo:  # 表示用の軽い器
                def __init__(self, sp):
                    self.span = sp
                    self.verdict = Verdict.VARIATION
                    from deference.types import ErrorType as _ET

                    self.error_type = _ET.NONE

            entry["runs"][key] = {
                "segments": _segments(
                    ex["text"],
                    list(res.reportable)
                    + list(res.variations)
                    + [_Pseudo(k["span"]) for k in known],
                ),
                "elapsed_ms": round(res.elapsed_ms, 1),
                "n_reportable": len(res.reportable),
                "n_variation": len(res.variations),
                "textlint_count": len(tl.findings),
                "textlint_available": not tl.meta.get("unavailable_reason"),
                "findings": [
                    {
                        "text": f.span.text,
                        "type": error_type_name(f.error_type, "en"),
                        "type_key": f.error_type.value,
                        "needs_context": f.error_type.requires_context,
                        "message": f.message,
                        "suggestions": [s.text for s in f.suggestions[:4]],
                        "citation": (
                            {
                                "source": f.citation.source_label("en"),
                                "section": f.citation.section_label("en"),
                                "page": f.citation.page,
                                "quote": f.citation.quote,
                                "url": f.citation.url,
                            }
                            if f.citation
                            else None
                        ),
                    }
                    for f in res.reportable
                ],
                "variations": [
                    {"text": f.span.text, "message": f.message}
                    for f in res.variations
                ]
                + [
                    {
                        "text": k["span"].text,
                        "message": (
                            "Held apart as variation: " + k["case"].reason
                        ),
                    }
                    for k in known
                ],
            }
        payload["examples"].append(entry)
    return payload


# ---------------------------------------------------------------------------
def _results_summary() -> Dict[str, Any]:
    """benchmarks/results.json から表に使う数字だけ取り出す。"""
    path = ROOT / "benchmarks" / "results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key in ("A", "B", "C", "D", "E"):
        ev = data["conditions"].get(key)
        if not ev:
            rows.append(
                {
                    "label": {
                        "A": "(A) textlint",
                        "B": "(B) surface rules",
                        "C": "(C) general LLM",
                        "D": "(D) Deference",
                        "E": "(E) Deference (int8)",
                    }[key],
                    "skipped": data.get("skipped", {}).get(key, "not run"),
                }
            )
            continue
        rows.append(
            {
                "label": ev["label"]
                .replace("表層規則群", "surface rules")
                .replace("（量子化前）", "")
                .replace("（量子化後）", " (int8)"),
                "type": ev["axis1_overall"]["type_recall"] * 100,
                "direction": ev["axis1_direction_only"]["type_recall"] * 100,
                "novar": ev["axis2_variation"]["no_overflag_rate"] * 100,
                "ms": ev["axis5_latency"]["median_ms"],
            }
        )
    per_type: Dict[str, Dict[str, Any]] = {}
    for key in ("A", "B", "D", "E"):
        ev = data["conditions"].get(key)
        if not ev:
            continue
        for t, d in ev["axis1_detection_by_type"].items():
            per_type.setdefault(
                t,
                {
                    "name": error_type_name(__import__(
                        "deference.types", fromlist=["ErrorType"]
                    ).ErrorType(t), "en"),
                    "needs_context": d["is_direction"],
                },
            )
            per_type[t][key] = d["type_recall"] * 100
    from deference.types import ErrorType

    for t in per_type:
        per_type[t]["needs_context"] = ErrorType(t).requires_context
    return {"rows": rows, "per_type": per_type, "meta": data.get("meta", {})}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="build/space-static")
    ap.add_argument("--model-dir", default="checkpoints/deference-base")
    ap.add_argument("--repo-id", default="NagaYu/deference")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args(argv)

    model_dir = args.model_dir if Path(args.model_dir).exists() else None
    if model_dir is None:
        print("no local checkpoint; precomputing with the rule engine only")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print("running the examples ...")
    payload = run_examples(model_dir)
    payload["benchmarks"] = _results_summary()
    (out / "data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    figs = out / "figures"
    figs.mkdir()
    for name in (
        "fig1_detection_by_error_type.png",
        "fig2_detection_vs_overflag.png",
        "fig3_latency.png",
        "fig4_confusion.png",
    ):
        src = ROOT / "figures" / name
        if src.exists():
            shutil.copy(src, figs / name)

    (out / "index.html").write_text(_INDEX_HTML, encoding="utf-8")
    (out / "style.css").write_text(_STYLE_CSS, encoding="utf-8")
    (out / "app.js").write_text(_APP_JS, encoding="utf-8")
    (out / "README.md").write_text(_SPACE_README, encoding="utf-8")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"built {out} ({total/1024:.0f} KB)")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print("   ", f.relative_to(out), f"{f.stat().st_size/1024:.0f} KB")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            args.repo_id, repo_type="space", space_sdk="static", exist_ok=True
        )
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.repo_id,
            repo_type="space",
            commit_message="Deference: the direction of deference in Japanese keigo",
        )
        print(f"pushed: https://huggingface.co/spaces/{args.repo_id}")
    return 0


_SPACE_README = """---
title: Deference — Japanese Honorific Checker
emoji: 🎎
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
models:
  - NagaYu/deference-keigo
datasets:
  - NagaYu/deference-keigo-corpus
short_description: Direction of deference in Japanese keigo
tags:
  - japanese
  - keigo
  - honorifics
  - grammatical-error-detection
---

# Deference

**The direction of deference, with the source to back it up.**

A rule-based Japanese linter says nothing about 「お持ちします」. As a string it is
fine. But **if the reader is the one carrying something, that form does not raise
them.**

This page shows **real output** from the model, computed ahead of time so the
page can stay static — switch between examples and between an internal and an
external audience to see the judgement change.

To run it on your own text:

```bash
pip install git+https://github.com/NagaYu/deference
deference check mail.txt --audience external
```

- Code and evaluation: <https://github.com/NagaYu/deference>
- Model: <https://huggingface.co/NagaYu/deference-keigo>
- Dataset: <https://huggingface.co/datasets/NagaYu/deference-keigo-corpus>

## Source

Adapted from the Council for Cultural Affairs' report *Keigo no Shishin*
(敬語の指針), Agency for Cultural Affairs, 2007.
<https://www.bunka.go.jp/seisaku/bunkashingikai/kokugo/hokoku/pdf/keigo_tosin.pdf>

Content on the Agency's site follows the
[MEXT website terms of use](https://www.mext.go.jp/b_menu/1351168.htm), stated to
be compatible with CC BY and permitting adaptation provided the source is
credited. Only short quotations needed as the basis for each note are kept here;
the report is not redistributed in full.
"""

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deference — the direction of deference in Japanese keigo</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header>
  <h1>Deference</h1>
  <p class="tagline">The direction of deference, with the source to back it up.</p>
  <p class="lede">
    A rule-based Japanese linter says nothing about <b lang="ja">「お持ちします」</b>.
    As a string it is fine. But <b>if the reader is the one carrying something,
    that form does not raise them</b> &mdash; it raises whoever the act is directed
    to, which here is the writer.
  </p>
  <p class="lede">
    Deference decides from <i>whose action it is</i>, <i>who it is directed to</i>,
    and <i>whether the message goes inside or outside your company</i>, then cites
    the page of the Council for Cultural Affairs&rsquo;
    <b>Keigo no Shishin</b> (<span lang="ja">敬語の指針</span>, 2007) the judgement
    rests on.
  </p>
  <nav class="links">
    <a href="https://github.com/NagaYu/deference">Code &amp; evaluation</a>
    <a href="https://huggingface.co/NagaYu/deference-keigo">Model</a>
    <a href="https://huggingface.co/datasets/NagaYu/deference-keigo-corpus">Dataset</a>
  </nav>
</header>

<section class="demo">
  <h2>Real output, computed ahead of time</h2>
  <p class="note">
    This page is a <b>static</b> Space, so the model does not run in your browser.
    Everything below is genuine output from the published model, recorded in
    advance. To check your own text, see <a href="#run">Run it yourself</a>.
  </p>

  <div class="controls">
    <div class="group" id="example-buttons" role="group" aria-label="Example"></div>
    <div class="group" id="audience-buttons" role="group" aria-label="Audience"></div>
  </div>

  <p class="blurb" id="blurb"></p>

  <div class="panes">
    <div class="pane">
      <h3>Message</h3>
      <pre class="body" id="body" lang="ja"></pre>
      <p class="legend">
        <span class="chip direction">needs context</span>
        <span class="chip form">surface form</span>
        <span class="chip variation">variation — not reported</span>
      </p>
      <p class="timing" id="timing"></p>
    </div>
    <div class="pane">
      <h3>Notes from the guidelines</h3>
      <div id="findings"></div>
    </div>
  </div>
</section>

<section>
  <h2>Where the difference is, and where it is not</h2>
  <p>
    On error types that <b>cannot be decided without context</b>, textlint and a
    surface-rule set both score <b>0.0%</b>. On types that <b>are visible in the
    surface pattern</b>, the rule set also reaches 100% &mdash; and no advantage is
    claimed there.
  </p>
  <figure>
    <img src="figures/fig1_detection_by_error_type.png"
         alt="Detection rate by error type. On the four context-dependent types, textlint and the surface rule set are at zero while Deference reaches 86 to 100 percent. On the four surface-visible types, the rule set and Deference are both at 100 percent.">
  </figure>
  <div id="summary-table"></div>
  <figure>
    <img src="figures/fig2_detection_vs_overflag.png"
         alt="Detection rate against over-flagging rate. Deference sits near 98 percent detection with about 1.5 percent over-flagging; the surface rule set is at 43 percent detection; textlint is at 1 percent detection with 9.7 percent over-flagging.">
    <figcaption>
      Upper-left is better. textlint detects 1% while over-flagging 9.7% &mdash; its
      punctuation and sentence-length rules fire on perfectly correct text.
    </figcaption>
  </figure>
</section>

<section>
  <h2>Variation is not error</h2>
  <p>
    Painting every divided usage red destroys trust in a checker. The guidelines
    say as much themselves:
  </p>
  <blockquote lang="ja">
    男女の違いや世代の違いなどによって画一的に考える態度は避けるべきである。
    <span class="gloss">(One should avoid treating usage uniformly by gender or
    generation.) &mdash; Ch.1 Sec.2-2, p.8</span>
  </blockquote>
  <p>
    53 such cases are held apart and reported as <b>none of them</b> by default:
    doubled honorifics the report calls established by custom
    (<span lang="ja">お伺いする</span>), forms it calls equally usable
    (<span lang="ja">ご利用いただく ↔ ご利用くださる</span>), and usages whose
    tolerance it says varies by individual
    (<span lang="ja">させていただく</span>).
  </p>
</section>

<section>
  <h2>It does not tell you your Japanese is wrong</h2>
  <p>
    Output stays informational &mdash; how the guidelines organise the matter, not a
    verdict. Tests enforce this in both languages, checking that phrases like
    &ldquo;is wrong&rdquo; or <span lang="ja">「誤用」</span> never reach the user.
    Every run closes with:
  </p>
  <blockquote>
    Honorifics are chosen as a form of self-expression (Keigo no Shishin,
    Ch.1 Sec.1-3). What is shown here is how the guidelines organise the matter;
    it is not a judgement on the writer&rsquo;s Japanese.
  </blockquote>
</section>

<section id="run">
  <h2>Run it yourself</h2>
<pre class="code"><code>pip install git+https://github.com/NagaYu/deference

deference check mail.txt --audience external
deference check mail.txt --audience internal   # the judgement changes</code></pre>
<pre class="code"><code>from deference import Deference, MailContext, Audience

result = Deference().check(open("mail.txt").read(), MailContext(audience=Audience.EXTERNAL))
for f in result.reportable: print(f.span.text, f.message, f.citation.render("en"))</code></pre>
  <p class="note">
    The model is <b>277.5M parameters</b> and runs on CPU in about 26&nbsp;ms per
    message after int8 quantisation. It works without the trained checkpoint too,
    falling back to the rule engine.
  </p>
</section>

<footer>
  <h2>Source</h2>
  <p>
    Adapted from the Council for Cultural Affairs&rsquo; report
    <i>Keigo no Shishin</i> (<span lang="ja">敬語の指針</span>), Agency for
    Cultural Affairs, 2007.
    <a href="https://www.bunka.go.jp/seisaku/bunkashingikai/kokugo/hokoku/pdf/keigo_tosin.pdf">Original PDF</a>
  </p>
  <p>
    Content on the Agency&rsquo;s site follows the
    <a href="https://www.mext.go.jp/b_menu/1351168.htm">MEXT website terms of use</a>,
    stated to be compatible with CC BY and permitting adaptation provided the
    source is credited; where the material has been edited or adapted, that must
    be stated &mdash; and it is. This project keeps <b>only the short quotations
    needed as the basis for each note</b> and does <b>not</b> redistribute the
    report in full.
  </p>
  <p class="fineprint">
    Code Apache-2.0. Example messages are invented for illustration and do not
    depict real people or correspondence.
  </p>
</footer>

<script src="app.js"></script>
</body>
</html>
"""

_STYLE_CSS = """:root {
  color-scheme: light dark;
  --bg: #fbfaf8;
  --fg: #1d2321;
  --muted: #5d6b66;
  --line: #e2e0da;
  --card: #ffffff;
  --accent: #2d6a4f;
  --direction: #e26d5c;
  --form: #f2a65a;
  --variation: #b9bfc4;
  --maxw: 62rem;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14181a;
    --fg: #e8eceb;
    --muted: #9aa8a3;
    --line: #2a3134;
    --card: #1b2124;
    --accent: #74c69d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.7 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Hiragino Sans", "Noto Sans JP", sans-serif;
  padding: 0 1.2rem 5rem;
}
header, section, footer { max-width: var(--maxw); margin: 0 auto; }
header { padding: 3.2rem 0 2rem; border-bottom: 1px solid var(--line); }
h1 { font-size: clamp(2.4rem, 6vw, 3.4rem); margin: 0; letter-spacing: -0.02em; }
.tagline { font-size: 1.2rem; color: var(--accent); font-weight: 600; margin: .4rem 0 1.4rem; }
.lede { max-width: 46rem; color: var(--fg); margin: 0 0 .9rem; }
.links { display: flex; flex-wrap: wrap; gap: .6rem; margin-top: 1.4rem; }
.links a {
  border: 1px solid var(--line); border-radius: 999px; padding: .4rem .95rem;
  text-decoration: none; color: var(--fg); font-size: .92rem; background: var(--card);
}
.links a:hover { border-color: var(--accent); color: var(--accent); }
section { padding: 2.6rem 0; border-bottom: 1px solid var(--line); }
h2 { font-size: 1.45rem; margin: 0 0 .9rem; letter-spacing: -0.01em; }
h3 { font-size: .82rem; text-transform: uppercase; letter-spacing: .09em;
     color: var(--muted); margin: 0 0 .7rem; }
p { margin: 0 0 .9rem; }
.note, .fineprint { color: var(--muted); font-size: .92rem; }
.controls { display: flex; flex-wrap: wrap; gap: 1.2rem; margin: 1.4rem 0 .6rem; }
.group { display: flex; flex-wrap: wrap; gap: .4rem; }
button {
  font: inherit; font-size: .9rem; padding: .42rem .9rem; cursor: pointer;
  background: var(--card); color: var(--fg);
  border: 1px solid var(--line); border-radius: 8px;
}
button:hover { border-color: var(--accent); }
button[aria-pressed="true"] {
  background: var(--accent); border-color: var(--accent); color: #fff;
}
.blurb { color: var(--muted); font-size: .95rem; margin: .3rem 0 1.1rem; }
.panes { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
@media (max-width: 800px) { .panes { grid-template-columns: 1fr; } }
.pane { background: var(--card); border: 1px solid var(--line);
        border-radius: 12px; padding: 1.1rem 1.2rem; min-width: 0; }
pre.body {
  white-space: pre-wrap; word-break: break-word; margin: 0;
  font: inherit; line-height: 2.05;
}
mark { padding: .1em .12em; border-radius: 4px; color: inherit;
       background: none; box-shadow: inset 0 -0.62em 0 var(--hl); }
mark.direction { --hl: rgba(226,109,92,.42); }
mark.form { --hl: rgba(242,166,90,.42); }
mark.variation { --hl: rgba(185,191,196,.42); }
.legend { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0 .2rem; }
.chip { font-size: .78rem; padding: .16rem .55rem; border-radius: 999px;
        border: 1px solid var(--line); color: var(--muted); }
.chip.direction { border-color: var(--direction); color: var(--direction); }
.chip.form { border-color: var(--form); }
.chip.variation { border-color: var(--variation); }
.timing { color: var(--muted); font-size: .84rem; margin: .5rem 0 0; }
.finding { border-top: 1px solid var(--line); padding: .95rem 0; }
.finding:first-child { border-top: none; padding-top: 0; }
.finding .head { display: flex; flex-wrap: wrap; gap: .5rem; align-items: baseline; }
.finding .surface { font-weight: 600; }
.finding .type { font-size: .8rem; color: var(--muted); }
.finding .type.ctx { color: var(--direction); font-weight: 600; }
.finding .msg { font-size: .92rem; margin: .45rem 0; }
.finding .alt { font-size: .9rem; }
.finding .alt b { color: var(--accent); }
.finding .cite { font-size: .82rem; color: var(--muted); margin-top: .35rem; }
.finding .cite a { color: var(--muted); }
.quiet { color: var(--muted); }
.ok { color: var(--accent); font-weight: 600; }
figure { margin: 1.4rem 0; }
figure img { width: 100%; height: auto; border: 1px solid var(--line);
             border-radius: 10px; background: #fff; }
figcaption { color: var(--muted); font-size: .88rem; margin-top: .55rem; }
blockquote {
  margin: 1rem 0; padding: .7rem 0 .7rem 1.1rem;
  border-left: 3px solid var(--accent); color: var(--fg);
}
blockquote .gloss { display: block; color: var(--muted); font-size: .88rem; margin-top: .4rem; }
pre.code {
  background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: .9rem 1.1rem; overflow-x: auto; font-size: .88rem; line-height: 1.6;
}
table { border-collapse: collapse; width: 100%; font-size: .9rem; margin: 1rem 0; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: .82rem; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.hl td { font-weight: 600; }
footer { padding-top: 2.4rem; }
"""

_APP_JS = """(async function () {
  const res = await fetch("data.json");
  const data = await res.json();
  let exIndex = 0;
  let audience = "external";

  const exBar = document.getElementById("example-buttons");
  const audBar = document.getElementById("audience-buttons");

  data.examples.forEach((ex, i) => {
    const b = document.createElement("button");
    b.textContent = ex.title;
    b.addEventListener("click", () => { exIndex = i; render(); });
    exBar.appendChild(b);
  });
  [["external", "Addressed outside the company"],
   ["internal", "Addressed inside the company"]].forEach(([key, label]) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.dataset.aud = key;
    b.addEventListener("click", () => { audience = key; render(); });
    audBar.appendChild(b);
  });

  function render() {
    const ex = data.examples[exIndex];
    const run = ex.runs[audience];
    [...exBar.children].forEach((b, i) =>
      b.setAttribute("aria-pressed", String(i === exIndex)));
    [...audBar.children].forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.aud === audience)));

    document.getElementById("blurb").textContent = ex.blurb;

    const body = document.getElementById("body");
    body.textContent = "";
    run.segments.forEach((seg) => {
      if (!seg.kind) { body.appendChild(document.createTextNode(seg.t)); return; }
      const m = document.createElement("mark");
      m.className = seg.kind;
      m.textContent = seg.t;
      body.appendChild(m);
    });

    const bits = [`${run.elapsed_ms} ms on CPU`];
    if (run.textlint_available) {
      bits.push(`textlint found ${run.textlint_count}`);
    }
    document.getElementById("timing").textContent = bits.join(" · ");

    const box = document.getElementById("findings");
    box.textContent = "";
    if (!run.findings.length) {
      const p = document.createElement("p");
      p.className = "ok";
      p.textContent = "Nothing stood out.";
      box.appendChild(p);
    }
    run.findings.forEach((f) => {
      const d = document.createElement("div");
      d.className = "finding";

      const head = document.createElement("div");
      head.className = "head";
      const s = document.createElement("span");
      s.className = "surface";
      s.lang = "ja";
      s.textContent = "\\u300c" + f.text + "\\u300d";
      const t = document.createElement("span");
      t.className = "type" + (f.needs_context ? " ctx" : "");
      t.textContent = f.type + (f.needs_context ? " · needs context" : "");
      head.append(s, t);
      d.appendChild(head);

      if (f.message) {
        const m = document.createElement("p");
        m.className = "msg";
        m.textContent = f.message;
        d.appendChild(m);
      }
      if (f.suggestions && f.suggestions.length) {
        const a = document.createElement("p");
        a.className = "alt";
        const b = document.createElement("b");
        b.textContent = "Alternative forms: ";
        const v = document.createElement("span");
        v.lang = "ja";
        v.textContent = f.suggestions.join(" / ");
        a.append(b, v);
        d.appendChild(a);
      }
      if (f.citation) {
        const c = document.createElement("p");
        c.className = "cite";
        const label = f.citation.source + " " + f.citation.section +
                      " (" + f.citation.page + ")";
        if (f.citation.url) {
          const a = document.createElement("a");
          a.href = f.citation.url;
          a.textContent = label;
          c.appendChild(a);
        } else {
          c.textContent = label;
        }
        d.appendChild(c);
      }
      box.appendChild(d);
    });

    if (run.variations.length) {
      const p = document.createElement("p");
      p.className = "quiet";
      p.style.marginTop = "1rem";
      p.style.fontSize = ".88rem";
      p.textContent =
        run.variations.length + " passage(s) were treated as variation and " +
        "are not reported: " +
        run.variations.map((v) => "\\u300c" + v.text + "\\u300d").join(" ");
      box.appendChild(p);
    }
  }

  // --- benchmark table -----------------------------------------------------
  const bench = data.benchmarks;
  if (bench && bench.rows) {
    const host = document.getElementById("summary-table");
    const table = document.createElement("table");
    table.innerHTML =
      "<thead><tr><th>condition</th><th>detection (type)</th>" +
      "<th>direction errors</th><th>no-over-flag</th><th>median</th></tr></thead>";
    const tb = document.createElement("tbody");
    bench.rows.forEach((r) => {
      const tr = document.createElement("tr");
      if (r.label.indexOf("Deference") >= 0) tr.className = "hl";
      if (r.skipped) {
        const td = document.createElement("td");
        td.textContent = r.label;
        const td2 = document.createElement("td");
        td2.colSpan = 4;
        td2.className = "quiet";
        td2.textContent = "not measured — " + r.skipped;
        tr.append(td, td2);
      } else {
        const cells = [
          r.label,
          r.type.toFixed(1) + "%",
          r.direction.toFixed(1) + "%",
          r.novar.toFixed(1) + "%",
          r.ms.toFixed(2) + " ms",
        ];
        cells.forEach((v, i) => {
          const td = document.createElement("td");
          if (i > 0) td.className = "num";
          td.textContent = v;
          tr.appendChild(td);
        });
      }
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    host.appendChild(table);
  }

  render();
})();
"""


if __name__ == "__main__":
    raise SystemExit(main())
