"""Deference — Gradio Space.

Paste a Japanese message body, choose whether it is addressed inside or outside
your company, and the app highlights each passage, shows the relevant section of
the Keigo no Shishin, and offers alternative forms. The textlint result is shown
side by side so you can see what a surface-pattern linter does and does not reach.

文言の方針（製品要件）:
    利用者の日本語を否定しない。見出しは「エラー」ではなく「規範上の案内」
    （英語では "Notes from the guidelines"）とし、揺れは既定で指摘しない。
    根拠は指針 第1章第1-3「『自己表現』としての敬語使用」。

実証する主張との対応:
    - 「向きの誤り検出」: 宛先のラジオを切り替えると結果が変わることを、
      利用者がその場で確かめられる。
    - 「過剰指摘の少なさ」: 揺れは別枠・控えめな色で、既定では指摘しない。
    - 「根拠提示」: 各指摘に指針の章・ページ・リンクを添える。
    - 「速度」: 応答時間をミリ秒で常時表示する。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from deference import norms
from deference.baselines import TextlintBaseline
from deference.cite import KEIGO_CLASS_DEFINITION_EN, RuleCitation
from deference.pipeline import Deference
from deference.types import (
    Audience,
    CheckResult,
    ErrorType,
    Finding,
    KeigoClass,
    MailContext,
    Party,
    Person,
    Verdict,
    ERROR_TYPE_JA,
    KEIGO_CLASS_EN,
    KEIGO_CLASS_JA,
    error_type_name,
)

#: 学習済みチェックポイント。ローカルに無ければ Hugging Face Hub から読む。
#: Space には 1.1GB の重みを同梱せず、Hub のモデルリポジトリを参照する。
_LOCAL_MODEL = Path("checkpoints/deference-base")
_HUB_MODEL = os.environ.get("DEFERENCE_MODEL", "NagaYu/deference-keigo")
MODEL_DIR = _LOCAL_MODEL if _LOCAL_MODEL.exists() else _HUB_MODEL
_HAS_MODEL = True

_engines: Dict[str, Deference] = {}
_textlint = TextlintBaseline()


def _engine(lang: str) -> Deference:
    """言語ごとに検査器を用意する（モデルは1度だけ読み込まれる）。"""
    if lang not in _engines:
        try:
            _engines[lang] = Deference(
                engine="hybrid",
                model_dir=MODEL_DIR,
                report_variations=True,
                lang=lang,
            )
            _engines[lang].model  # 起動時に読み込み、初回リクエストを待たせない
        except Exception as exc:  # noqa: BLE001
            # モデルが取れなくても規範エンジンだけで動く。
            # 「学習済みチェックポイントが無くても必ず動く」という設計の要。
            print(f"model unavailable, falling back to the rule engine: {exc}")
            _engines[lang] = Deference(
                engine="norm", report_variations=True, lang=lang
            )
    return _engines[lang]


AUDIENCE_CHOICES = {
    "External (a customer or another company)": Audience.EXTERNAL,
    "Internal (within your own organisation)": Audience.INTERNAL,
    "Public notice": Audience.PUBLIC,
}

#: 色分けの分類。向きの誤りは目立つ色、揺れは控えめなグレー。
CATEGORY = {
    "direction": "Direction of deference",
    "form": "Honorific form",
    "style": "Style / usage",
    "variation": "Variation (not reported)",
    "textlint": "textlint",
}

COLOR_MAP: Dict[str, str] = {
    CATEGORY["direction"]: "#e26d5c",
    CATEGORY["form"]: "#f2a65a",
    CATEGORY["style"]: "#7fb3d5",
    CATEGORY["variation"]: "#c8ccd0",
    CATEGORY["textlint"]: "#b8a1d9",
}

DISCLAIMER = """
This tool reports how the Council for Cultural Affairs' **Keigo no Shishin**
(敬語の指針, 2007) frames each passage. Honorifics are chosen as a form of
self-expression (Ch.1 Sec.1-3), so what you see here is **information, not a
verdict on your Japanese**.

Expressions whose acceptability varies by situation or generation are kept apart
as **variation** and are not reported by default — the guidelines themselves warn
against treating such usage uniformly (Ch.1 Sec.2-2, p.8:
「男女の違いや世代の違いなどによって画一的に考える態度は避けるべきである。」).
"""


def _category(f: Finding) -> str:
    if f.verdict is Verdict.VARIATION:
        return CATEGORY["variation"]
    if f.error_type.requires_context:
        return CATEGORY["direction"]
    if f.error_type in (ErrorType.STYLE_MIXING, ErrorType.SASETE_ITADAKU_OVERUSE):
        return CATEGORY["style"]
    return CATEGORY["form"]


def _highlight(
    text: str, findings: Sequence[Finding], *, label: Optional[str] = None
) -> List[Tuple[str, Optional[str]]]:
    """gr.HighlightedText 用のタプル列に変換する。"""
    out: List[Tuple[str, Optional[str]]] = []
    pos = 0
    for f in sorted(findings, key=lambda x: (x.span.start, x.span.end)):
        if f.span.start < pos:
            continue
        if f.span.start > pos:
            out.append((text[pos : f.span.start], None))
        out.append((f.span.text, label or _category(f)))
        pos = f.span.end
    if pos < len(text):
        out.append((text[pos:], None))
    return out or [(text, None)]


def _build_context(
    audience_label: str, writer_org: str, recipient_org: str, people: str
) -> MailContext:
    """UI の入力から MailContext を組み立てる。

    ``people`` は「佐藤:self_group, 田中:addressee」のような一行指定。
    誰が自分側で誰が相手側かが、敬意の向きの判定を決める。
    """
    aliases = {
        "self": Party.SELF,
        "self_group": Party.SELF_GROUP,
        "addressee": Party.ADDRESSEE,
        "addressee_group": Party.ADDRESSEE_GROUP,
        "third_party": Party.THIRD_PARTY,
    }
    persons: List[Person] = []
    for chunk in (people or "").replace("、", ",").split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, side = chunk.rsplit(":", 1)
        party = aliases.get(side.strip().lower())
        if party is not None and name.strip():
            persons.append(Person(name.strip(), party))
    return MailContext(
        audience=AUDIENCE_CHOICES.get(audience_label, Audience.EXTERNAL),
        writer_org=writer_org.strip() or "弊社",
        recipient_org=recipient_org.strip() or "貴社",
        persons=tuple(persons),
    )


def _findings_markdown(result: CheckResult, show_variations: bool, lang: str) -> str:
    lines: List[str] = []
    reportable = result.reportable
    if not reportable:
        lines.append("### Nothing stood out\n")
    else:
        lines.append(f"### Notes from the guidelines ({len(reportable)})\n")

    for i, f in enumerate(reportable, 1):
        tag = error_type_name(f.error_type, lang)
        ctx_mark = " · **context-dependent**" if f.error_type.requires_context else ""
        lines.append(f"**{i}. 「{f.span.text}」** — {tag}{ctx_mark}")
        lines.append("")
        if f.message:
            lines.append(f.message)
            lines.append("")
        if f.suggestions:
            forms = " / ".join(dict.fromkeys(s.text for s in f.suggestions[:5]))
            lines.append(f"- **Alternative forms**: 「{forms}」")
        if f.citation:
            src = (
                f"{f.citation.source_label(lang)} "
                f"{f.citation.section_label(lang)} ({f.citation.page})"
            )
            if f.citation.url:
                lines.append(f"- **Basis**: [{src}]({f.citation.url})")
            else:
                lines.append(f"- **Basis**: {src}")
        lines.append("")

    variations = result.variations
    if variations:
        lines.append("---")
        lines.append(
            f"### Treated as variation ({len(variations)}) — not reported as issues\n"
        )
        if show_variations:
            for f in variations:
                lines.append(f"- 「{f.span.text}」 — {f.message}")
        else:
            lines.append(
                "Acceptability of these is divided by situation or generation. "
                "Tick *Show variation* to see them."
            )
        lines.append("")
    lines.append(
        f"<sub>{result.elapsed_ms:.1f} ms on CPU ({result.engine})</sub>"
    )
    return "\n".join(lines)


def _textlint_markdown(result: CheckResult) -> str:
    reason = result.meta.get("unavailable_reason")
    if reason:
        return (
            "### textlint is not available here\n\n"
            f"{reason}\n\n"
            "Run `cd .baseline_textlint && npm install` to enable the comparison."
        )
    n = len(result.findings)
    lines = [f"### textlint findings ({n})\n"]
    for f in result.findings:
        lines.append(
            f"- 「{f.span.text}」 — {f.message}  \n"
            f"  <sub>{f.meta.get('ruleId','')}</sub>"
        )
    if not result.findings:
        lines.append("- none")
    lines.append("")
    lines.append(
        "<sub>textlint's Japanese presets target technical-writing conventions "
        "(sentence length, double negatives, ra-nuki, polite/plain consistency). "
        "**There is no keigo-specific rule in the ecosystem**, so the direction of "
        "deference is simply outside what it was built to check.</sub>"
    )
    lines.append(f"\n<sub>{result.elapsed_ms:.0f} ms</sub>")
    return "\n".join(lines)


def _comparison_markdown(df_result: CheckResult, tl_result: CheckResult, lang: str) -> str:
    """両者が何を拾い、何を拾わなかったかの対照表。"""
    if tl_result.meta.get("unavailable_reason"):
        return "textlint is not available, so the side-by-side view is disabled."

    df_hits = df_result.reportable
    tl_hits = list(tl_result.findings)
    only_df = [f for f in df_hits if not any(f.span.overlaps(t.span) for t in tl_hits)]
    only_tl = [t for t in tl_hits if not any(t.span.overlaps(f.span) for f in df_hits)]
    both = [f for f in df_hits if any(f.span.overlaps(t.span) for t in tl_hits)]

    lines = ["### What each tool reached\n"]
    lines.append(f"#### Only Deference ({len(only_df)})\n")
    for f in only_df:
        mark = " **[direction of deference]**" if f.error_type.requires_context else ""
        lines.append(f"- 「{f.span.text}」 — {error_type_name(f.error_type, lang)}{mark}")
    if not only_df:
        lines.append("- none")
    lines.append("")
    lines.append(f"#### Only textlint ({len(only_tl)})\n")
    for t in only_tl:
        lines.append(f"- 「{t.span.text}」 — {t.message}")
    if not only_tl:
        lines.append("- none")
    lines.append("")
    lines.append(f"#### Both ({len(both)})\n")
    for f in both:
        lines.append(f"- 「{f.span.text}」")
    if not both:
        lines.append("- none")
    lines.append("")
    n_ctx = sum(1 for f in only_df if f.error_type.requires_context)
    if n_ctx:
        lines.append(
            f"> Of the {len(only_df)} passages only Deference reached, **{n_ctx}** "
            "cannot be judged without knowing whose action it is and who the message "
            "is addressed to. A tool that looks only at surface patterns has no way "
            "to get there."
        )
    return "\n".join(lines)


def _teach_markdown(lang: str) -> str:
    citer = RuleCitation(lang)
    names = KEIGO_CLASS_EN if lang == "en" else KEIGO_CLASS_JA
    defs = KEIGO_CLASS_DEFINITION_EN if lang == "en" else norms.KEIGO_CLASS_DEFINITION
    lines = ["## The five classes of Japanese honorifics\n"]
    lines.append("*Keigo no Shishin*, Ch.2 Sec.1\n")
    for cls in (
        KeigoClass.SONKEIGO,
        KeigoClass.KENJOUGO_1,
        KeigoClass.KENJOUGO_2,
        KeigoClass.TEINEIGO,
        KeigoClass.BIKAGO,
    ):
        lines.append(f"**{names[cls]}** — {defs.get(cls, '')}\n")
        if lang == "en":
            lines.append(f"<sub>原文: {norms.KEIGO_CLASS_DEFINITION.get(cls,'')}</sub>\n")
    lines.append("---\n")
    lines.append("## Error types\n")
    for et in ErrorType:
        if et is ErrorType.NONE:
            continue
        mark = " ★ needs context" if et.requires_context else ""
        lines.append(f"### {error_type_name(et, lang)}{mark}")
        lines.append(f"<sub>{ERROR_TYPE_JA[et]} · `{et.value}`</sub>\n")
        lines.append(citer.teach(et))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def analyse(
    text: str,
    audience_label: str,
    writer_org: str,
    recipient_org: str,
    people: str,
    options: Sequence[str],
    lang_label: str,
):
    """UI のメインハンドラ。"""
    lang = "ja" if str(lang_label).startswith("日本") else "en"
    show_variations = any("ariation" in o or "揺れ" in o for o in (options or []))
    text = text or ""
    ctx = _build_context(audience_label, writer_org, recipient_org, people)

    df_result = _engine(lang).check(text, ctx)
    tl_result = _textlint.check(text, ctx)

    shown = list(df_result.reportable) + (
        list(df_result.variations) if show_variations else []
    )
    return (
        _highlight(text, shown),
        _findings_markdown(df_result, show_variations, lang),
        _highlight(text, tl_result.findings, label=CATEGORY["textlint"]),
        _textlint_markdown(tl_result),
        _comparison_markdown(df_result, tl_result, lang),
        f"**{df_result.elapsed_ms:.1f} ms** on CPU ({df_result.engine})",
        _teach_markdown(lang),
    )


_EX_ERRORS = (
    "田中様\n\nいつもお世話になっております。株式会社アルファの山田です。\n\n"
    "弊社の佐藤社長が、大変参考になったとおっしゃっておりました。\n"
    "なお、当日は私が資料をお持ちになります。\n"
    "恐れ入りますが、担当者に伺ってください。\n\n"
    "よろしくお願いいたします。\n"
)
_EX_CLEAN = (
    "田中様\n\nいつもお世話になっております。株式会社アルファの山田です。\n\n"
    "弊社の佐藤が、大変参考になったと申しておりました。\n"
    "なお、当日は私が資料をお持ちします。\n"
    "恐れ入りますが、担当者にお尋ねください。\n\n"
    "よろしくお願いいたします。\n"
)
_EX_VARIATION = (
    "佐藤様\n\n本日はお忙しいところ、お伺いいたします。\n"
    "資料はご持参くださいますようお願い申し上げます。\n"
    "いつもご利用いただきまして、ありがとうございます。\n"
)
_EX_AUDIENCE = (
    "各位\n\nいつもお世話になっております。\n"
    "弊社の佐藤社長が、そのようにおっしゃっておりました。\n"
    "よろしくお願いいたします。\n"
)

EXAMPLES = [
    [_EX_ERRORS, "External (a customer or another company)", "株式会社アルファ",
     "株式会社ベータ", "佐藤:self_group, 田中:addressee", [], "English"],
    [_EX_CLEAN, "External (a customer or another company)", "株式会社アルファ",
     "株式会社ベータ", "佐藤:self_group, 田中:addressee", [], "English"],
    [_EX_VARIATION, "External (a customer or another company)", "弊社", "貴社",
     "佐藤:addressee", ["Show variation"], "English"],
    [_EX_AUDIENCE, "Internal (within your own organisation)", "株式会社アルファ",
     "株式会社アルファ", "佐藤:self_group", ["Show variation"], "English"],
]


def build_demo() -> "gr.Blocks":
    with gr.Blocks(
        title="Deference — Japanese honorific checker",
        theme=gr.themes.Soft(primary_hue="emerald"),
    ) as demo:
        gr.Markdown("# Deference")
        gr.Markdown(
            "**The direction of deference, with the source to back it up.**  \n"
            "A rule-based linter says nothing about 「お持ちします」 — as a string it is "
            "fine. But **if the reader is the one carrying, that form does not raise "
            "them.** Deference judges from whose action it is, who it is directed to, "
            "and whether the message goes inside or outside your company."
        )
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=4):
                text = gr.Textbox(
                    label="Message body (Japanese)",
                    lines=14,
                    placeholder="Paste the body of a Japanese email here.",
                )
                audience = gr.Radio(
                    list(AUDIENCE_CHOICES),
                    value="External (a customer or another company)",
                    label="Audience",
                    info=(
                        "The same text can be judged differently depending on this "
                        "(Keigo no Shishin, Ch.3 Sec.3-2 Q25)."
                    ),
                )
                with gr.Row():
                    writer_org = gr.Textbox(label="Your organisation", value="弊社", scale=1)
                    recipient_org = gr.Textbox(label="Reader's organisation", value="貴社", scale=1)
                people = gr.Textbox(
                    label="People in the text",
                    value="",
                    placeholder="佐藤:self_group, 田中:addressee",
                    info=(
                        "name:side — self / self_group / addressee / addressee_group "
                        "/ third_party. This is what makes the direction decidable."
                    ),
                )
                with gr.Row():
                    options = gr.CheckboxGroup(
                        ["Show variation"], value=[], label="Display", scale=2
                    )
                    lang = gr.Radio(
                        ["English", "日本語"], value="English", label="Language", scale=1
                    )
                run = gr.Button("Check", variant="primary")
                timing = gr.Markdown("")

            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab("Deference"):
                        df_hl = gr.HighlightedText(
                            label="Message",
                            color_map=COLOR_MAP,
                            combine_adjacent=True,
                            show_legend=True,
                        )
                        df_md = gr.Markdown()
                    with gr.Tab("textlint"):
                        tl_hl = gr.HighlightedText(
                            label="Message", color_map=COLOR_MAP, combine_adjacent=True
                        )
                        tl_md = gr.Markdown()
                    with gr.Tab("Side by side"):
                        cmp_md = gr.Markdown()
                    with gr.Tab("How keigo works"):
                        teach_md = gr.Markdown(_teach_markdown("en"))

        gr.Examples(
            examples=EXAMPLES,
            inputs=[text, audience, writer_org, recipient_org, people, options, lang],
            label=(
                "Examples — with issues / the same message in line with the guidelines "
                "/ variation only / judged differently by audience"
            ),
        )
        gr.Markdown("---")
        gr.Markdown(
            "### Source\n\n"
            "Adapted from the Council for Cultural Affairs' report "
            "*Keigo no Shishin* (敬語の指針), Agency for Cultural Affairs, 2007.  \n"
            "<https://www.bunka.go.jp/seisaku/bunkashingikai/kokugo/hokoku/pdf/keigo_tosin.pdf>\n\n"
            "Content on the Agency's site follows the "
            "[MEXT website terms of use](https://www.mext.go.jp/b_menu/1351168.htm), "
            "which are stated to be compatible with CC BY and permit reproduction and "
            "adaptation provided the source is credited; where the material has been "
            "edited or adapted, that must be stated. This project keeps **only short "
            "quotations needed as the basis for each note** and does **not** "
            "redistribute the report in full."
        )

        inputs = [text, audience, writer_org, recipient_org, people, options, lang]
        outputs = [df_hl, df_md, tl_hl, tl_md, cmp_md, timing, teach_md]
        run.click(analyse, inputs=inputs, outputs=outputs)
        text.submit(analyse, inputs=inputs, outputs=outputs)
    return demo


if __name__ == "__main__":
    build_demo().launch()
