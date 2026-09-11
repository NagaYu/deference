"""deference コマンドライン。

    deference check mail.txt --audience external

実証する主張との対応:
    - 「速度」: 既定エンジンは規範ベースで、torch を読み込まない。
      応答時間を毎回表示するので、利用者が体感と数値を突き合わせられる。
    - 「根拠提示」: 各指摘に指針の章・ページ・URL を必ず添える。
    - 「過剰指摘の少なさ」: 揺れは既定で出さず、``--show-variations`` のときだけ
      控えめな色で別枠に出す。
    - 文言: 「エラー」ではなく「規範上の案内」。利用者の日本語を否定しない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import norms
from .types import (
    Audience,
    CheckResult,
    ErrorType,
    Finding,
    MailContext,
    Party,
    Person,
    Verdict,
    ERROR_TYPE_JA,
    KEIGO_CLASS_EN,
    KEIGO_CLASS_JA,
    error_type_name,
    to_jsonable,
)

__all__ = ["main", "build_parser"]


# ---------------------------------------------------------------------------
# 色
# ---------------------------------------------------------------------------

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "under": "\033[4m",
    "red": "\033[38;5;203m",
    "orange": "\033[38;5;215m",
    "yellow": "\033[38;5;222m",
    "blue": "\033[38;5;111m",
    "green": "\033[38;5;114m",
    "grey": "\033[38;5;245m",
}

#: 誤り種別ごとの色。向きの誤りは目立つ色、それ以外は落ち着いた色にする。
_TYPE_COLOR = {
    ErrorType.DIRECTION_SWAP: "red",
    ErrorType.UCHI_SONKEIGO: "red",
    ErrorType.SELF_SONKEIGO: "red",
    ErrorType.DEFERENCE_INVERSION: "red",
    ErrorType.BAD_KEIGO_LINK: "orange",
    ErrorType.OGO_DEKIRU: "orange",
    ErrorType.GO_SARERU: "orange",
    ErrorType.DOUBLE_KEIGO: "yellow",
    ErrorType.SA_INSERTION: "yellow",
    ErrorType.SASETE_ITADAKU_OVERUSE: "blue",
    ErrorType.STYLE_MIXING: "blue",
}


#: UI 文言のカタログ。英語を既定にしつつ、このツールが実際に役立つ相手である
#: 日本語話者のために日本語も残す（--lang ja）。
_MSG = {
    "en": {
        "heading": "Notes from the guidelines (audience: {aud})",
        "none": "  Nothing stood out.",
        "alt": "Alternative forms: ",
        "basis": "Basis: ",
        "variations_heading": "Treated as variation (not reported as issues)",
        "variations_hint": (
            "  ({n} passage(s) where acceptability is divided were treated as "
            "variation. Use --show-variations to see them.)"
        ),
        "elapsed": "  {ms:.1f} ms ({engine})",
        "audience": {"external": "external", "internal": "internal", "public": "public notice"},
        "unknown_verb": (
            "Not in the verb lexicon: {verb!r}\nVerbs included: {sample} ..."
        ),
        "forms_title": "Honorific forms for 「{verb}」 (audience: {aud})",
        "forms_none": "    (no form can be built for this standpoint)",
        "sides": {
            "自分 → 相手": "writer → reader",
            "身内 → 相手": "writer's side → reader",
            "相手": "reader",
            "第三者": "third party",
        },
        "explain_unknown": "Could not identify the error type. Choose one of:",
        "related": "Honorific classes involved:",
    },
    "ja": {
        "heading": "規範上の案内（宛先: {aud}）",
        "none": "  気づいた点はありませんでした。",
        "alt": "別の形の候補: ",
        "basis": "根拠: ",
        "variations_heading": "揺れとして扱った箇所（指摘ではありません）",
        "variations_hint": (
            "  （許容度が分かれる表現を {n} 箇所、揺れとして扱いました。"
            "--show-variations で表示できます）"
        ),
        "elapsed": "  応答時間: {ms:.1f} ms（{engine}）",
        "audience": {"external": "社外", "internal": "社内", "public": "不特定多数"},
        "unknown_verb": "辞書にない動詞です: {verb!r}\n収録している動詞: {sample} …",
        "forms_title": "「{verb}」の敬語形（宛先: {aud}）",
        "forms_none": "    （この立場で作れる形はありません）",
        "sides": {},
        "explain_unknown": "誤り種別を特定できませんでした。次のいずれかを指定してください:",
        "related": "関係する敬語の分類:",
    },
}


def _lang_of(args: argparse.Namespace) -> str:
    value = str(getattr(args, "lang", "en") or "en").lower()
    return "ja" if value.startswith("ja") else "en"


class _Paint:
    """色付け。NO_COLOR と --no-color を尊重する。"""

    def __init__(self, enabled: bool) -> None:
        """色付けの有効・無効を保持する。

        実証する主張: 「過剰指摘の少なさ」。色は指摘の重さを伝える手段なので、揺れと誤りで塗り分けられるようにする。
        """
        self.enabled = enabled

    def __call__(self, text: str, *styles: str) -> str:
        """文字列に色を付ける（無効なら素通し）。

        実証する主張: 「過剰指摘の少なさ」。NO_COLOR を尊重し、色に依存しない読み方も残す。
        """
        if not self.enabled or not styles:
            return text
        pre = "".join(_ANSI.get(s, "") for s in styles)
        return f"{pre}{text}{_ANSI['reset']}"


def _color_enabled(no_color: bool) -> bool:
    if no_color or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


# ---------------------------------------------------------------------------
# 文脈の組み立て
# ---------------------------------------------------------------------------

_SIDE_ALIASES = {
    "self": Party.SELF,
    "self_group": Party.SELF_GROUP,
    "uchi": Party.SELF_GROUP,
    "身内": Party.SELF_GROUP,
    "addressee": Party.ADDRESSEE,
    "addressee_group": Party.ADDRESSEE_GROUP,
    "soto": Party.ADDRESSEE_GROUP,
    "相手": Party.ADDRESSEE,
    "third_party": Party.THIRD_PARTY,
    "third": Party.THIRD_PARTY,
    "第三者": Party.THIRD_PARTY,
}


def _parse_person(spec: str) -> Person:
    """``--person 佐藤:self_group`` を :class:`Person` に変換する。"""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--person は「名前:立場」の形で指定してください: {spec!r}"
        )
    name, side = spec.rsplit(":", 1)
    key = side.strip().lower()
    if key not in _SIDE_ALIASES:
        raise argparse.ArgumentTypeError(
            f"未知の立場です: {side!r}（"
            + " / ".join(sorted({v.value for v in _SIDE_ALIASES.values()}))
            + "）"
        )
    return Person(name.strip(), _SIDE_ALIASES[key])


def _build_context(args: argparse.Namespace) -> MailContext:
    persons = tuple(args.person or ())
    return MailContext(
        audience=Audience.parse(args.audience),
        writer_org=args.writer_org,
        recipient_org=args.recipient_org,
        recipient_name=args.recipient_name,
        persons=persons,
    )


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------


def _render_text(
    result: CheckResult, *, paint: _Paint, show_variations: bool, lang: str = "en"
) -> str:
    from .cite import RuleCitation
    from .types import error_type_name

    M = _MSG[lang]

    lines: List[str] = []
    reportable = result.reportable
    variations = result.variations

    # --- 本文（該当箇所に下線と色） ---------------------------------------
    shown = list(reportable) + (list(variations) if show_variations else [])
    shown.sort(key=lambda f: f.span.start)
    body = []
    pos = 0
    for f in shown:
        if f.span.start < pos:
            continue
        body.append(result.text[pos : f.span.start])
        if f.verdict is Verdict.VARIATION:
            body.append(paint(f.span.text, "grey", "under"))
        else:
            body.append(paint(f.span.text, _TYPE_COLOR.get(f.error_type, "yellow"), "under"))
        pos = f.span.end
    body.append(result.text[pos:])
    lines.append("".join(body).rstrip())
    lines.append("")

    aud = M["audience"][result.context.audience.value]
    lines.append(paint("── " + M["heading"].format(aud=aud), "bold"))
    lines.append("")
    if not reportable:
        lines.append(paint(M["none"], "green"))
    for i, f in enumerate(reportable, 1):
        color = _TYPE_COLOR.get(f.error_type, "yellow")
        head = f"  {i}. " + paint(f"「{f.span.text}」", color, "bold")
        head += paint(f"  [{error_type_name(f.error_type, lang)}]", "dim")
        lines.append(head)
        if f.message:
            lines.append(f"     {f.message}")
        if f.suggestions:
            forms = " / ".join(dict.fromkeys(s.text for s in f.suggestions[:5]))
            lines.append("     " + paint(M["alt"], "green") + forms)
        if f.citation:
            lines.append(
                "     "
                + paint(
                    M["basis"]
                    + f"{f.citation.source_label(lang)} "
                    + f"{f.citation.section_label(lang)} ({f.citation.page})",
                    "dim",
                )
            )
            if f.citation.url:
                lines.append("     " + paint(f"      {f.citation.url}", "dim"))
            # note は日本語の補足なので、英語表示では出さない。
            # 英語の説明は RuleCitation.explain / teach が担う。
            if f.citation.note and lang == "ja":
                lines.append("     " + paint(f"      {f.citation.note}", "dim"))
        lines.append("")

    if variations:
        if show_variations:
            lines.append(paint("── " + M["variations_heading"], "bold"))
            lines.append("")
            for f in variations:
                lines.append("  ・" + paint(f"「{f.span.text}」", "grey"))
                if f.message:
                    lines.append(f"     {f.message}")
            lines.append("")
        else:
            lines.append(paint(M["variations_hint"].format(n=len(variations)), "dim"))
            lines.append("")

    lines.append(
        paint(M["elapsed"].format(ms=result.elapsed_ms, engine=result.engine), "dim")
    )
    lines.append("")
    lines.append(paint(RuleCitation(lang).disclaimer(), "dim"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# サブコマンド
# ---------------------------------------------------------------------------


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _cmd_check(args: argparse.Namespace) -> int:
    from .pipeline import Deference

    text = _read_input(args.file)
    ctx = _build_context(args)
    lang = _lang_of(args)
    df = Deference(
        engine=args.engine,
        model_dir=args.model_dir,
        threshold=args.threshold,
        report_variations=args.show_variations,
        lang=lang,
    )
    result = df.check(text, ctx)

    if args.format == "json":
        payload = to_jsonable(result)
        payload["reportable_count"] = len(result.reportable)
        payload["variation_count"] = len(result.variations)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        paint = _Paint(_color_enabled(args.no_color))
        print(
            _render_text(
                result,
                paint=paint,
                show_variations=args.show_variations,
                lang=lang,
            )
        )
    # 指摘があっても異常終了にはしない。これは「誤り」を咎めるツールではない。
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from .cite import RuleCitation

    lang = _lang_of(args)
    M = _MSG[lang]
    rc = RuleCitation(lang)
    paint = _Paint(_color_enabled(args.no_color))
    key = args.error_type
    try:
        et = ErrorType(key)
    except ValueError:
        matches = [
            e
            for e in ErrorType
            if key in e.value or key in ERROR_TYPE_JA.get(e, "")
        ]
        if len(matches) != 1:
            print(M["explain_unknown"], file=sys.stderr)
            for e in ErrorType:
                print(
                    f"  {e.value:26} {error_type_name(e, lang)}", file=sys.stderr
                )
            return 2
        et = matches[0]

    print(paint(f"■ {error_type_name(et, lang)}  ({et.value})", "bold"))
    print()
    print(rc.teach(et))
    print()
    cls = rc.classes_involved(et)
    if cls:
        print(paint(M["related"], "bold"))
        for c in cls:
            name = (KEIGO_CLASS_EN if lang == "en" else KEIGO_CLASS_JA).get(c, c.value)
            print(f"  - {name}: {rc.class_definition(c)}")
        print()
    cit = rc.for_error(et)
    print(
        paint(M["basis"].rstrip(": "), "bold") + ":",
        f"{cit.source_label(lang)} {cit.section_label(lang)} ({cit.page})",
    )
    print(f"      {cit.url}")
    print()
    print(paint(rc.disclaimer(), "dim"))
    return 0


def _cmd_forms(args: argparse.Namespace) -> int:
    from .pipeline import Deference

    lang = _lang_of(args)
    M = _MSG[lang]
    paint = _Paint(_color_enabled(args.no_color))
    df = Deference(engine="norm", lang=lang)
    try:
        table = df.forms(args.verb, audience=Audience.parse(args.audience))
    except KeyError:
        print(
            M["unknown_verb"].format(
                verb=args.verb,
                sample="、".join(v.plain for v in norms.VERBS[:20]),
            ),
            file=sys.stderr,
        )
        return 2
    print(paint("■ " + M["forms_title"].format(verb=args.verb, aud=args.audience), "bold"))
    print()
    names = KEIGO_CLASS_EN if lang == "en" else KEIGO_CLASS_JA
    for label, sugs in table.items():
        print(paint(f"  {M['sides'].get(label, label)}", "bold"))
        if not sugs:
            print(paint(M["forms_none"], "dim"))
        for s in sugs:
            print(f"    {s.text:22} {paint(names.get(s.keigo_class,''),'dim')}")
            if s.reason and lang == "ja":
                print(paint(f"      {s.reason}", "dim"))
        print()
    print(paint(norms.LICENSE_NOTE, "dim"))
    return 0


_DEMO = """田中様

いつもお世話になっております。株式会社アルファの山田です。

先日は資料をお送りくださいまして、ありがとうございました。
弊社の佐藤社長が、大変参考になったとおっしゃっておりました。
つきましては、来週の打ち合わせについてご相談させていただきたく存じます。
なお、当日は私が資料をお持ちになります。
恐れ入りますが、担当者に伺ってください。
明日は都合により休まさせていただきます。

よろしくお願いいたします。
"""


def _cmd_demo(args: argparse.Namespace) -> int:
    from .pipeline import Deference

    lang = _lang_of(args)
    paint = _Paint(_color_enabled(args.no_color))
    for aud in (Audience.EXTERNAL, Audience.INTERNAL):
        ctx = MailContext(
            audience=aud,
            writer_org="株式会社アルファ",
            recipient_org="株式会社ベータ",
            recipient_name="田中",
            persons=(
                Person("佐藤", Party.SELF_GROUP, "社長", "株式会社アルファ"),
                Person("田中", Party.ADDRESSEE, "様", "株式会社ベータ"),
            ),
        )
        df = Deference(
            engine="norm", report_variations=args.show_variations, lang=lang
        )
        res = df.check(_DEMO, ctx)
        print(paint(f"════ audience: {aud.value} ════", "bold"))
        print(
            _render_text(
                res,
                paint=paint,
                show_variations=args.show_variations,
                lang=lang,
            )
        )
        print()
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数の定義。

    実証する主張: 「向きの誤り検出」。--audience を必須級の入力として前面に出す。
    """
    p = argparse.ArgumentParser(
        prog="deference",
        description=(
            "Japanese honorific (keigo) checker. Reports how the Council for "
            "Cultural Affairs' Keigo no Shishin frames each passage, with the "
            "source citation."
        ),
        epilog="example: deference check mail.txt --audience external",
    )
    p.add_argument("--no-color", action="store_true", help="disable colour output")
    p.add_argument(
        "--lang",
        default="en",
        choices=["en", "ja"],
        help="language of the output (default: en)",
    )
    sub = p.add_subparsers(dest="command")

    c = sub.add_parser("check", help="check a message body")
    c.add_argument("file", help="path to the message body; '-' reads stdin")
    c.add_argument(
        "--audience",
        default="external",
        choices=["internal", "external", "public"],
        help="audience (default: external). The same text can be judged differently",
    )
    c.add_argument("--writer-org", default="弊社", help="your own organisation name")
    c.add_argument("--recipient-org", default="貴社", help="the reader's organisation name")
    c.add_argument("--recipient-name", default="", help="the reader's name")
    c.add_argument(
        "--person",
        action="append",
        type=_parse_person,
        metavar="NAME:SIDE",
        help="standpoint of a person in the text (self / self_group / addressee / "
        "addressee_group / third_party). May be repeated",
    )
    c.add_argument(
        "--engine", default="auto", choices=["auto", "norm", "neural", "hybrid"]
    )
    c.add_argument("--model-dir", default=None, help="directory of a trained checkpoint")
    c.add_argument("--threshold", type=float, default=0.5)
    c.add_argument("--format", default="text", choices=["text", "json"])
    c.add_argument(
        "--show-variations", action="store_true", help="also show passages treated as variation"
    )
    c.add_argument("--no-color", action="store_true")
    c.add_argument("--lang", default="en", choices=["en", "ja"])
    c.set_defaults(func=_cmd_check)

    e = sub.add_parser("explain", help="explain how an error type works")
    e.add_argument("error_type", help="error type (e.g. uchi_sonkeigo)")
    e.add_argument("--no-color", action="store_true")
    e.add_argument("--lang", default="en", choices=["en", "ja"])
    e.set_defaults(func=_cmd_explain)

    f = sub.add_parser("forms", help="list the honorific forms of a verb")
    f.add_argument("verb", help="plain verb (e.g. 言う)")
    f.add_argument(
        "--audience", default="external", choices=["internal", "external", "public"]
    )
    f.add_argument("--no-color", action="store_true")
    f.add_argument("--lang", default="en", choices=["en", "ja"])
    f.set_defaults(func=_cmd_forms)

    d = sub.add_parser("demo", help="run the built-in sample")
    d.add_argument("--show-variations", action="store_true")
    d.add_argument("--no-color", action="store_true")
    d.add_argument("--lang", default="en", choices=["en", "ja"])
    d.set_defaults(func=_cmd_demo)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """エントリポイント。

    実証する主張: 「速度」。既定エンジンは規範ベースで、torch を読み込まずに起動する。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"ファイルが見つかりません: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
