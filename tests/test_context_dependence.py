"""同じ本文でも、宛先が社内か社外かで結果が変わること。

実証する主張: 「向きの誤り検出」。指針 第3章第3-2【25】は、社内の会なら
「社長からごあいさつを頂きます」、社外の人が多い会なら「社長からごあいさつを
申し上げます」が適切だとする。適否が宛先で入れ替わるのだから、宛先を入力に
取らないツールにはこの判定ができない。ここが Deference の構造的な差である。
"""

from __future__ import annotations

from deference.pipeline import Deference
from deference.types import (
    Audience,
    ErrorType,
    MailContext,
    Party,
    Person,
    Verdict,
)

MAIL = (
    "田中様\n\n"
    "いつもお世話になっております。\n"
    "弊社の佐藤社長が、そのようにおっしゃっておりました。\n"
    "よろしくお願いいたします。\n"
)


def _context(audience: Audience) -> MailContext:
    return MailContext(
        audience=audience,
        writer_org="株式会社アルファ",
        recipient_org="株式会社ベータ",
        recipient_name="田中",
        persons=(
            Person("佐藤", Party.SELF_GROUP, "社長", "株式会社アルファ"),
            Person("田中", Party.ADDRESSEE, "様", "株式会社ベータ"),
        ),
    )


def test_uchi_sonkeigo_is_reported_only_for_external():
    df = Deference(engine="norm")
    external = df.check(MAIL, _context(Audience.EXTERNAL))
    internal = df.check(MAIL, _context(Audience.INTERNAL))

    ext_types = {f.error_type for f in external.reportable}
    int_types = {f.error_type for f in internal.reportable}

    assert ErrorType.UCHI_SONKEIGO in ext_types, (
        "社外宛で身内敬語が指摘されていません。"
        f"実際の指摘: {[f.error_type.value for f in external.reportable]}"
    )
    assert ErrorType.UCHI_SONKEIGO not in int_types, (
        "社内宛でも身内敬語を指摘しています。指針【26】は二通りの考え方の"
        "どちらにも理があるとしており、規範違反として断ずるのは過剰です。"
    )
    # 社内宛では「揺れ」として保持されていること
    assert any(
        f.error_type is ErrorType.UCHI_SONKEIGO and f.verdict is Verdict.VARIATION
        for f in internal.findings
    ), "社内宛で身内敬語が揺れとして保持されていません"

    assert len(external.reportable) > len(internal.reportable), (
        f"宛先で指摘数が変わっていません "
        f"(社外={len(external.reportable)}, 社内={len(internal.reportable)})"
    )


def test_citation_changes_with_audience():
    """根拠に引く指針の問いも、宛先によって変わること。"""
    df = Deference(engine="norm")
    ext = df.check(MAIL, _context(Audience.EXTERNAL))
    hit = next(f for f in ext.reportable if f.error_type is ErrorType.UCHI_SONKEIGO)
    assert "【25】" in hit.citation.section, hit.citation.section
