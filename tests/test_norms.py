"""規範データベースの整合性。

実証する主張: 「根拠提示」。全ての誤り種別が原典の該当箇所に対応づくこと、
活用の規則が往復すること、定着形が誤り側に落ちないことを固定する。
"""

from __future__ import annotations

import pytest

from deference import norms
from deference.types import ErrorType, KeigoClass


@pytest.mark.parametrize("error_type", list(ErrorType))
def test_every_error_type_has_a_citation(error_type):
    citation = norms.cite_for(error_type)
    assert citation.source, error_type
    assert citation.section, error_type


def test_citations_do_not_misattribute_source():
    """指針を根拠にしていない項目が、指針の URL を持たないこと。

    さ入れ言葉は「敬語の指針」が扱っていない。根拠の出所を偽らないための検査。
    """
    sa = norms.cite("sa_insertion")
    assert "該当記述なし" in sa.source
    assert sa.url == "", "指針を根拠にしていないのに指針の URL が付いています"


def test_five_classes_have_definitions():
    for cls in (
        KeigoClass.SONKEIGO,
        KeigoClass.KENJOUGO_1,
        KeigoClass.KENJOUGO_2,
        KeigoClass.TEINEIGO,
        KeigoClass.BIKAGO,
    ):
        assert norms.KEIGO_CLASS_DEFINITION[cls]


@pytest.mark.parametrize("verb", [v for v in norms.VERBS])
def test_verb_forms_are_constructible(verb):
    """全ての動詞で一般形が例外なく作れること。"""
    verb.sonkeigo_general()
    verb.kenjougo1_general()
    verb.kenjougo2_general()
    assert verb.stem
    if verb.kind == "sahen":
        assert verb.ogo_stem == verb.plain[:-2], (
            "サ変動詞の「お(ご)……」語幹は語基でなければなりません"
            "（ご利用になる であって ご利用しになる ではない）"
        )


def test_ogo_ni_naru_not_built_for_forbidden_verbs():
    """指針【ア－3】が挙げる作れない形を作らないこと。"""
    for name in ("死ぬ", "失敗する", "運転する"):
        forms = norms.verb(name).sonkeigo_general()
        assert not any(f.startswith(("お", "ご")) and "になる" in f for f in forms), (
            f"{name}: 指針は「お死にになる」「ご失敗になる」「ご運転になる」を"
            f"作れない例として挙げています。実際: {forms}"
        )


def test_kenjougo1_requires_a_target():
    """＜向かう先＞が無い動詞に「お(ご)……する」を作らないこと（指針【補足イ－1】）。"""
    for name in ("乗車する", "利用する", "休む"):
        assert norms.verb(name).kenjougo1_general() == [], name


@pytest.mark.parametrize(
    "form,kind,expected",
    [
        ("読む", "godan", "読み"),
        ("届ける", "ichidan", "届け"),
        ("利用する", "sahen", "利用し"),
        ("おっしゃる", "godan", "おっしゃい"),
        ("くださる", "godan", "ください"),
        ("なさる", "godan", "なさい"),
    ],
)
def test_masu_stem(form, kind, expected):
    assert norms.masu_stem(form, kind) == expected


@pytest.mark.parametrize(
    "form,kind,expected",
    [("読む", "godan", "読ま"), ("休む", "godan", "休ま"), ("伺う", "godan", "伺わ")],
)
def test_a_stem_for_sa_insertion(form, kind, expected):
    assert norms.a_stem(form, kind) == expected


def test_potential_is_built_after_keigo():
    """指針 第2章第2-1（1）② に従い、敬語形にしてから可能形にすること。"""
    assert norms.potential("お読みになる", "godan") == "お読みになれる"
    assert norms.potential("お届けする", "sahen") == "お届けできる"


def test_established_double_keigo_is_listed():
    for form in ("お伺いする", "お召し上がりになる", "お見えになる"):
        assert form in norms.ESTABLISHED_DOUBLE_KEIGO


def test_special_forms_all_conjugate():
    for sf in norms.SPECIAL_FORMS:
        assert sf.polite_form, sf
