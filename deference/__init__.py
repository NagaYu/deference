"""Deference — 日本語敬語の誤り検出・修正。

規則ベースの校正ツールが捕まえられない「敬意の向き」の誤りを、CPU で動く小型
モデルで検出し、**規範の根拠と修正候補を添えて**返す。

基準は文化審議会答申「敬語の指針」（平成19年2月2日）。引用の扱いは
:data:`deference.norms.LICENSE_NOTE` と docs/keigo_shishin_reference.md を参照。

3行で使う::

    from deference import Deference, MailContext, Audience
    result = Deference().check(open("mail.txt").read(), MailContext(audience=Audience.EXTERNAL))
    for f in result.reportable: print(f.span.text, f.message, f.citation.render())
"""

from .types import (
    Audience,
    CheckResult,
    Citation,
    ErrorType,
    Finding,
    KeigoClass,
    MailContext,
    Party,
    Person,
    Span,
    Suggestion,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "Deference",
    "Audience",
    "CheckResult",
    "Citation",
    "ErrorType",
    "Finding",
    "KeigoClass",
    "MailContext",
    "Party",
    "Person",
    "Span",
    "Suggestion",
    "Verdict",
    "__version__",
]


def __getattr__(name: str):
    """``Deference`` を遅延 import する。

    実証する主張: 「速度」。``import deference`` の時点では pipeline も
    torch も読み込まない。CLI の起動を軽く保つための措置である。
    """
    if name == "Deference":
        from .pipeline import Deference as _D

        return _D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
