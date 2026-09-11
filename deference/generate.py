"""Generator：立場と述語から「規範上作れる正しい敬語形」を規則で構成する。

Deference のデータはすべてここから出る。

* :meth:`Generator.correct_forms` は〈動作主の立場 × ＜向かう先＞ × 目標敬意度〉から
  作れる述語形の**閉じた集合**を返す。Corrector はこの集合の外から候補を出さない。
* :meth:`Generator.generate_corpus` はその集合を文に埋め込んだ**正例（負例集合）**を返す。
  ここに指摘が出れば、それは定義上の偽陽性である。
* :meth:`Generator.generate_mail` は宛名・あいさつ・本文・結びを備えたメール全体を返す。
  Deference は単文ではなくメール全体で判定するため、全体を持つサンプルが要る。

実証する主張:
    - 「向きの誤り検出」: 語形は :class:`~deference.types.Party` の組合せから決まる。
      同じ動詞でも動作主が自分側か相手側かで作れる集合が入れ替わることを、
      生成側で先に固定する。表層文字列からは決められないことの裏返しである。
    - 「過剰指摘の少なさ」: ここで作った文はすべて規範に沿う。評価時の負例集合として
      使い、偽陽性率を測る。
    - 「根拠提示」: 各 :class:`~deference.types.Suggestion` の ``reason`` に
      「敬語の指針」の該当箇所（章・節）を必ず書き込む。
    - 「速度」: 生成は純粋な文字列処理のみで、外部モデルも辞書ファイル I/O も使わない。

方針:
    - 語形・分類・引用は :mod:`deference.norms` からのみ引く。本モジュールが独自に
      持つのは「どの立場でどの一般形を採るか」という組合せ規則と、例文の骨組み
      （目的語の名詞・定型句）だけである。ローカル定数にはすべて理由を書いた。
    - 利用者に見える ``reason`` は情報提示に留め、断定的な否定表現を使わない。
    - 時制は原則として非過去に固定する。過去形の活用まで扱うと規則が増える割に
      敬語の向きの検証には効かないため。
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import norms
from .norms import Verb
from .types import (
    Audience,
    FunctionTag,
    GeneratedSentence,
    KEIGO_CLASS_JA,
    KeigoClass,
    MailContext,
    Party,
    Person,
    Span,
    Suggestion,
)

__all__ = [
    "Generator",
    "default_context",
    "ORIGINS",
]


# ---------------------------------------------------------------------------
# ローカル定数（なぜ norms.py ではなくここにあるか）
# ---------------------------------------------------------------------------

#: 合成形の末尾に来る不規則活用の助動詞的動詞。
#:
#: norms.masu_stem() は「語全体」が norms._IRREGULAR_MASU_STEM に一致したときだけ
#: 不規則な連用形を返す。ところが本モジュールが組み立てるのは「ご利用なさる」
#: 「お読みくださる」のような合成形なので、語全体では一致しない。そこで末尾だけを
#: 切り出して norms 側の関数に渡すための糊としてここに置く。
#: 新しい敬語知識ではなく、norms の活用規則を合成形へ適用するための実装補助である。
_IRREGULAR_TAILS: Tuple[str, ...] = (
    "いらっしゃる",
    "おっしゃる",
    "なさる",
    "くださる",
    "ござる",
)

#: 特定形の見出し語（norms.SPECIAL_FORMS の ``base``）のうち norms.VERBS に無い語の活用種別。
#:
#: norms.SpecialForm は特定形側の活用種別しか持たないため、素の動詞を丁寧語
#: （「行きます」）にするには素の側の活用種別が要る。これは敬語の知識ではなく
#: 素の活用情報なので、必要最小限をここに置く。
_BASE_KINDS: Dict[str, str] = {
    "行く": "godan",
    "来る": "kahen",
    "いる": "ichidan",
    "言う": "godan",
    "する": "sahen",
    "食べる": "ichidan",
    "飲む": "godan",
    "くれる": "ichidan",
    "見る": "ichidan",
    "寝る": "ichidan",
    "着る": "ichidan",
    "知る": "godan",
    "訪ねる": "ichidan",
    "尋ねる": "ichidan",
    "聞く": "godan",
    "上げる": "ichidan",
    "もらう": "godan",
    "会う": "godan",
    "見せる": "ichidan",
    "借りる": "ichidan",
    "思う": "godan",
}

#: norms.VERBS に無いが、機能タグの定型文に必要な語。
#:
#: 語彙項目だけをここで足し、語形の作り方（お(ご)……する／申し上げる など）は
#: norms.Verb の一般形メソッドにそのまま委ねる。つまり敬語の作り方は
#: norms.py の規則を一切迂回していない。
_EXTRA_VERBS: Tuple[Verb, ...] = (
    Verb("詫びる", "ichidan", "詫びる", "お", True, True, ("apology",)),
)

#: 例文の目的語（素の名詞, 助詞）。敬語の知識ではなく例文の骨組みなのでローカル。
#: 空文字は「目的語を置かない」を意味する。
_OBJECTS: Dict[str, Tuple[str, str]] = {
    "読む": ("資料", "を"),
    "書く": ("議事録", "を"),
    "送る": ("資料", "を"),
    "届ける": ("書類", "を"),
    "待つ": ("ご返信", "を"),
    "使う": ("システム", "を"),
    "呼ぶ": ("担当者", "を"),
    "渡す": ("書類", "を"),
    "預かる": ("資料", "を"),
    "持つ": ("資料", "を"),
    "知らせる": ("結果", "を"),
    "願う": ("ご協力", "を"),
    "勧める": ("新プラン", "を"),
    "誘う": ("懇親会", "に"),
    "受け取る": ("書類", "を"),
    "決める": ("日程", "を"),
    "選ぶ": ("プラン", "を"),
    "調べる": ("原因", "を"),
    "帰る": ("本社", "へ"),
    "休む": ("明日", "は"),
    "急ぐ": ("作業", "を"),
    "考える": ("方針", "を"),
    "使いこなす": ("システム", "を"),
    "運転する": ("社用車", "を"),
    "利用する": ("本サービス", "を"),
    "出席する": ("会議", "に"),
    "案内する": ("会場", "を"),
    "説明する": ("内容", "を"),
    "連絡する": ("結果", "を"),
    "確認する": ("内容", "を"),
    "検討する": ("ご提案", "を"),
    "報告する": ("進捗", "を"),
    "相談する": ("本件", "について"),
    "対応する": ("本件", "に"),
    "参加する": ("説明会", "に"),
    "送付する": ("資料", "を"),
    "提案する": ("改善案", "を"),
    "返信する": ("メール", "に"),
    "訪問する": ("貴社", "を"),
    "確保する": ("席", "を"),
    "承知する": ("事情", "を"),
    "乗車する": ("電車", "に"),
    "持参する": ("資料", "を"),
    "記入する": ("申込書", "に"),
    "指導する": ("新人", "を"),
    "協力する": ("本件", "に"),
    "行く": ("貴社", "へ"),
    "来る": ("弊社", "へ"),
    "くれる": ("資料", "を"),
    "見る": ("資料", "を"),
    "知る": ("事情", "を"),
    "訪ねる": ("貴社", "を"),
    "尋ねる": ("担当者", "に"),
    "聞く": ("ご意見", "を"),
    "上げる": ("資料", "を"),
    "もらう": ("資料", "を"),
    "会う": ("担当者", "に"),
    "見せる": ("資料", "を"),
    "借りる": ("会議室", "を"),
}

#: 例文に登場させる人物名。実在の個人を指さない一般的な姓のみ。
_NAME_POOL: Dict[Party, Tuple[str, ...]] = {
    Party.SELF_GROUP: ("田中", "高橋", "伊藤", "渡辺", "中村"),
    Party.ADDRESSEE_GROUP: ("鈴木", "佐々木", "小林", "加藤"),
    Party.THIRD_PARTY: ("山本", "井上", "木村", "林"),
}

#: 社内宛のときだけ身内に付ける役職。社外宛で付けないのは指針【24】に従う
#: （「田中部長」と呼ぶのはウチ扱いにした呼び方にならない）。
_INTERNAL_TITLES: Tuple[str, ...] = ("部長", "課長", "主任")


#: 候補の由来ラベル。テンプレートはこのラベルで欲しい形を指定する。
ORIGINS: Tuple[str, ...] = (
    "sonkeigo_special",  # 尊敬語の特定形（おっしゃる 等）
    "sonkeigo_ogo_ni_naru",  # お(ご)……になる
    "sonkeigo_nasaru",  # ……なさる／ご……なさる
    "sonkeigo_rareru",  # ……(ら)れる
    "sonkeigo_ogo_da",  # お(ご)……だ
    "sonkeigo_kudasaru",  # お(ご)……くださる
    "te_kudasaru",  # ……てくださる
    "kenjougo1_special",  # 謙譲語Ⅰの特定形（伺う 等）
    "kenjougo1_ogo_suru",  # お(ご)……する
    "kenjougo1_ogo_moushiageru",  # お(ご)……申し上げる
    "kenjougo1_ogo_itadaku",  # お(ご)……いただく
    "te_itadaku",  # ……ていただく
    "te_sashiageru",  # お(ご)……して差し上げる
    "kenjougo2_special",  # 謙譲語Ⅱの特定形（参る・申す 等）
    "kenjougo2_itasu",  # ……いたす
    "kenjougo1and2_ogo_itasu",  # お(ご)……いたす（謙譲語Ⅰ兼Ⅱ）
    "teineigo_plain",  # 素の動詞（丁寧語「ます」を付けるだけ）
)


# ---------------------------------------------------------------------------
# 活用の糊（norms の関数を合成形へ適用する）
# ---------------------------------------------------------------------------


def _split_tail(form: str) -> Tuple[str, str]:
    """合成形を〈前部, 不規則活用の末尾語〉に割る。一致しなければ前部は空。"""
    for tail in _IRREGULAR_TAILS:
        if form.endswith(tail) and form != tail:
            return form[: -len(tail)], tail
    return "", form


def _masu_stem(form: str, kind: str) -> str:
    """合成形にも効く連用形。末尾の不規則語だけ切り出して norms へ渡す。"""
    head, tail = _split_tail(form)
    return head + norms.masu_stem(tail, kind)


def _polite(form: str, kind: str) -> str:
    """合成形にも効く「ます／です」形。"""
    if kind == "copula":
        return norms.polite(form, kind)
    return _masu_stem(form, kind) + "ます"


# ---------------------------------------------------------------------------
# 候補
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """敬意度を付ける前の述語候補（終止形）。"""

    form: str  # 終止形（お読みになる）
    kind: str  # 活用種別（godan / ichidan / sahen / kahen / copula）
    keigo_class: KeigoClass
    origin: str  # ORIGINS のいずれか
    reason: str


# 候補の並び順。先頭ほど「その立場で真っ先に案内したい形」。
_RANK_SELF: Tuple[str, ...] = (
    "kenjougo1_special",
    "kenjougo1and2_ogo_itasu",
    "kenjougo1_ogo_suru",
    "kenjougo1_ogo_moushiageru",
    "kenjougo2_special",
    "kenjougo2_itasu",
    "te_sashiageru",
    "teineigo_plain",
)
_RANK_OTHER: Tuple[str, ...] = (
    "sonkeigo_special",
    "sonkeigo_ogo_ni_naru",
    "sonkeigo_nasaru",
    "sonkeigo_kudasaru",
    "sonkeigo_rareru",
    "sonkeigo_ogo_da",
    "kenjougo1_ogo_itadaku",
    "te_kudasaru",
    "te_itadaku",
    "teineigo_plain",
)
#: 敬意度2（より改まった形）では「……(ら)れる」より特定形・お(ご)……になるを優先する。
_RANK_OTHER_FORMAL: Tuple[str, ...] = (
    "sonkeigo_special",
    "sonkeigo_ogo_ni_naru",
    "sonkeigo_nasaru",
    "sonkeigo_kudasaru",
    "kenjougo1_ogo_itadaku",
    "sonkeigo_ogo_da",
    "sonkeigo_rareru",
    "te_kudasaru",
    "te_itadaku",
    "teineigo_plain",
)


# ---------------------------------------------------------------------------
# 文型テンプレート
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Template:
    """機能タグごとの文の骨組み。

    ``pre`` / ``post`` のプレースホルダ:
        ``{subj}``  動作主 + 「が」（動作主が書き手自身のときは空）
        ``{you}``   宛先の人物（佐藤様）
        ``{obj}``   目的語（名詞 + 助詞）
        ``{objn}``  目的語の名詞のみ
    """

    function: FunctionTag
    actor_side: str  # "self" | "other"
    prefer: Tuple[str, ...]  # 使いたい候補の由来（前から順に探す）
    ending: str  # declarative / question / request / kane / renyou / potential
    pre: str
    post: str
    verbs: Tuple[str, ...] = ()  # 空なら動詞を限定しない
    min_politeness: int = 0


_T = _Template

_SELF_DECL = (
    "kenjougo1and2_ogo_itasu",
    "kenjougo1_special",
    "kenjougo1_ogo_suru",
    "kenjougo1_ogo_moushiageru",
    "kenjougo2_special",
    "kenjougo2_itasu",
    "teineigo_plain",
)
_OTHER_DECL = (
    "sonkeigo_special",
    "sonkeigo_ogo_ni_naru",
    "sonkeigo_nasaru",
    "sonkeigo_rareru",
    "sonkeigo_ogo_da",
    "teineigo_plain",
)
_OTHER_SONKEIGO = (
    "sonkeigo_special",
    "sonkeigo_ogo_ni_naru",
    "sonkeigo_nasaru",
    "sonkeigo_rareru",
)

#: 機能タグ × 立場ごとの文型。ビジネス頻出の10機能をすべて覆う。
_TEMPLATES: Tuple[_Template, ...] = (
    # --- 報告・連絡 --------------------------------------------------------
    # 主語を明示する文型。動作主が文面から読めないと敬意の向きは判定できないため、
    # 立場が明示された文をコーパスに十分入れておく必要がある。
    _T(FunctionTag.REPORT, "self", _SELF_DECL, "declarative", "{subj}{obj}", "。"),
    _T(FunctionTag.REPORT, "self", _SELF_DECL, "declarative",
       "本日は{subj}{obj}", "。"),
    _T(FunctionTag.REPORT, "self", _SELF_DECL, "declarative", "本日、{obj}", "。"),
    _T(FunctionTag.REPORT, "self", _SELF_DECL, "declarative",
       "取り急ぎ、{obj}", "。"),
    _T(FunctionTag.REPORT, "other", _OTHER_DECL, "declarative", "{subj}{obj}", "。"),
    _T(FunctionTag.REPORT, "other", _OTHER_DECL, "declarative",
       "その後、{subj}{obj}", "。"),
    # --- 問い合わせ --------------------------------------------------------
    _T(FunctionTag.INQUIRY, "other", _OTHER_SONKEIGO, "question",
       "恐れ入りますが、{obj}", "。", min_politeness=1),
    _T(FunctionTag.INQUIRY, "self",
       ("kenjougo1_special", "kenjougo1_ogo_suru", "kenjougo1_ogo_moushiageru"),
       "declarative", "{objn}につきまして、", "。",
       verbs=("聞く", "尋ねる", "訪ねる", "相談する", "確認する", "問う"),
       min_politeness=1),
    # --- 日程調整 ----------------------------------------------------------
    _T(FunctionTag.SCHEDULING, "other", _OTHER_SONKEIGO, "question",
       "来週の打ち合わせに", "。",
       verbs=("出席する", "参加する", "訪問する", "来る", "行く", "確認する"),
       min_politeness=1),
    _T(FunctionTag.SCHEDULING, "other", _OTHER_SONKEIGO, "declarative",
       "{subj}来週の打ち合わせに", "。",
       verbs=("出席する", "参加する", "訪問する", "来る", "行く"),
       min_politeness=1),
    _T(FunctionTag.SCHEDULING, "self", _SELF_DECL, "declarative",
       "{subj}来週の打ち合わせに", "。",
       verbs=("出席する", "参加する", "訪問する", "行く", "来る"),
       min_politeness=1),
    _T(FunctionTag.SCHEDULING, "self", _SELF_DECL, "declarative",
       "来週の打ち合わせに", "。",
       verbs=("出席する", "参加する", "訪問する", "行く", "来る", "確保する"),
       min_politeness=1),
    # --- 告知 --------------------------------------------------------------
    _T(FunctionTag.NOTICE, "self",
       ("kenjougo1and2_ogo_itasu", "kenjougo1_ogo_suru",
        "kenjougo1_ogo_moushiageru", "kenjougo2_itasu", "teineigo_plain"),
       "declarative", "このたび、{subj}{obj}", "。",
       verbs=("案内する", "知らせる", "提案する", "送付する", "連絡する", "説明する"),
       min_politeness=1),
    _T(FunctionTag.NOTICE, "self",
       ("kenjougo1and2_ogo_itasu", "kenjougo1_ogo_suru",
        "kenjougo1_ogo_moushiageru", "kenjougo2_itasu", "teineigo_plain"),
       "declarative", "このたび、{obj}", "。",
       verbs=("案内する", "知らせる", "提案する", "送付する", "連絡する", "説明する"),
       min_politeness=1),
    _T(FunctionTag.NOTICE, "other", ("sonkeigo_ogo_ni_naru",), "potential",
       "来月より、{obj}", "。",
       verbs=("利用する", "参加する", "確認する", "読む", "見る", "使う"),
       min_politeness=1),
    # --- 依頼 --------------------------------------------------------------
    _T(FunctionTag.REQUEST, "other", ("sonkeigo_kudasaru", "te_kudasaru"),
       "request", "恐れ入りますが、{obj}", "。", min_politeness=1),
    _T(FunctionTag.REQUEST, "other", ("kenjougo1_ogo_itadaku", "te_itadaku"),
       "request", "お手数ですが、{obj}", "。", min_politeness=1),
    # --- 断り --------------------------------------------------------------
    _T(FunctionTag.REFUSAL, "self",
       ("kenjougo1and2_ogo_itasu", "kenjougo2_itasu", "kenjougo1_ogo_suru"),
       "kane", "誠に恐縮ですが、{objn}につきましては", "。",
       verbs=("対応する", "送付する", "案内する", "確保する", "協力する",
              "承知する", "参加する", "出席する"),
       min_politeness=1),
    # --- 謝罪 --------------------------------------------------------------
    _T(FunctionTag.APOLOGY, "self",
       ("kenjougo1_ogo_moushiageru", "kenjougo1_ogo_suru"),
       "declarative", "このたびはご迷惑をおかけしましたこと、深く", "。",
       verbs=("詫びる",), min_politeness=1),
    # --- 感謝 --------------------------------------------------------------
    _T(FunctionTag.THANKS, "other", ("kenjougo1_ogo_itadaku", "te_itadaku"),
       "renyou", "このたびは{objn}に関し", "、誠にありがとうございます。",
       min_politeness=1),
    # --- あいさつ ----------------------------------------------------------
    _T(FunctionTag.GREETING, "self",
       ("kenjougo1_ogo_moushiageru", "kenjougo1_ogo_suru"),
       "declarative", "今後ともよろしく", "。",
       verbs=("願う",), min_politeness=1),
    _T(FunctionTag.GREETING, "self",
       ("kenjougo1_ogo_moushiageru", "kenjougo1_ogo_suru"),
       "declarative", "引き続きご指導のほど、よろしく", "。",
       verbs=("願う",), min_politeness=1),
    # --- 授受（差し上げる側） ------------------------------------------------
    _T(FunctionTag.GIVING, "self",
       ("kenjougo1_special", "kenjougo1and2_ogo_itasu", "kenjougo1_ogo_suru",
        "kenjougo1_ogo_moushiageru"),
       "declarative", "{obj}", "。",
       verbs=("送る", "届ける", "渡す", "送付する", "案内する", "上げる",
              "見せる", "知らせる", "説明する"),
       min_politeness=1),
    _T(FunctionTag.GIVING, "self", ("te_sashiageru",), "declarative",
       "ご希望でしたら、{obj}", "。",
       verbs=("案内する", "説明する", "送付する", "読む"), min_politeness=1),
    # --- 授受（いただく／くださる側） ----------------------------------------
    _T(FunctionTag.RECEIVING, "other",
       ("sonkeigo_kudasaru", "sonkeigo_special", "te_kudasaru"),
       "declarative", "{subj}{obj}", "。", min_politeness=1),
    _T(FunctionTag.RECEIVING, "self",
       ("kenjougo1_special", "kenjougo1_ogo_suru"),
       "declarative", "{you}から{obj}", "。",
       verbs=("もらう", "受け取る", "借りる", "聞く", "見る"), min_politeness=1),
)


# ---------------------------------------------------------------------------
# 既定の文脈
# ---------------------------------------------------------------------------


def default_context(audience: Audience = Audience.EXTERNAL) -> MailContext:
    """既定のメール文脈を作る。

    実証する主張: 「向きの誤り検出」。宛先（社内／社外）が無いと同じ文の適否が
    決まらないため、既定値であっても :class:`~deference.types.Audience` を必ず持たせる。
    """
    if audience is Audience.INTERNAL:
        return MailContext(
            audience=audience,
            writer_name="山田",
            writer_org="営業部",
            recipient_name="佐藤",
            recipient_title="さん",
            recipient_org="開発部",
        )
    if audience is Audience.PUBLIC:
        return MailContext(
            audience=audience,
            writer_name="山田",
            writer_org="弊社",
            recipient_name="お客様",
            recipient_title="",
            recipient_org="",
        )
    return MailContext(audience=audience)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator:
    """規範から正しい敬語形と正例文を構成する。

    実証する主張:
        - 「向きの誤り検出」: 作れる語形の集合が動作主・＜向かう先＞の立場だけで
          決まることを、生成の側から示す。
        - 「過剰指摘の少なさ」: 出力はすべて規範に沿う正例なので、評価の負例集合
          （＝指摘が出てはいけない集合）として使える。
        - 「根拠提示」: すべての候補に「敬語の指針」の該当箇所を ``reason`` として付す。
        - 「速度」: 文字列処理のみで構成され、外部モデルを一切呼ばない。
    """

    def __init__(self, seed: int = 0) -> None:
        """乱数種を固定して決定的に生成する。

        実証する主張: 「過剰指摘の少なさ」。負例集合が再現可能でなければ
        偽陽性率の比較ができないため、種を明示的に受け取る。
        """
        self.seed = int(seed)
        self._verbs: Dict[str, Verb] = {v.plain: v for v in norms.VERBS}
        for v in _EXTRA_VERBS:
            self._verbs.setdefault(v.plain, v)
        # 生成の対象になりうる素の動詞（norms.VERBS ＋ 補助語 ＋ 特定形の見出し語）
        bases: List[str] = [v.plain for v in norms.VERBS]
        bases += [v.plain for v in _EXTRA_VERBS]
        bases += [sf.base for sf in norms.SPECIAL_FORMS]
        seen: set[str] = set()
        self.base_verbs: Tuple[str, ...] = tuple(
            b for b in bases if not (b in seen or seen.add(b))
        )

    # -- 内部ユーティリティ ------------------------------------------------

    def _rng(self, key: str) -> random.Random:
        return random.Random(f"{self.seed}|{key}")

    def _verb(self, base_verb: str) -> Optional[Verb]:
        return self._verbs.get(base_verb)

    def _plain_kind(self, base_verb: str) -> Optional[str]:
        v = self._verb(base_verb)
        if v is not None:
            return v.kind
        return _BASE_KINDS.get(base_verb)

    # -- 候補構成 ----------------------------------------------------------

    def _candidates(
        self,
        base_verb: str,
        actor: Party,
        target: Party,
        audience: Audience,
    ) -> List[_Candidate]:
        """立場から作れる候補（終止形）を並べる。順位付け済み。"""
        v = self._verb(base_verb)
        kind = self._plain_kind(base_verb)
        if v is None and kind is None and not norms.special_forms_for(base_verb):
            raise KeyError(
                f"生成できない動詞です: {base_verb!r}"
                "（norms.VERBS / norms.SPECIAL_FORMS のいずれにもありません）"
            )

        out: List[_Candidate] = []
        sec_son = norms.cite("form.sonkeigo").section
        sec_ken1 = norms.cite("form.kenjougo1").section
        sec_ken2 = norms.cite("class.kenjougo2").section
        sec_cond = norms.cite("form.kenjougo1_condition").section
        sec_self = norms.cite("principle.self_not_raised").section
        sec_17 = norms.cite("variation.itadaku_kudasaru").section
        sec_link = norms.cite("keigo_link").section
        sec_third = norms.cite("principle.third_party").section

        third_note = (
            f"（第三者については、相手から見て立てる対象かどうかに配慮する余地が"
            f"あります。{sec_third}）"
            if actor is Party.THIRD_PARTY
            else ""
        )

        if actor.is_self_side:
            # 指針 第2章第1-6【解説1】「自分側は立てない」。尊敬語は一切作らない。
            # 社内宛では身内の上位者を立てる言い方も案内されている（指針【25】）が、
            # Deference の Generator は宛先によらず自分側に尊敬語を作らない方針を採る。
            if target.is_other_side and v is not None:
                for form in v.kenjougo1_general():
                    # norms.kenjougo1_general() は「お(ご)……いただく」も返すが、
                    # その形の動作主は＜向かう先＞側（相手）である。動作主が自分側の
                    # ときは意味が合わないので、ここでは除いて相手側の集合に回す。
                    if form == f"{v.ogo}{v.ogo_stem}いただく":
                        continue
                    if form.endswith("申し上げる"):
                        origin, ckind = "kenjougo1_ogo_moushiageru", "ichidan"
                    else:
                        origin, ckind = "kenjougo1_ogo_suru", "sahen"
                    out.append(
                        _Candidate(
                            form, ckind, KeigoClass.KENJOUGO_1, origin,
                            f"自分側から{_ja(target)}に向かう行為なので、"
                            f"指針では謙譲語Ⅰの一般形が案内されています（{sec_ken1}）。",
                        )
                    )
                suru = f"{v.ogo}{v.ogo_stem}する" if v.ogo and v.has_target else ""
                if suru:
                    # 「ご案内してさしあげる」型。norms.ACCEPTABLE_KEIGO_LINKS が
                    # 許容される敬語連結として挙げている形をそのまま組む。
                    out.append(
                        _Candidate(
                            norms.te_form(suru, "sahen") + "差し上げる",
                            "ichidan", KeigoClass.KENJOUGO_1, "te_sashiageru",
                            f"許容される敬語連結として挙げられている形です（{sec_link}）。"
                            "場面によっては恩恵を押し出す響きになることがあります。",
                        )
                    )
            if target.is_other_side:
                for sf in norms.special_forms_for(base_verb, KeigoClass.KENJOUGO_1):
                    out.append(
                        _Candidate(
                            sf.form, sf.kind, KeigoClass.KENJOUGO_1,
                            "kenjougo1_special",
                            f"指針が挙げる謙譲語Ⅰの特定形です（{sec_ken1}）。",
                        )
                    )
            for sf in norms.special_forms_for(base_verb, KeigoClass.KENJOUGO_2):
                out.append(
                    _Candidate(
                        sf.form, sf.kind, KeigoClass.KENJOUGO_2, "kenjougo2_special",
                        f"自分側の行為を相手に丁重に述べる謙譲語Ⅱの特定形です（{sec_ken2}）。",
                    )
                )
            if v is not None:
                base = v.plain[:-2] if v.kind == "sahen" else ""
                ogo_itasu = f"{v.ogo}{base}いたす" if (base and v.ogo) else None
                for form in v.kenjougo2_general():
                    if ogo_itasu is not None and form == ogo_itasu:
                        out.append(
                            _Candidate(
                                form, "godan", KeigoClass.KENJOUGO_1_AND_2,
                                "kenjougo1and2_ogo_itasu",
                                "「お(ご)……いたす」は謙譲語Ⅰ兼謙譲語Ⅱの一般形です"
                                f"（{sec_ken1}／{sec_ken2}）。",
                            )
                        )
                    else:
                        out.append(
                            _Candidate(
                                form, "godan", KeigoClass.KENJOUGO_2,
                                "kenjougo2_itasu",
                                f"サ変動詞の謙譲語Ⅱの一般形です（{sec_ken2}）。",
                            )
                        )
        elif actor.is_other_side:
            for sf in norms.special_forms_for(base_verb, KeigoClass.SONKEIGO):
                out.append(
                    _Candidate(
                        sf.form, sf.kind, KeigoClass.SONKEIGO, "sonkeigo_special",
                        f"{_ja(actor)}の行為なので、指針が挙げる尊敬語の特定形が"
                        f"使えます（{sec_son}）。{third_note}",
                    )
                )
            if v is not None:
                for form in v.sonkeigo_general():
                    origin, ckind = _classify_sonkeigo(v, form)
                    out.append(
                        _Candidate(
                            form, ckind, KeigoClass.SONKEIGO, origin,
                            f"{_ja(actor)}の行為なので、指針では尊敬語の一般形が"
                            f"案内されています（{sec_son}）。{third_note}",
                        )
                    )
                kudasaru = v.sonkeigo_kudasaru()
                if kudasaru:
                    out.append(
                        _Candidate(
                            kudasaru, "godan", KeigoClass.SONKEIGO,
                            "sonkeigo_kudasaru",
                            "相手の行為を立てる「お(ご)……くださる」型です"
                            f"（{sec_17}：「いただく」と「くださる」はどちらも使えます）。",
                        )
                    )
                    # 「ご利用いただく」型。norms.kenjougo1_general() は三つの一般形を
                    # まとめて has_target で絞るが、指針【17】は＜向かう先＞を持たない
                    # 「利用する」についても「ご利用いただく」を適切としている。
                    # そこで「くださる」と同じ条件で、この形だけローカルに組む。
                    out.append(
                        _Candidate(
                            f"{v.ogo}{v.ogo_stem}いただく", "godan",
                            KeigoClass.KENJOUGO_1, "kenjougo1_ogo_itadaku",
                            "行為をする側を立てる「お(ご)……いただく」型です"
                            f"（{sec_17}：「くださる」とほぼ同じように使えます）。",
                        )
                    )
            if kind is not None:
                te = norms.te_form(base_verb, kind)
                out.append(
                    _Candidate(
                        te + "くださる", "godan", KeigoClass.SONKEIGO, "te_kudasaru",
                        f"{_ja(actor)}が行う「……てくださる」型（尊敬語）です"
                        f"（{sec_17}）。{third_note}",
                    )
                )
                out.append(
                    _Candidate(
                        te + "いただく", "godan", KeigoClass.KENJOUGO_1, "te_itadaku",
                        f"自分側が恩恵を受ける「……ていただく」型（謙譲語Ⅰ）です"
                        f"（{sec_17}）。",
                    )
                )

        # 丁寧語（素の動詞に「ます」）。立場によらず作れる。
        if kind is not None:
            if actor.is_self_side:
                note = (
                    "自分側の行為を丁寧語だけで述べる形です。"
                    f"自分側は立てないという原則には反しません（{sec_self}）。"
                )
            elif actor.is_other_side:
                note = (
                    "丁寧語だけで述べる形です。相手を特に立てない言い方になります"
                    "（指針 第3章第1-4「敬語は過剰でなく適度に使う」）。"
                )
            else:
                note = (
                    "動作主の立場が定まらないため、立場によらず使える丁寧語の形のみを"
                    "挙げています。"
                )
            out.append(
                _Candidate(base_verb, kind, KeigoClass.TEINEIGO, "teineigo_plain", note)
            )

        if v is not None and v.ogo and not v.has_target and actor.is_self_side:
            # 「ご乗車する」「ご利用する」が作れないことの説明を、候補が痩せる理由として
            # 残しておく（指針【補足イ－1】）。候補そのものは作らない。
            pass

        # 並べ替え
        if actor.is_self_side:
            rank = _RANK_SELF
        elif actor.is_other_side:
            rank = _RANK_OTHER
        else:
            rank = ("teineigo_plain",)
        order = {o: i for i, o in enumerate(rank)}
        out.sort(key=lambda c: order.get(c.origin, len(rank)))
        _ = sec_cond  # 引用キーの存在確認（未使用でも参照して取りこぼしを防ぐ）
        return out

    # -- 公開API ----------------------------------------------------------

    def correct_forms(
        self,
        base_verb: str,
        actor: Party,
        target: Party = Party.UNKNOWN,
        politeness: int = 1,
        audience: Audience = Audience.EXTERNAL,
    ) -> List[Suggestion]:
        """この立場の組合せで規範上作れる正しい述語形をすべて返す。

        **Corrector はこの集合の外から候補を出してはならない。**
        pytest がこの不変条件を検査する。

        規範（「敬語の指針」）に基づく構成規則:
            - 動作主が自分側（SELF / SELF_GROUP）なら尊敬語は作らない
              （第2章第1-6【解説1】「自分側は立てない」）。
            - 動作主が相手側・第三者なら尊敬語を作る（第2章第1-1）。
            - 謙譲語Ⅰの一般形「お(ご)……する」は＜向かう先＞のある動詞に限る
              （第2章第2-2【補足イ－1】）。＜向かう先＞が自分側のときは作らない。
            - 謙譲語Ⅱの一般形「……いたす」はサ変動詞のみ。
            - 「お(ご)……いただく」は行為をする側（＝相手側）を立てる形なので、
              動作主が相手側のときに作る（第3章第2-5【17】）。
            - 動作主の立場が不明なときは、立場によらず使える丁寧語の形だけを返す。

        Args:
            base_verb: 素の動詞（"言う" "利用する" など）。
            actor: 動作主の立場。
            target: ＜向かう先＞の立場。
            politeness: 0=常体 / 1=です・ます / 2=より改まった形。
            audience: 宛先。文言の丁重さの目安に使う。

        実証する主張:
            - 「向きの誤り検出」: 返る集合が ``actor`` / ``target`` だけで反転する。
            - 「根拠提示」: 各候補の ``reason`` に指針の該当箇所が入る。
            - 「速度」: 純粋な文字列処理のみ。
        """
        pol = _clamp_politeness(politeness)
        cands = self._candidates(base_verb, actor, target, audience)
        if pol >= 2 and actor.is_other_side:
            order = {o: i for i, o in enumerate(_RANK_OTHER_FORMAL)}
            cands = sorted(
                cands, key=lambda c: order.get(c.origin, len(_RANK_OTHER_FORMAL))
            )

        out: List[Suggestion] = []
        seen: set[str] = set()
        for c in cands:
            text = self._finalize(c, pol, actor)
            if text in seen:
                continue
            seen.add(text)
            klass = self._effective_class(c, pol, actor)
            reason = c.reason
            if pol >= 2 and audience in (Audience.EXTERNAL, Audience.PUBLIC):
                reason += "（社外宛てのため、より改まった言い方にしています。）"
            out.append(
                Suggestion(
                    text=text,
                    keigo_class=klass,
                    reason=reason,
                    generated_by="rule",
                )
            )
        return out

    def _effective_class(
        self, cand: _Candidate, politeness: int, actor: Party
    ) -> KeigoClass:
        """敬意度を反映した実際の分類を返す。"""
        if cand.origin == "teineigo_plain":
            if politeness <= 0:
                return KeigoClass.PLAIN
            if politeness >= 2 and actor.is_self_side:
                # 「……ております」は自分側の行為を丁重に述べる形（謙譲語Ⅱ）。
                return KeigoClass.KENJOUGO_2
            return KeigoClass.TEINEIGO
        return cand.keigo_class

    def _finalize(self, cand: _Candidate, politeness: int, actor: Party) -> str:
        """終止形に敬意度を反映した表層を作る。

        0=常体（終止形のまま）、1=「ます／です」、
        2=より改まった形（「……ております」「……ていらっしゃいます」）。
        """
        if politeness <= 0:
            return cand.form
        if politeness == 1 or cand.kind == "copula":
            return _polite(cand.form, cand.kind)

        # politeness == 2
        if cand.keigo_class is KeigoClass.SONKEIGO:
            if cand.origin == "sonkeigo_ogo_ni_naru":
                # 「お読みになっていらっしゃる」は norms.ACCEPTABLE_KEIGO_LINKS が
                # 許容される敬語連結として挙げている形。
                return norms.te_form(cand.form, cand.kind) + "いらっしゃいます"
            return _polite(cand.form, cand.kind)
        if cand.origin == "teineigo_plain" and not actor.is_self_side:
            # 「……ております」は自分側の行為に使う形なので、相手側の行為には付けない。
            return _polite(cand.form, cand.kind)
        if "おる" in cand.form or "いらっしゃ" in cand.form:
            return _polite(cand.form, cand.kind)
        return norms.te_form(cand.form, cand.kind) + "おります"

    # -- 文の生成 ----------------------------------------------------------

    def generate_sentence(
        self,
        *,
        base_verb: str,
        actor: Party,
        target: Party,
        politeness: int = 1,
        function: FunctionTag = FunctionTag.REPORT,
        context: MailContext | None = None,
    ) -> GeneratedSentence:
        """立場・敬意度・機能から規範に沿った1文を組み立てる。

        文中に現れる人物は必ず ``context.persons`` に登録して返す。単文だけを見ても
        敬語の適否は決まらないため、文と文脈を切り離さない。

        指定した機能に合う文型が無い場合は報告文型に落とし、
        ``meta["requested_function"]`` に元の指定を残す。

        実証する主張:
            - 「向きの誤り検出」: 生成された文は立場情報と対で保存されるので、
              検出器が文脈を使えているかを直接測れる。
            - 「過剰指摘の少なさ」: 出力はすべて正例であり、指摘が出れば偽陽性。
            - 「根拠提示」: ``meta["reason"]`` に採用した語形の根拠が入る。
        """
        ctx = context or default_context()
        pol = _clamp_politeness(politeness)
        cands = self._candidates(base_verb, actor, target, ctx.audience)

        tpl, cand = self._select_template(function, base_verb, actor, cands, pol)
        if tpl is None or cand is None:
            tpl, cand = self._select_template(
                FunctionTag.REPORT, base_verb, actor, cands, pol
            )
        if tpl is None or cand is None:
            raise ValueError(
                f"文型を選べませんでした: {base_verb!r} / actor={actor.value}"
            )

        used_pol = max(tpl.min_politeness, pol)
        predicate = self._realize(cand, used_pol, tpl.ending, actor)

        ctx2, subj = self._person_for(actor, ctx, base_verb, function)
        you = f"{ctx2.recipient_name}{ctx2.recipient_title}"
        noun, particle = _OBJECTS.get(base_verb, ("", ""))
        fields = {
            "subj": subj,
            "you": you,
            "obj": f"{noun}{particle}" if noun else "",
            "objn": noun or "本件",
        }
        pre = tpl.pre.format(**fields)
        post = tpl.post.format(**fields)
        text = pre + predicate + post
        span = Span(len(pre), len(pre) + len(predicate), predicate)

        return GeneratedSentence(
            text=text,
            context=ctx2,
            actor=actor,
            target=target,
            predicate=base_verb,
            keigo_class=self._effective_class(cand, used_pol, actor),
            politeness=used_pol,
            function=tpl.function,
            predicate_span=span,
            meta={
                "origin": cand.origin,
                "dictionary_form": cand.form,
                "reason": cand.reason,
                "requested_function": function.value,
                "requested_politeness": pol,
                "audience": ctx2.audience.value,
                "generated_by": "rule",
            },
        )

    def _select_template(
        self,
        function: FunctionTag,
        base_verb: str,
        actor: Party,
        cands: Sequence[_Candidate],
        politeness: int,
    ) -> Tuple[Optional[_Template], Optional[_Candidate]]:
        """機能タグに合い、かつ候補が揃っている文型を1つ選ぶ。"""
        by_origin: Dict[str, _Candidate] = {}
        for c in cands:
            by_origin.setdefault(c.origin, c)

        usable: List[Tuple[_Template, _Candidate]] = []
        for t in _TEMPLATES:
            if t.function is not function:
                continue
            if t.actor_side == "self" and not actor.is_self_side:
                continue
            if t.actor_side == "other" and not actor.is_other_side:
                continue
            if t.verbs and base_verb not in t.verbs:
                continue
            for o in t.prefer:
                if o in by_origin:
                    usable.append((t, by_origin[o]))
                    break
        if not usable:
            return None, None
        rng = self._rng(f"tpl|{function.value}|{base_verb}|{actor.value}|{politeness}")
        return usable[rng.randrange(len(usable))]

    def _realize(
        self, cand: _Candidate, politeness: int, ending: str, actor: Party
    ) -> str:
        """文末の機能（依頼・疑問・可能など）に合わせて述語を作る。"""
        form, kind = cand.form, cand.kind
        if ending == "declarative":
            return self._finalize(cand, politeness, actor)
        if ending == "question":
            return _polite(form, kind) + "か"
        if ending == "renyou":
            # 連用中止。「くださる」は連用中止が「くださり」で「ください」と衝突するため
            # テンプレート側で「いただく」型に限定してある。
            return _masu_stem(form, kind)
        if ending == "potential":
            # 指針 第2章第2-1(1)② 「まず尊敬語の形にした上で可能の形にする」。
            return _polite(norms.potential(form, kind), "ichidan")
        if ending == "kane":
            return _masu_stem(form, kind) + "かねます"
        if ending == "request":
            if cand.origin in ("sonkeigo_kudasaru", "te_kudasaru"):
                if politeness >= 2:
                    return _polite(form, kind) + "ようお願い申し上げます"
                return _masu_stem(form, kind)  # 「ご確認ください」
            if politeness >= 2:
                return _polite(form, kind) + "ようお願い申し上げます"
            return _polite(norms.potential(form, kind), "ichidan") + "と幸いです"
        raise ValueError(f"未知の文末種別です: {ending!r}")

    def _person_for(
        self, actor: Party, ctx: MailContext, base_verb: str, function: FunctionTag
    ) -> Tuple[MailContext, str]:
        """動作主の人物を文脈に登録し、「〜が」の主語句を返す。

        社外宛てでは身内に役職を付けない（指針 第3章第3-2【24】）。
        """
        if actor is Party.SELF:
            # 書き手自身は主語を省くのが自然だが、省くと「誰の行為か」が
            # 文面から決まらなくなる。実務のメールでも「私が対応いたします」の
            # ように明示することは多いので、主語を要求するテンプレートでは
            # 「私が」を出す（{subj} を使わないテンプレートには影響しない）。
            return ctx, "私が" 
        if actor is Party.ADDRESSEE:
            person = Person(ctx.recipient_name, Party.ADDRESSEE,
                            ctx.recipient_title, ctx.recipient_org)
            return (
                _ensure_person(ctx, person),
                f"{ctx.recipient_name}{ctx.recipient_title}が",
            )
        if actor is Party.UNKNOWN:
            return ctx, ""

        pool = _NAME_POOL.get(actor, ("担当者",))
        rng = self._rng(f"name|{actor.value}|{base_verb}|{function.value}")
        name = pool[rng.randrange(len(pool))]
        if actor is Party.SELF_GROUP:
            if ctx.audience is Audience.INTERNAL:
                title = _INTERNAL_TITLES[rng.randrange(len(_INTERNAL_TITLES))]
                display = f"{name}{title}"
            else:
                title = ""
                display = f"{ctx.writer_org}の{name}"
            person = Person(name, Party.SELF_GROUP, title, ctx.writer_org)
        elif actor is Party.ADDRESSEE_GROUP:
            title = "様"
            display = (
                f"{ctx.recipient_org}の{name}{title}"
                if ctx.recipient_org
                else f"{name}{title}"
            )
            person = Person(name, Party.ADDRESSEE_GROUP, title, ctx.recipient_org)
        else:  # THIRD_PARTY
            title = "様"
            display = f"{name}{title}"
            person = Person(name, Party.THIRD_PARTY, title, "")
        return _ensure_person(ctx, person), f"{display}が"

    # -- コーパス ----------------------------------------------------------

    def generate_corpus(self, n: int = 2000) -> List[GeneratedSentence]:
        """立場×述語×敬意度×機能を体系的に組み合わせた正例コーパス。

        機能 → 動作主 → 敬意度 → 動詞 の順に直積を走査し、その立場で作れない
        組合せを飛ばしながら ``n`` 件になるまで集める。走査順は決定的なので、
        同じ ``seed`` なら常に同じコーパスが得られる。

        実証する主張:
            - 「過剰指摘の少なさ」: これがそのまま偽陽性を測る負例集合になる。
            - 「向きの誤り検出」: 立場の全組合せを覆うので、向きに依存する語形が
              まんべんなく入る。
            - 「速度」: 数千文の生成が外部依存なしに済むことを示す。
        """
        if n <= 0:
            return []
        actors = (
            Party.SELF,
            Party.ADDRESSEE,
            Party.SELF_GROUP,
            Party.ADDRESSEE_GROUP,
            Party.THIRD_PARTY,
        )
        audiences = (Audience.EXTERNAL, Audience.INTERNAL, Audience.PUBLIC)
        politenesses = (1, 2, 0)
        out: List[GeneratedSentence] = []
        seen_text: set[str] = set()

        combos = list(
            itertools.product(tuple(FunctionTag), actors, politenesses, self.base_verbs)
        )
        # 直積の走査順のままだと、n を小さくしたときに動作主の後ろの方
        # （SELF_GROUP など）が丸ごと落ちる。立場ごとの被覆は評価の前提なので、
        # seed 固定でシャッフルしてどの n でも立場が均等に混ざるようにする。
        random.Random(f"corpus|{self.seed}").shuffle(combos)
        for idx, (function, actor, pol, base_verb) in enumerate(combos):
            if len(out) >= n:
                break
            target = _default_target(actor)
            ctx = default_context(audiences[idx % len(audiences)])
            try:
                s = self.generate_sentence(
                    base_verb=base_verb,
                    actor=actor,
                    target=target,
                    politeness=pol,
                    function=function,
                    context=ctx,
                )
            except (KeyError, ValueError):
                continue
            if s.meta.get("requested_function") != s.function.value:
                # 機能が落ちた組合せは網羅性の観点で価値が薄いので飛ばす。
                continue
            if s.text in seen_text:
                continue
            seen_text.add(s.text)
            out.append(s)

        if len(out) < n and out:
            # 直積を使い切っても足りないときは、宛先を変えて第2周を回す。
            extra_audiences = (Audience.INTERNAL, Audience.PUBLIC, Audience.EXTERNAL)
            combos2 = list(
                itertools.product(
                    tuple(FunctionTag), actors, politenesses, self.base_verbs
                )
            )
            random.Random(f"corpus2|{self.seed}").shuffle(combos2)
            for idx, (function, actor, pol, base_verb) in enumerate(combos2):
                if len(out) >= n:
                    break
                ctx = default_context(extra_audiences[idx % len(extra_audiences)])
                try:
                    s = self.generate_sentence(
                        base_verb=base_verb,
                        actor=actor,
                        target=_alt_target(actor),
                        politeness=pol,
                        function=function,
                        context=ctx,
                    )
                except (KeyError, ValueError):
                    continue
                if s.text in seen_text:
                    continue
                seen_text.add(s.text)
                out.append(s)
        return out[:n]

    # -- メール全体 --------------------------------------------------------

    def generate_mail(
        self,
        *,
        context: MailContext,
        n_sentences: int = 5,
        function: FunctionTag | None = None,
    ) -> GeneratedSentence:
        """宛名・本文・結びを備えたメール本文全体を生成する（text は複数文）。

        単文ではなくメール全体で判定するという設計上、本文全体を持つサンプルが
        必須である。``meta["sentences"]`` に1文ごとの内訳（絶対オフセットの
        スパン・立場・機能・分類）を入れる。

        Args:
            context: 宛先や人物を含むメールの文脈。
            n_sentences: 本文の文数（あいさつと結びは別に付く）。
            function: 本文の機能を固定したいときに指定する。省略時は
                宛先に応じた既定の並び（あいさつ→報告→依頼→感謝…）を使う。

        実証する主張:
            - 「向きの誤り検出」: 1通の中に自分側の行為と相手側の行為が混在する
              状況を作れる。向きの判定には文をまたいだ文脈が要ることを示す。
            - 「過剰指摘の少なさ」: メール1通まるごとが正例なので、通単位の
              偽陽性率（1通あたり何件の誤指摘が出るか）を測れる。
            - 「根拠提示」: 各文の根拠を ``meta["sentences"][i]["reason"]`` に残す。
        """
        n_body = max(1, int(n_sentences))
        rng = self._rng(f"mail|{context.audience.value}|{context.recipient_name}|{n_body}")

        plan = self._mail_plan(context, n_body, function, rng)
        ctx = context
        body: List[GeneratedSentence] = []
        for base_verb, actor, target, fn, pol in plan:
            try:
                s = self.generate_sentence(
                    base_verb=base_verb, actor=actor, target=target,
                    politeness=pol, function=fn, context=ctx,
                )
            except (KeyError, ValueError):
                continue
            # 人物が追加された文脈を引き継ぎ、1通の中で人物表を一貫させる。
            ctx = s.context
            body.append(s)
        if not body:
            raise ValueError("本文を1文も生成できませんでした。")

        head = _mail_salutation(ctx)
        greeting = _mail_greeting(ctx)
        closing = _mail_closing(ctx)

        parts: List[str] = [head, "", greeting]
        offsets: List[int] = []
        cursor = sum(len(p) + 1 for p in parts)  # 各要素の後に "\n"
        for s in body:
            offsets.append(cursor)
            parts.append(s.text)
            cursor += len(s.text) + 1
        parts.append("")
        parts.append(closing)
        text = "\n".join(parts)

        sentences_meta: List[Dict[str, object]] = []
        for off, s in zip(offsets, body):
            span = s.predicate_span
            abs_span = None
            if span is not None:
                abs_span = span.shifted(off).bind(text)
            sentences_meta.append(
                {
                    "text": s.text,
                    "offset": off,
                    "actor": s.actor.value,
                    "target": s.target.value,
                    "predicate": s.predicate,
                    "keigo_class": s.keigo_class.value,
                    "keigo_class_ja": KEIGO_CLASS_JA[s.keigo_class],
                    "politeness": s.politeness,
                    "function": s.function.value,
                    "predicate_span": (
                        [abs_span.start, abs_span.end, abs_span.text]
                        if abs_span is not None
                        else None
                    ),
                    "reason": s.meta.get("reason", ""),
                    "origin": s.meta.get("origin", ""),
                }
            )

        primary = body[0]
        primary_span = None
        if primary.predicate_span is not None:
            primary_span = primary.predicate_span.shifted(offsets[0]).bind(text)

        return GeneratedSentence(
            text=text,
            context=ctx,
            actor=primary.actor,
            target=primary.target,
            predicate=primary.predicate,
            keigo_class=primary.keigo_class,
            politeness=primary.politeness,
            function=primary.function,
            predicate_span=primary_span,
            meta={
                "kind": "mail",
                "salutation": head,
                "greeting": greeting,
                "closing": closing,
                "n_sentences": len(body),
                "sentences": sentences_meta,
                "audience": ctx.audience.value,
                "generated_by": "rule",
            },
        )

    def _mail_plan(
        self,
        context: MailContext,
        n_body: int,
        function: Optional[FunctionTag],
        rng: random.Random,
    ) -> List[Tuple[str, Party, Party, FunctionTag, int]]:
        """本文の各文に割り当てる〈動詞・立場・機能・敬意度〉を決める。"""
        pol = 1 if context.audience is Audience.INTERNAL else 2
        if function is not None:
            order = [function] * n_body
        else:
            base_order = [
                FunctionTag.THANKS,
                FunctionTag.REPORT,
                FunctionTag.REQUEST,
                FunctionTag.SCHEDULING,
                FunctionTag.NOTICE,
                FunctionTag.INQUIRY,
                FunctionTag.GIVING,
                FunctionTag.RECEIVING,
                FunctionTag.APOLOGY,
                FunctionTag.REFUSAL,
            ]
            order = [base_order[i % len(base_order)] for i in range(n_body)]

        #: 機能ごとに、自然に収まる〈動詞, 動作主〉の候補。
        pool: Dict[FunctionTag, Tuple[Tuple[str, Party], ...]] = {
            FunctionTag.THANKS: (("協力する", Party.ADDRESSEE),
                                 ("対応する", Party.ADDRESSEE),
                                 ("確認する", Party.ADDRESSEE)),
            FunctionTag.REPORT: (("連絡する", Party.SELF),
                                 ("報告する", Party.SELF),
                                 ("送付する", Party.SELF)),
            FunctionTag.REQUEST: (("確認する", Party.ADDRESSEE),
                                  ("検討する", Party.ADDRESSEE),
                                  ("記入する", Party.ADDRESSEE)),
            FunctionTag.SCHEDULING: (("出席する", Party.ADDRESSEE),
                                     ("訪問する", Party.SELF),
                                     ("参加する", Party.ADDRESSEE)),
            FunctionTag.NOTICE: (("案内する", Party.SELF),
                                 ("知らせる", Party.SELF),
                                 ("利用する", Party.ADDRESSEE)),
            FunctionTag.INQUIRY: (("確認する", Party.ADDRESSEE),
                                  ("聞く", Party.SELF)),
            FunctionTag.GIVING: (("送る", Party.SELF), ("送付する", Party.SELF)),
            FunctionTag.RECEIVING: (("送る", Party.ADDRESSEE),
                                    ("もらう", Party.SELF)),
            FunctionTag.APOLOGY: (("詫びる", Party.SELF),),
            FunctionTag.REFUSAL: (("対応する", Party.SELF),),
            FunctionTag.GREETING: (("願う", Party.SELF),),
        }
        plan: List[Tuple[str, Party, Party, FunctionTag, int]] = []
        for fn in order:
            choices = pool.get(fn) or (("連絡する", Party.SELF),)
            base_verb, actor = choices[rng.randrange(len(choices))]
            plan.append((base_verb, actor, _default_target(actor), fn, pol))
        return plan


# ---------------------------------------------------------------------------
# 補助関数
# ---------------------------------------------------------------------------


def _ja(party: Party) -> str:
    from .types import PARTY_JA

    return PARTY_JA[party]


def _clamp_politeness(politeness: int) -> int:
    return max(0, min(2, int(politeness)))


def _default_target(actor: Party) -> Party:
    """動作主から自然な＜向かう先＞を決める。"""
    if actor.is_self_side:
        return Party.ADDRESSEE
    if actor.is_other_side:
        return Party.SELF
    return Party.UNKNOWN


def _alt_target(actor: Party) -> Party:
    """コーパス第2周で使う別の＜向かう先＞。"""
    if actor.is_self_side:
        return Party.ADDRESSEE_GROUP
    if actor.is_other_side:
        return Party.SELF_GROUP
    return Party.UNKNOWN


def _classify_sonkeigo(v: Verb, form: str) -> Tuple[str, str]:
    """norms.Verb.sonkeigo_general() の各要素を〈由来, 活用種別〉に割る。

    norms 側が組み立てる形をそのまま組み直して照合するので、
    表層の当てずっぽうな部分一致には頼らない。
    """
    if v.ogo and form == f"{v.ogo}{v.ogo_stem}になる":
        return "sonkeigo_ogo_ni_naru", "godan"
    if v.ogo and form == f"{v.ogo}{v.ogo_stem}だ":
        return "sonkeigo_ogo_da", "copula"
    if form.endswith("なさる"):
        return "sonkeigo_nasaru", "godan"
    # 残るのは「……(ら)れる／……される／来られる」。いずれも下一段。
    return "sonkeigo_rareru", "ichidan"


def _ensure_person(ctx: MailContext, person: Person) -> MailContext:
    """人物を文脈に登録済みにする（同名があれば何もしない）。"""
    for p in ctx.persons:
        if p.name == person.name and p.side is person.side:
            return ctx
    return ctx.with_persons(list(ctx.persons) + [person])


def _mail_salutation(ctx: MailContext) -> str:
    """宛名の行。"""
    name = f"{ctx.recipient_name}{ctx.recipient_title}"
    if ctx.audience is Audience.EXTERNAL and ctx.recipient_org:
        return f"{ctx.recipient_org}\n{name}"
    return name


def _mail_greeting(ctx: MailContext) -> str:
    """書き出しのあいさつ。宛先で言い方を変える。"""
    if ctx.audience is Audience.INTERNAL:
        return f"お疲れ様です。{ctx.writer_org}の{ctx.writer_name}です。"
    if ctx.audience is Audience.PUBLIC:
        return "平素より格別のご高配を賜り、厚く御礼申し上げます。"
    return (
        "いつも大変お世話になっております。"
        f"{ctx.writer_org}の{ctx.writer_name}でございます。"
    )


def _mail_closing(ctx: MailContext) -> str:
    """結びの一文。"""
    if ctx.audience is Audience.INTERNAL:
        return "よろしくお願いします。"
    return "何卒よろしくお願い申し上げます。"
