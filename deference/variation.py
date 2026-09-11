"""揺れ（許容度が割れる表現）の登録簿。誤りとは別のラベルで扱う。

Deference の実用性はこのモジュールに懸かっている。指針自身が「習慣として定着して
いる」「許容される」「個人差が大きい」と述べている表現、および世代差・場面差で
許容度が割れる表現を誤りとして赤く出すと、ツールは信頼されない。ここに集めた
表現は :class:`~deference.types.Verdict` の ``VARIATION`` に落とし、既定では
指摘しない。

方針:
    - 判定の根拠は :mod:`deference.norms` から引く。指針に基づく引用は
      ``norms.cite(...)`` を通し、このモジュールで指針の語形・分類・引用文を
      新たにハードコードしない。
    - 指針の射程外（接客表現・世代差・ら抜きなど）は :data:`OUTSIDE_SOURCE` を
      出典に持つ別建ての :class:`~deference.types.Citation` を付け、
      **指針を根拠に見せかけない**。
    - 利用者に見える文言（reason / note）は「規範上はこうなる」「指針では
      〜と案内されている」という情報提示に留め、断定的に否定しない。

実証する主張との対応:
    - 「過剰指摘の少なさ」: 本モジュールが最終フィルタとして働き、揺れに対する
      指摘を抑制する。:meth:`VariationSet.cases` はそのまま偽陽性評価用の
      データセットになる。
    - 「根拠提示」: 各ケースは指針の該当箇所（または射程外である旨）を
      :class:`~deference.types.Citation` として持つ。
    - 「速度」: 照合は正規表現を使わず、先頭文字で候補を絞る最左最長一致の
      1パス走査で行う。本文長に対して線形で、外部依存も持たない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import norms
from .types import (
    Audience,
    Citation,
    ErrorType,
    MailContext,
    Party,
    Person,
    Span,
    VariationCase,
)

__all__ = [
    "OUTSIDE_SOURCE",
    "NOT_VARIATION",
    "VariationSet",
    "default_variation_set",
]


# ---------------------------------------------------------------------------
# 指針の射程外である旨を明示するための出典
# ---------------------------------------------------------------------------

#: 指針に該当記述がない事項に付ける出典名。指針を根拠に見せかけないための仕切り。
OUTSIDE_SOURCE = "指針の射程外（Deference の整理）"


def _outside(section: str, note: str) -> Citation:
    """指針の射程外であることを明示する :class:`Citation` を作る。

    ``url`` を空にして指針 PDF へのリンクを付けず、``quote`` も空にして
    指針本文の引用があるかのように見せない。

    実証する主張: 「根拠提示」。根拠の強さ（指針の明記か、Deference 側の整理か）を
    出典レベルで区別できるようにする。
    """
    return Citation(
        source=OUTSIDE_SOURCE,
        section=section,
        page="—",
        quote="",
        url="",
        note=note,
    )


# ---------------------------------------------------------------------------
# 揺れに **入れない** もの（誤り側に置く）
# ---------------------------------------------------------------------------

#: 二重敬語のうち、指針 p.30 の【習慣として定着している二重敬語の例】に
#: 挙がっていないもの。世間での使用頻度は低くないが、指針が定着を認めていない
#: 以上は揺れ扱いにせず、規範からの分岐（誤り側）に置く。
#: :data:`norms.ESTABLISHED_DOUBLE_KEIGO` に載る語だけを揺れとする、という
#: 線引きを機械的に守るため明示的に列挙し、:meth:`VariationSet.self_check` で
#: 「これらに反応する登録表現が一つもない」ことを確かめる。
NOT_VARIATION: Tuple[str, ...] = (
    "お帰りになられる",
    "お読みになられる",
    "ご覧になられる",
    "おっしゃられる",
    "お召し上がりになられる",
    "お見えになられる",
    "伺わせていただきます",
)


# ---------------------------------------------------------------------------
# 想定場面（MailContext）
# ---------------------------------------------------------------------------

_EXTERNAL = MailContext(
    audience=Audience.EXTERNAL,
    writer_name="山田",
    writer_org="株式会社アルファ",
    recipient_name="佐藤",
    recipient_title="様",
    recipient_org="株式会社ベータ",
)

_INTERNAL = MailContext(
    audience=Audience.INTERNAL,
    writer_name="山田",
    writer_org="営業部",
    recipient_name="鈴木",
    recipient_title="課長",
    recipient_org="営業部",
)

_PUBLIC = MailContext(
    audience=Audience.PUBLIC,
    writer_name="事務局",
    writer_org="株式会社アルファ",
    recipient_name="",
    recipient_title="各位",
    recipient_org="",
)

#: 接客・店頭の場面。宛先は客なので社外扱い。
_COUNTER = MailContext(
    audience=Audience.EXTERNAL,
    writer_name="店員",
    writer_org="株式会社アルファ",
    recipient_name="お客様",
    recipient_title="様",
    recipient_org="",
)

#: 学校場面（指針【23】）。同僚の田中教諭は書き手にとって「ウチ」の人。
_SCHOOL = MailContext(
    audience=Audience.EXTERNAL,
    writer_name="山田",
    writer_title="教諭",
    writer_org="第一中学校",
    recipient_name="佐藤",
    recipient_title="様",
    recipient_org="保護者",
    persons=(Person(name="田中", side=Party.SELF_GROUP, title="先生", org="第一中学校"),),
)

#: 上司の上司に宛てる場面（指針【26】）。鈴木課長は書き手より上位だが、
#: 宛先の高橋部長から見れば同じ「ウチ」の人物である。
_MIDDLE_BOSS = MailContext(
    audience=Audience.INTERNAL,
    writer_name="山田",
    writer_org="営業部",
    recipient_name="高橋",
    recipient_title="部長",
    recipient_org="営業部",
    persons=(Person(name="鈴木", side=Party.SELF_GROUP, title="課長", org="営業部"),),
)


# ---------------------------------------------------------------------------
# 登録簿の項目
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Entry:
    """揺れ1件の内部表現。:class:`VariationCase` に展開して外へ出す。

    ``surfaces`` は本文照合に使う表層文字列、``example`` と ``focus`` は
    評価データとしての例文と注目箇所である。``suppressed`` が空のときは
    どの誤り種別の指摘も抑制する。

    実証する主張: 「過剰指摘の少なさ」。抑制の範囲（どの誤り種別を、どの宛先で）を
    データとして持ち、コードに散らさない。
    """

    key: str
    surfaces: Tuple[str, ...]
    example: str
    focus: str
    reason: str
    citation: Optional[Citation]
    acceptability: str  # established / split / shifting
    context: MailContext
    related: ErrorType = ErrorType.NONE
    suppressed: Tuple[ErrorType, ...] = ()
    audiences: Optional[Tuple[Audience, ...]] = None
    scope: str = "shishin"  # shishin（指針が根拠） / outside（指針の射程外）
    note: str = ""


def _e(
    key: str,
    surfaces: Tuple[str, ...],
    example: str,
    focus: str,
    reason: str,
    citation: Optional[Citation],
    acceptability: str,
    context: MailContext,
    related: ErrorType = ErrorType.NONE,
    suppressed: Tuple[ErrorType, ...] = (),
    audiences: Optional[Tuple[Audience, ...]] = None,
    scope: str = "shishin",
    note: str = "",
) -> _Entry:
    return _Entry(
        key=key,
        surfaces=surfaces,
        example=example,
        focus=focus,
        reason=reason,
        citation=citation,
        acceptability=acceptability,
        context=context,
        related=related,
        suppressed=suppressed,
        audiences=audiences,
        scope=scope,
        note=note,
    )


# 参照の短縮。引用はすべて norms.CITATIONS から引く（このモジュールでは作らない）。
_CITE = norms.cite

_DK = ErrorType.DOUBLE_KEIGO
_LINK = ErrorType.BAD_KEIGO_LINK
_SWAP = ErrorType.DIRECTION_SWAP
_UCHI = ErrorType.UCHI_SONKEIGO
_SELF = ErrorType.SELF_SONKEIGO
_SASETE = ErrorType.SASETE_ITADAKU_OVERUSE
_STYLE = ErrorType.STYLE_MIXING
_INV = ErrorType.DEFERENCE_INVERSION


# ---------------------------------------------------------------------------
# A. 指針自身が「定着」「許容」「問題ない」と明記しているもの
# ---------------------------------------------------------------------------

_ENTRIES_ESTABLISHED: Tuple[_Entry, ...] = (
    _e(
        "double.o_ukagai",
        ("お伺い申し上げ", "お伺いいたし", "お伺いいたす", "お伺いする", "お伺いし"),
        "明日十四時に御社へお伺いいたします。",
        "お伺いいたします",
        "「お伺いする」型は二重敬語だが、指針は習慣として定着した例に挙げている。",
        _CITE("double_keigo.established"),
        "established",
        _EXTERNAL,
        related=_DK,
        suppressed=(_DK,),
    ),
    _e(
        "double.o_meshiagari",
        ("お召し上がりになる", "お召し上がりになり", "お召し上がりになっ",
         "お召し上がりになれ", "お召し上がりくださ"),
        "温かいうちにお召し上がりになるとよろしいかと存じます。",
        "お召し上がりになる",
        "「お召し上がりになる」は二重敬語だが、指針は習慣として定着した例に挙げている。",
        _CITE("double_keigo.established"),
        "established",
        _EXTERNAL,
        related=_DK,
        suppressed=(_DK,),
    ),
    _e(
        "double.o_mie",
        ("お見えになる", "お見えになり", "お見えになっ", "お見えです"),
        "先ほど佐藤様がお見えになりました。",
        "お見えになりました",
        "「お見えになる」は二重敬語だが、指針は習慣として定着した例に挙げている。",
        _CITE("double_keigo.established"),
        "established",
        _EXTERNAL,
        related=_DK,
        suppressed=(_DK,),
    ),
    _e(
        "link.oyomi_irassharu",
        ("お読みになっていらっしゃ",),
        "部長はもう資料をお読みになっていらっしゃいます。",
        "お読みになっていらっしゃいます",
        "二つの語をそれぞれ敬語にして「て」でつないだ敬語連結であり、"
        "指針は許容される例として挙げている。",
        _CITE("keigo_link"),
        "established",
        _INTERNAL,
        related=_LINK,
        suppressed=(_LINK, _DK),
    ),
    _e(
        "link.oyomi_kudasaru",
        ("お読みになってくださ",),
        "お忙しい中、原稿をお読みになってくださり、ありがとうございました。",
        "お読みになってくださり",
        "敬語連結として指針が許容される例に挙げている形。",
        _CITE("keigo_link"),
        "established",
        _EXTERNAL,
        related=_LINK,
        suppressed=(_LINK, _DK),
    ),
    _e(
        "link.oyomi_itadaku",
        ("お読みになっていただ",),
        "当日までに資料をお読みになっていただけますでしょうか。",
        "お読みになっていただけます",
        "敬語連結として指針が許容される例に挙げている形。",
        _CITE("keigo_link"),
        "established",
        _EXTERNAL,
        related=_LINK,
        suppressed=(_LINK, _DK),
    ),
    _e(
        "link.goannai_sashiageru",
        ("ご案内してさしあげ", "ご案内して差し上げ"),
        "受付から会場までご案内してさしあげてください。",
        "ご案内してさしあげて",
        "敬語連結として指針が許容される例に挙げている形。",
        _CITE("keigo_link"),
        "established",
        _INTERNAL,
        related=_LINK,
        suppressed=(_LINK, _DK, _SELF),
    ),
    _e(
        "q17.goriyou_itadaku",
        ("ご利用いただ", "御利用いただ"),
        "いつも当社のサービスをご利用いただき、ありがとうございます。",
        "ご利用いただき",
        "指針【17】は「ご利用いただく」と「ご利用くださる」をどちらもほぼ同じように"
        "使える敬語だとしている。受け止め方には個人差がある。",
        _CITE("variation.itadaku_kudasaru"),
        "established",
        _PUBLIC,
        related=_SWAP,
        suppressed=(_SWAP, _LINK, _SELF),
    ),
    _e(
        "q17.goriyou_kudasaru",
        ("ご利用くださ", "御利用くださ"),
        "いつも当社のサービスをご利用くださり、ありがとうございます。",
        "ご利用くださり",
        "指針【17】は「ご利用くださる」と「ご利用いただく」をどちらもほぼ同じように"
        "使える敬語だとしている。",
        _CITE("variation.itadaku_kudasaru"),
        "established",
        _PUBLIC,
        related=_SWAP,
        suppressed=(_SWAP, _LINK),
    ),
    _e(
        "q14.gojisan",
        ("ご持参くださ", "御持参くださ"),
        "当日は筆記用具をご持参ください。",
        "ご持参ください",
        "指針【14】は、この表現に含まれる「参る」は謙譲語Ⅱとしては働かないため、"
        "相手側の行為に用いて問題ないとしている。",
        _CITE("variation.jisan"),
        "established",
        _PUBLIC,
        related=_SWAP,
        suppressed=(_SWAP, _SELF, _LINK),
    ),
    _e(
        "q14.omoushide",
        ("お申し出くださ", "お申出くださ"),
        "ご不明な点がございましたら、お申し出ください。",
        "お申し出ください",
        "指針【14】は、この表現に含まれる「申す」は謙譲語Ⅱとしては働かないため、"
        "相手側の行為に用いて問題ないとしている。",
        _CITE("variation.jisan"),
        "established",
        _PUBLIC,
        related=_SWAP,
        suppressed=(_SWAP, _SELF, _LINK),
    ),
    _e(
        "q14.omoushikomi",
        ("お申し込みくださ", "お申込みくださ"),
        "参加をご希望の方は、下記のフォームからお申し込みください。",
        "お申し込みください",
        "指針【14】は、この表現に含まれる「申す」は謙譲語Ⅱとしては働かないため、"
        "相手側の行為に用いて問題ないとしている。",
        _CITE("variation.jisan"),
        "established",
        _PUBLIC,
        related=_SWAP,
        suppressed=(_SWAP, _SELF, _LINK),
    ),
    _e(
        "q15.moushitsutaeru",
        ("申し伝え",),
        "その旨、担当者に申し伝えます。",
        "申し伝えます",
        "指針【15】は、この「申す」に＜向かう先＞である担当者を立てる働きはないとし、"
        "問題ない表現としている。",
        _CITE("variation.jisan"),
        "established",
        _EXTERNAL,
        related=_SWAP,
        suppressed=(_SWAP, _INV),
    ),
    _e(
        "q16.omachi_shite",
        ("お待ちして", "お待ち申し上げ"),
        "ご連絡をお待ちしております。",
        "お待ちしております",
        "指針【16】は、自分側の動作でも＜向かう先＞を立てる謙譲語Ⅰであれば"
        "「お」を付けて問題ないとしている。",
        _CITE("self_sonkeigo.q16"),
        "established",
        _EXTERNAL,
        related=_SELF,
        suppressed=(_SELF, _SWAP),
    ),
    _e(
        "q16.gosetsumei_shitai",
        ("ご説明したい", "ご説明いたしたい", "御説明したい"),
        "その件について、ご説明したいのですが、お時間をいただけますか。",
        "ご説明したい",
        "指針【16】は、自分側の動作でも＜向かう先＞を立てる謙譲語Ⅰであれば"
        "「ご」を付けて問題ないとしている。",
        _CITE("self_sonkeigo.q16"),
        "established",
        _EXTERNAL,
        related=_SELF,
        suppressed=(_SELF, _SWAP),
    ),
    _e(
        "adj.takai_desu",
        ("高いです",),
        "今回のお見積りは、当初の想定より少し高いです。",
        "高いです",
        "形容詞に「です」を付ける形。指針は抵抗を感じる人もあるとしつつ、"
        "既にかなりの人が許容するようになってきていると述べている。",
        _CITE("variation.adj_desu"),
        "shifting",
        _EXTERNAL,
        related=_STYLE,
        suppressed=(_STYLE,),
    ),
    _e(
        "adj.muzukashii_desu",
        ("難しいです",),
        "今週中の対応は少し難しいです。",
        "難しいです",
        "形容詞に「です」を付ける形。指針は既にかなりの人が許容するようになって"
        "きていると述べている。",
        _CITE("variation.adj_desu"),
        "shifting",
        _INTERNAL,
        related=_STYLE,
        suppressed=(_STYLE,),
    ),
    _e(
        "ageru.mizu",
        ("水をあげ", "水を上げ"),
        "留守の間、植木に水をあげるのをお願いできますか。",
        "水をあげる",
        "指針は、この「あげる」の謙譲語から美化語への意味変化は既に進行し、"
        "定着しつつあると述べている。",
        _CITE("variation.ageru"),
        "shifting",
        _INTERNAL,
        related=_SWAP,
        suppressed=(),
    ),
    _e(
        "q27.otsukaresama_gozaimashita",
        ("お疲れ様でござい", "お疲れさまでござい"),
        "本日は遠方までお越しいただき、お疲れ様でございました。",
        "お疲れ様でございました",
        "指針【27】は、上位者へのねぎらいを避けたい場合の言い換えとして"
        "この形を挙げている。受け止め方には個人差がある。",
        _CITE("principle.moderate"),
        "established",
        _INTERNAL,
        related=ErrorType.NONE,
        suppressed=(),
    ),
)


# ---------------------------------------------------------------------------
# B. 指針が「個人差がある」「どちらの考え方にも理がある」としているもの
# ---------------------------------------------------------------------------

_ENTRIES_SPLIT: Tuple[_Entry, ...] = (
    _e(
        "q18.sasete_itadaku",
        ("させていただきます", "させていただきたく", "させていただければ",
         "させていただく", "させて頂きます"),
        "本日は、新製品の概要をご報告させていただきます。",
        "させていただきます",
        "指針【18】は、許可と恩恵の「見立て」をどの程度自然と受け止めるかが"
        "個人ごとの許容度を決めるとしており、一律に不適切とはしていない。",
        _CITE("sasete_itadaku.tolerance"),
        "split",
        _EXTERNAL,
        related=_SASETE,
        suppressed=(_SASETE,),
    ),
    _e(
        "q18.yasumasete",
        ("休ませていただ", "休まセていただ"),
        "勝手ながら、明日は休ませていただきます。",
        "休ませていただきます",
        "指針【18】が挙げる、許可を受けるという見立てが比較的立ちやすい用法。"
        "許容度には個人差がある。",
        _CITE("sasete_itadaku"),
        "split",
        _INTERNAL,
        related=_SASETE,
        suppressed=(_SASETE,),
    ),
    _e(
        "q18.tsukawasete",
        ("使わせていただ", "使わせて頂"),
        "頂戴した資料は、社内の検討に使わせていただきます。",
        "使わせていただきます",
        "指針【18】が挙げる、許可と恩恵の見立てが立つ用法。許容度には個人差がある。",
        _CITE("sasete_itadaku"),
        "split",
        _EXTERNAL,
        related=_SASETE,
        suppressed=(_SASETE,),
    ),
    _e(
        "q18.haiken_sasete",
        ("拝見させていただ",),
        "お送りいただいた企画書を拝見させていただきました。",
        "拝見させていただきました",
        "謙譲語Ⅰに「させていただく」を重ねた形。冗長と受け取る人もいるが、"
        "指針は見立ての受け入れ方に個人差があるとしている。",
        _CITE("sasete_itadaku.tolerance"),
        "split",
        _EXTERNAL,
        related=_SASETE,
        suppressed=(_SASETE, _DK),
    ),
    _e(
        "q23.sensei_sonkeigo",
        ("先生はいらっしゃ", "先生がいらっしゃ", "先生はおっしゃ", "先生がおっしゃ"),
        "田中先生はいらっしゃいますが、ただいま授業中です。",
        "いらっしゃいます",
        "指針【23】は「ウチ・ソト」の意識からは「田中はおりません」と伝えた方が"
        "良いとしつつ、世論調査では「田中先生」を支持する人が多いことも併記している。",
        _CITE("uchi_soto.q23"),
        "split",
        _SCHOOL,
        related=_UCHI,
        suppressed=(_UCHI, _INV, _SELF),
    ),
    _e(
        "q23.sensei_kenjougo2",
        ("先生は席を外して", "先生はおりません", "先生は本日休んでおり"),
        "あいにく田中先生は席を外しております。",
        "田中先生は席を外しております",
        "指針【23】が「ウチ・ソト」の観点から案内する形。ただし同僚を「先生」と"
        "呼ぶかどうかは受け止め方が分かれる。",
        _CITE("uchi_soto.q23"),
        "split",
        _SCHOOL,
        related=_UCHI,
        # 同じ現象を direction_swap や self_sonkeigo と読む検出器もある。
        # 揺れかどうかは「どのラベルが付いたか」ではなく「その表現が
        # 許容度の割れるものか」で決まるので、関連する種別をまとめて抑制する。
        suppressed=(_UCHI, _INV, _SWAP, _SELF),
    ),
    _e(
        "q26.chuukan_joushi",
        ("課長がおっしゃ", "課長はおっしゃ"),
        "その件は課長がおっしゃっていました。",
        "課長がおっしゃっていました",
        "指針【26】は、部長に対して課長を立てずに「申しておりました」と言えば良い"
        "としつつ、課長を立てる考え方にも理があるとしている。",
        _CITE("uchi_soto.q26"),
        "split",
        _MIDDLE_BOSS,
        related=_UCHI,
        # 同じ現象を direction_swap や self_sonkeigo と読む検出器もある。
        # 揺れかどうかは「どのラベルが付いたか」ではなく「その表現が
        # 許容度の割れるものか」で決まるので、関連する種別をまとめて抑制する。
        suppressed=(_UCHI, _INV, _SWAP, _SELF),
    ),
    _e(
        "q24.internal_buchou",
        ("部長は席を外していらっしゃ", "部長はお戻りになり", "部長がお戻りになり"),
        "田中部長はただいま席を外していらっしゃいます。",
        "席を外していらっしゃいます",
        "宛先が社内であれば、同じ組織の上位者を立てて述べることに問題はない。"
        "指針【24】【25】が示すとおり、適否は宛先が社内か社外かで入れ替わる。",
        _CITE("uchi_soto.q24"),
        "split",
        _INTERNAL,
        related=_UCHI,
        suppressed=(_UCHI, _INV),
        # 宛先が社外の場合は身内を立てることになるため、抑制を社内宛に限定する。
        # ここを無条件に抑制すると「向きの誤り検出」という主張そのものを壊す。
        audiences=(Audience.INTERNAL,),
    ),
)


# ---------------------------------------------------------------------------
# C. 世代差・場面差で許容度が変わる、指針の射程外のもの
#    （出典は OUTSIDE_SOURCE。指針を根拠に見せかけない）
# ---------------------------------------------------------------------------

_ENTRIES_OUTSIDE: Tuple[_Entry, ...] = (
    _e(
        "outside.ryoukai",
        ("了解しました", "了解いたしました", "了解です"),
        "ご依頼の件、了解しました。",
        "了解しました",
        "上位者や社外には「承知しました」を勧める案内が広く流通しているが、"
        "その使い分けを定めた規範はなく、受け止め方は世代や職場で分かれる。",
        _outside(
            "応答表現の使い分け（了解／承知／かしこまりました）",
            "「敬語の指針」はこの三語の使い分けを扱っていない。ビジネスマナー書に"
            "由来する慣習であり、Deference は規範として扱わない。",
        ),
        "split",
        _EXTERNAL,
        scope="outside",
    ),
    _e(
        "outside.shouchi",
        ("承知しました", "承知いたしました"),
        "ご依頼の件、承知しました。",
        "承知しました",
        "「承知しました」と「承知いたしました」の丁寧度の差は場面によって"
        "受け止め方が分かれる。",
        _outside(
            "応答表現の使い分け（了解／承知／かしこまりました）",
            "「敬語の指針」はこの三語の使い分けを扱っていない。",
        ),
        "split",
        _EXTERNAL,
        scope="outside",
    ),
    _e(
        "outside.kashikomari",
        ("かしこまりました", "畏まりました"),
        "ご依頼の件、かしこまりました。",
        "かしこまりました",
        "接客場面では標準的だが、社内メールでは硬いと受け取る人もいる。",
        _outside(
            "応答表現の使い分け（了解／承知／かしこまりました）",
            "「敬語の指針」はこの三語の使い分けを扱っていない。",
        ),
        "split",
        _COUNTER,
        scope="outside",
    ),
    _e(
        "outside.gokurousama",
        ("ご苦労様", "ご苦労さま", "御苦労様"),
        "遅くまでご苦労様です。",
        "ご苦労様です",
        "上位者に用いるのを避ける案内が広く流通する一方、実際の受け止め方は"
        "世代や地域で大きく分かれる。",
        _outside(
            "ねぎらい表現の世代差（ご苦労様／お疲れ様）",
            "指針【27】は上位者へのねぎらいを扱うが、「ご苦労様」と「お疲れ様」の"
            "優劣は述べていない。世代差の存在自体は指針 第1章第2-2 が指摘している。",
        ),
        "split",
        _INTERNAL,
        scope="outside",
    ),
    _e(
        "outside.otsukaresama",
        ("お疲れ様です", "お疲れさまです", "お疲れ様でした", "お疲れさまでした"),
        "お疲れ様です。営業部の山田です。",
        "お疲れ様です",
        "社内メールの書き出しとして定型化しているが、社外に用いるかどうかは"
        "職場ごとに慣習が分かれる。",
        _outside(
            "ねぎらい表現の世代差（ご苦労様／お疲れ様）",
            "「敬語の指針」はメール冒頭のあいさつ定型を扱っていない。",
        ),
        "split",
        _INTERNAL,
        scope="outside",
    ),
    _e(
        "outside.naruhodo_desune",
        ("なるほどですね",),
        "なるほどですね、その点は持ち帰って検討いたします。",
        "なるほどですね",
        "話し言葉で広まった相づち。耳慣れないと感じる人もいるが、"
        "規範として可否を定めた記述はない。",
        _outside(
            "いわゆる接客表現・新しい相づち",
            "「敬語の指針」に該当記述はない。",
        ),
        "shifting",
        _INTERNAL,
        scope="outside",
    ),
    _e(
        "outside.no_hou",
        ("のほうを", "のほうは", "のほうへ", "のほうに", "のほう、", "の方を",),
        "資料のほうをお送りいたします。",
        "資料のほう",
        "対象をぼかす「ほう」。冗長と感じる人もいるが、規範として可否を"
        "定めた記述はない。",
        _outside(
            "いわゆる接客表現（〜のほう）",
            "「敬語の指針」に該当記述はない。",
        ),
        "shifting",
        _COUNTER,
        scope="outside",
    ),
    _e(
        "outside.ni_narimasu",
        ("円になります", "資料になります", "以上になります"),
        "こちらが本日の資料になります。",
        "資料になります",
        "変化を表さない「なります」。冗長と感じる人もいるが、規範として"
        "可否を定めた記述はない。",
        _outside(
            "いわゆる接客表現（〜になります）",
            "「敬語の指針」に該当記述はない。",
        ),
        "shifting",
        _COUNTER,
        scope="outside",
    ),
    _e(
        "outside.tondemo",
        ("とんでもございません", "とんでもありません"),
        "とんでもございません、こちらこそ助かりました。",
        "とんでもございません",
        "「とんでもない」を分割した形。旧来の規範からは「とんでもないことでございます」"
        "が案内されてきたが、現在は広く使われ、受け止め方が分かれる。",
        _outside(
            "「とんでもございません」の許容度",
            "「敬語の指針」に該当記述はない。国語施策上も可否の断定はされていない。",
        ),
        "shifting",
        _EXTERNAL,
        related=_DK,
        scope="outside",
    ),
    _e(
        "outside.ranuki_mireru",
        ("見れます", "見れる", "見れない"),
        "こちらのページから資料が見れます。",
        "見れます",
        "いわゆる「ら抜き言葉」。書き言葉では「見られます」が案内されるが、"
        "話し言葉では定着が進んでおり、世代差が大きい。",
        _outside(
            "いわゆる「ら抜き言葉」",
            "「敬語の指針」は敬語の問題として扱っていない。可能表現の変化に"
            "関する事柄であり、Deference は規範違反として提示しない。",
        ),
        "shifting",
        _PUBLIC,
        scope="outside",
    ),
    _e(
        "outside.ranuki_tabereru",
        ("食べれます", "食べれる"),
        "会場では軽食も食べれます。",
        "食べれます",
        "いわゆる「ら抜き言葉」。書き言葉では「食べられます」が案内されるが、"
        "世代差が大きい。",
        _outside(
            "いわゆる「ら抜き言葉」",
            "「敬語の指針」は敬語の問題として扱っていない。",
        ),
        "shifting",
        _PUBLIC,
        scope="outside",
    ),
    _e(
        "outside.osewa",
        ("お世話になっており", "お世話になります"),
        "いつもお世話になっております。株式会社アルファの山田です。",
        "お世話になっております",
        "取引先へのメール冒頭の定型句。初回の連絡で使うかどうかなど、"
        "運用は職場ごとに分かれる。",
        _outside(
            "メール定型句（お世話になっております）",
            "「敬語の指針」はメールの定型句を扱っていない。",
        ),
        "split",
        _EXTERNAL,
        scope="outside",
    ),
    _e(
        "outside.toriisogi",
        ("取り急ぎ", "取急ぎ"),
        "取り急ぎ、ご報告まで。",
        "取り急ぎ",
        "略式である旨を示す定型句。相手によっては簡略に過ぎると受け取られる。",
        _outside(
            "メール定型句（取り急ぎ）",
            "「敬語の指針」はメールの定型句を扱っていない。",
        ),
        "split",
        _EXTERNAL,
        scope="outside",
    ),
    _e(
        "outside.yoroshiku",
        ("よろしくお願いいたします", "よろしくお願い致します", "宜しくお願いいたします"),
        "何卒よろしくお願いいたします。",
        "よろしくお願いいたします",
        "「申し上げます」との丁寧度の差をどこまで意識するかは、"
        "書き手と場面によって分かれる。",
        _outside(
            "メール定型句（よろしくお願いいたします）",
            "「敬語の指針」はこの定型句の丁寧度の序列を定めていない。",
        ),
        "split",
        _EXTERNAL,
        scope="outside",
    ),
    _e(
        "outside.yoroshikatta",
        ("よろしかったでしょうか", "よろしかったですか"),
        "ご注文は以上でよろしかったでしょうか。",
        "よろしかったでしょうか",
        "いわゆる接客表現の過去形確認。違和感を持つ人もいるが、"
        "規範として可否を定めた記述はない。",
        _outside(
            "いわゆる接客表現（よろしかったでしょうか）",
            "「敬語の指針」に該当記述はない。",
        ),
        "shifting",
        _COUNTER,
        scope="outside",
    ),
    _e(
        "outside.kara_oazukari",
        ("円からお預かり",),
        "一万円からお預かりします。",
        "円からお預かりします",
        "いわゆる接客表現。冗長と感じる人もいるが、規範として可否を定めた"
        "記述はない。",
        _outside(
            "いわゆる接客表現（〜からお預かりします）",
            "「敬語の指針」に該当記述はない。",
        ),
        "shifting",
        _COUNTER,
        scope="outside",
    ),
    _e(
        "outside.onamae_choudai",
        ("お名前を頂戴", "お名前頂戴"),
        "恐れ入りますが、お名前を頂戴できますでしょうか。",
        "お名前を頂戴",
        "名前は授受の対象ではないとして「お名前を伺えますか」を勧める案内があるが、"
        "接客場面では広く用いられている。",
        _outside(
            "いわゆる接客表現（お名前を頂戴できますか）",
            "「敬語の指針」に該当記述はない。",
        ),
        "split",
        _COUNTER,
        scope="outside",
    ),
    _e(
        "outside.daijoubu",
        ("大丈夫です",),
        "お気遣いなく、こちらは大丈夫です。",
        "大丈夫です",
        "断りにも承諾にも読める用法が広まっている。世代によって受け止め方が異なる。",
        _outside(
            "「大丈夫です」の多義化",
            "「敬語の指針」に該当記述はない。",
        ),
        "shifting",
        _INTERNAL,
        scope="outside",
    ),
    _e(
        "outside.itadakemasu_deshouka",
        ("いただけますでしょうか", "頂けますでしょうか"),
        "お手数ですが、ご確認いただけますでしょうか。",
        "いただけますでしょうか",
        "「ます」と「でしょう」を重ねた依頼形。冗長と感じる人もいるが、"
        "依頼の緩和として広く用いられている。",
        _outside(
            "依頼形の丁寧度（いただけますでしょうか）",
            "「敬語の指針」はこの形の可否を述べていない。",
        ),
        "split",
        _EXTERNAL,
        scope="outside",
    ),
    _e(
        "outside.gosashuu",
        ("ご査収のほど", "ご査収くださ", "御査収"),
        "添付のとおりお送りいたしますので、ご査収のほどお願い申し上げます。",
        "ご査収のほど",
        "書面的で硬い定型句。読み手によっては過剰に感じられる。",
        _outside(
            "メール定型句（ご査収のほど）",
            "「敬語の指針」はメールの定型句を扱っていない。",
        ),
        "split",
        _EXTERNAL,
        scope="outside",
    ),
    _e(
        "outside.osewasama",
        ("お世話様", "お世話さま"),
        "先日はお世話様でした。",
        "お世話様でした",
        "地域差・世代差が大きいあいさつ。目上に用いるのを避ける案内もある。",
        _outside(
            "あいさつの世代差・地域差（お世話様）",
            "「敬語の指針」に該当記述はない。",
        ),
        "split",
        _INTERNAL,
        scope="outside",
    ),
    _e(
        "outside.goissho",
        ("ご一緒します", "ご一緒いたします"),
        "では、私もご一緒します。",
        "ご一緒します",
        "上位者に対しては「お供します」を勧める案内があるが、"
        "「ご一緒します」を自然と受け取る人も多い。",
        _outside(
            "「ご一緒します」と「お供します」の使い分け",
            "「敬語の指針」はこの使い分けを扱っていない。",
        ),
        "split",
        _INTERNAL,
        scope="outside",
    ),
    _e(
        "outside.omotome_yasui",
        ("お求めやすい", "お求めやすく"),
        "今なら、お求めやすい価格でご提供しております。",
        "お求めやすい",
        "指針【9】に沿えば「お求めになりやすい」となるが、"
        "短い形が販売の場面で広く定着しており、受け止め方が分かれる。",
        _outside(
            "「お求めやすい」の定着度",
            "指針【9】は「動詞＋形容詞」の尊敬語化を扱うが、この縮約形の"
            "定着度については述べていない。Deference は規範違反として提示しない。",
        ),
        "shifting",
        _COUNTER,
        scope="outside",
    ),
    _e(
        "outside.sasete_itadaite",
        ("させていただいて", "させて頂いて"),
        "本サービスは会員限定で提供させていただいております。",
        "させていただいております",
        "案内文の定型として広まった用法。許可と恩恵の見立てが薄いと感じる人も"
        "いるが、受け止め方は分かれる。",
        _outside(
            "案内文の「させていただいております」",
            "指針【18】は許可と恩恵の見立てを論じるが、案内文の定型としての"
            "定着度には触れていない。",
        ),
        "shifting",
        _PUBLIC,
        related=_SASETE,
        suppressed=(_SASETE,),
        scope="outside",
    ),
    _e(
        "outside.orareru",
        ("おられます", "おられる", "おられました"),
        "佐藤様は本日おられますでしょうか。",
        "おられます",
        "「おる」に「れる」を付けた形。西日本を中心に尊敬語として広く用いられ、"
        "許容度に地域差・世代差がある。",
        _outside(
            "「おられる」の地域差",
            "「敬語の指針」は「おる」を謙譲語Ⅱとして挙げるが、"
            "「おられる」の地域的な尊敬語用法については述べていない。",
        ),
        "split",
        _EXTERNAL,
        related=_UCHI,
        scope="outside",
    ),
    _e(
        "outside.kudasaimase",
        ("くださいませ", "下さいませ"),
        "どうぞご覧くださいませ。",
        "くださいませ",
        "接客・案内で用いられる丁寧形。硬い、あるいは過剰と感じる人もいる。",
        _outside(
            "「くださいませ」の丁寧度",
            "「敬語の指針」はこの形の可否を述べていない。",
        ),
        "split",
        _COUNTER,
        scope="outside",
    ),
)


_ALL_ENTRIES: Tuple[_Entry, ...] = (
    _ENTRIES_ESTABLISHED + _ENTRIES_SPLIT + _ENTRIES_OUTSIDE
)


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------


class VariationSet:
    """揺れの登録簿。誤り検出の最終フィルタと、偽陽性評価用データを提供する。

    実証する主張:
        - 「過剰指摘の少なさ」: :meth:`is_variation` / :meth:`suppresses` が
          揺れに対する指摘を抑制する。detect.py / model.py の最後段に置くことで、
          規範上の分岐と揺れを混ぜない。
        - 「根拠提示」: 収録した各ケースは指針の該当箇所、または指針の射程外で
          あることを明示した出典を持つ。
        - 「速度」: 照合は正規表現も外部辞書も使わず、先頭文字で候補を絞る
          最左最長一致の1パス走査で行う。
    """

    def __init__(self) -> None:
        """登録簿を構築し、整合性を自己検査する。

        実証する主張: 「過剰指摘の少なさ」。構築時に
        :meth:`self_check` を走らせ、(1) 例文と focus スパンの一致、
        (2) 表層の重複がないこと、(3) :data:`NOT_VARIATION`（定着リストに
        載っていない二重敬語）に反応しないこと、(4) norms の
        ``SPLIT_ACCEPTABILITY`` を取りこぼしていないことを確かめる。
        """
        self._entries: Tuple[_Entry, ...] = _ALL_ENTRIES
        self._entry_of: Dict[str, _Entry] = {}
        for e in self._entries:
            for s in e.surfaces:
                if s in self._entry_of:
                    raise ValueError(
                        f"表層が重複しています: {s!r}"
                        f"（{self._entry_of[s].key} と {e.key}）"
                    )
                self._entry_of[s] = e
        # 先頭文字ごとに、長い順の候補列を用意する（最左最長一致のため）。
        self._by_first: Dict[str, Tuple[str, ...]] = {}
        buckets: Dict[str, List[str]] = {}
        for s in self._entry_of:
            buckets.setdefault(s[0], []).append(s)
        for head, group in buckets.items():
            group.sort(key=len, reverse=True)
            self._by_first[head] = tuple(group)
        self._cases: Tuple[VariationCase, ...] = tuple(
            self._build_case(e) for e in self._entries
        )
        self.self_check()

    # -- 構築 ---------------------------------------------------------------

    def _build_case(self, entry: _Entry) -> VariationCase:
        """内部項目を公開型 :class:`VariationCase` に展開する。"""
        pos = entry.example.find(entry.focus)
        if pos < 0:
            raise ValueError(
                f"例文に focus が含まれていません: {entry.key}（{entry.focus!r}）"
            )
        focus = Span(pos, pos + len(entry.focus), entry.focus)
        return VariationCase(
            text=entry.example,
            context=entry.context,
            focus=focus,
            reason=entry.reason,
            citation=entry.citation,
            acceptability=entry.acceptability,
            related_error_type=entry.related,
            meta={
                "key": entry.key,
                "scope": entry.scope,
                "surfaces": list(entry.surfaces),
                "suppressed": [t.value for t in entry.suppressed],
                "audiences": (
                    [a.value for a in entry.audiences] if entry.audiences else None
                ),
                "note": entry.note,
            },
        )

    # -- 公開 API -----------------------------------------------------------

    def cases(self) -> List[VariationCase]:
        """揺れの事例一覧。評価用データセットにもなる。

        各ケースは例文・想定場面（:class:`MailContext`）・注目スパン・根拠を
        すべて備えるため、そのまま偽陽性評価に流せる。

        実証する主張: 「過剰指摘の少なさ」。ここに入っている表現へ指摘が出れば
        偽陽性であると定義でき、評価が自動化できる。
        """
        return list(self._cases)

    def match(
        self,
        text: str,
        span: Optional[Span] = None,
        context: Optional[MailContext] = None,
    ) -> List[Tuple[Span, VariationCase]]:
        """本文中の揺れ表現を位置つきで返す。

        返す :class:`Span` は ``text`` 上の位置、:class:`VariationCase` は
        登録簿側の代表例（例文・場面・根拠を持つ）である。``span`` を渡すと、
        その範囲と重なる結果だけに絞る。``context`` を渡すと、宛先を限定した
        項目（社内宛でのみ揺れとするものなど）の判定に用いる。

        実証する主張:
            - 「速度」: 正規表現を使わず、先頭文字で候補を絞る最左最長一致の
              1パス走査。本文長に対して線形で、辞書サイズにはほとんど依存しない。
            - 「過剰指摘の少なさ」: 抑制対象を位置つきで返すため、
              文単位ではなくスパン単位で指摘を落とせる。
        """
        out: List[Tuple[Span, VariationCase]] = []
        n = len(text)
        i = 0
        while i < n:
            candidates = self._by_first.get(text[i])
            if candidates:
                hit = None
                for surface in candidates:  # 長い順＝最長一致
                    if text.startswith(surface, i):
                        hit = surface
                        break
                if hit is not None:
                    entry = self._entry_of[hit]
                    if self._audience_ok(entry, context):
                        found = Span(i, i + len(hit), hit)
                        if span is None or found.overlaps(span):
                            out.append((found, self._case_of(entry)))
                    i += len(hit)
                    continue
            i += 1
        return out

    def is_variation(
        self,
        text: str,
        span: Span,
        context: Optional[MailContext] = None,
    ) -> Optional[VariationCase]:
        """このスパンの指摘を抑制すべきか。抑制すべきなら該当ケースを返す。

        detect.py / model.py の最終フィルタで使う。重なる候補が複数あるときは
        指摘スパンとの重なりが最も大きいものを返す。

        実証する主張: 「過剰指摘の少なさ」。揺れに当たる位置の指摘を
        既定で出さないことを、この1関数に集約して検証可能にする。
        """
        best: Optional[Tuple[float, VariationCase]] = None
        for found, case in self.match(text, span, context):
            score = found.iou(span)
            if best is None or score > best[0]:
                best = (score, case)
        return best[1] if best else None

    def suppresses(
        self,
        text: str,
        span: Span,
        error_type: ErrorType,
        context: Optional[MailContext] = None,
    ) -> Optional[VariationCase]:
        """この誤り種別の指摘を、この位置で抑制すべきか。

        項目ごとに「どの誤り種別を抑制するか」を持たせてあり、指定がない項目は
        すべての種別を抑制する。``ErrorType.NONE`` は種別の指定なしとみなす。

        実証する主張: 「過剰指摘の少なさ」。抑制を種別ごとに絞ることで、
        たとえば「ご利用いただき」で向きの指摘だけを落とし、
        同じ位置の別種の指摘までは落とさない、という制御ができる。
        """
        best: Optional[Tuple[float, VariationCase]] = None
        for found, case in self.match(text, span, context):
            entry = self._entry_of[found.text]
            if entry.suppressed and error_type is not ErrorType.NONE:
                if error_type not in entry.suppressed:
                    continue
            score = found.iou(span)
            if best is None or score > best[0]:
                best = (score, case)
        return best[1] if best else None

    # -- 補助 ---------------------------------------------------------------

    def surfaces(self) -> List[str]:
        """登録されている照合用表層の一覧。

        実証する主張: 「過剰指摘の少なさ」。抑制の範囲が一覧で点検できる。
        """
        return sorted(self._entry_of)

    def by_acceptability(self, acceptability: str) -> List[VariationCase]:
        """許容度の区分（established / split / shifting）で絞る。

        実証する主張: 「根拠提示」。指針が明記している定着例と、
        個人差・変化進行中の事例を分けて提示できる。
        """
        return [c for c in self._cases if c.acceptability == acceptability]

    def stats(self) -> Dict[str, int]:
        """収録件数の内訳。データセットカードと論文の表に使う。

        実証する主張: 「過剰指摘の少なさ」。揺れをどれだけ明示的に持っているかを
        数で示す。
        """
        counts: Dict[str, int] = {"total": len(self._cases)}
        for c in self._cases:
            counts[f"acceptability.{c.acceptability}"] = (
                counts.get(f"acceptability.{c.acceptability}", 0) + 1
            )
            scope = str(c.meta.get("scope", "shishin"))
            counts[f"scope.{scope}"] = counts.get(f"scope.{scope}", 0) + 1
            counts["with_citation"] = counts.get("with_citation", 0) + (
                1 if c.citation is not None else 0
            )
        return counts

    def norms_split_coverage(self) -> List[str]:
        """:data:`norms.SPLIT_ACCEPTABILITY` のうち、照合できない表層を返す。

        norms.py が「揺れ」と宣言した表現を本モジュールが取りこぼしていないか
        点検する。空リストなら全件をカバーしている。

        実証する主張: 「根拠提示」。揺れの定義元が norms.py であることを
        コードで保証し、二重管理による食い違いを防ぐ。
        """
        missing: List[str] = []
        for case in norms.SPLIT_ACCEPTABILITY:
            probe = case.surface
            if not self.match(probe):
                missing.append(probe)
        return missing

    def self_check(self) -> None:
        """登録簿の整合性を検査する。構築時に自動で呼ばれる。

        実証する主張: 「過剰指摘の少なさ」。抑制しすぎ（定着していない二重敬語を
        揺れ扱いする）と抑制もれ（norms が揺れとする表現を拾えない）の両方を
        起動時に検出する。
        """
        for case in self._cases:
            actual = case.text[case.focus.start : case.focus.end]
            if actual != case.focus.text:
                raise AssertionError(
                    f"focus スパン不整合: {case.meta.get('key')} "
                    f"期待 {case.focus.text!r} / 実際 {actual!r}"
                )
        for surface in NOT_VARIATION:
            hits = self.match(surface)
            if hits:
                raise AssertionError(
                    f"定着リストにない形を揺れとして拾っています: {surface!r} "
                    f"（{[h[0].text for h in hits]}）"
                )
        missing = self.norms_split_coverage()
        if missing:
            raise AssertionError(f"norms の揺れを取りこぼしています: {missing}")

    def _case_of(self, entry: _Entry) -> VariationCase:
        return self._cases[self._entries.index(entry)]

    @staticmethod
    def _audience_ok(entry: _Entry, context: Optional[MailContext]) -> bool:
        """宛先を限定した項目かどうかを判定する。

        宛先限定の項目は、文脈が与えられ、かつ宛先が一致するときだけ抑制する。
        文脈が無いときに抑制すると、社外宛で身内を立てた表現まで見逃すため、
        「向きの誤り検出」という主張を壊さないよう保守的に扱う。
        """
        if entry.audiences is None:
            return True
        if context is None:
            return False
        return context.audience in entry.audiences

    def __len__(self) -> int:
        """収録している揺れの件数。

        実証する主張: 「過剰指摘の少なさ」。揺れの被覆量そのものが、過剰指摘を抑える力の目安になる。
        """
        return len(self._cases)

    def __repr__(self) -> str:  # pragma: no cover - 表示のみ
        """表示用。

        実証する主張: 「過剰指摘の少なさ」。収録件数を一目で確かめられるようにする。
        """
        return f"<VariationSet cases={len(self._cases)} surfaces={len(self._entry_of)}>"


_DEFAULT: Optional[VariationSet] = None


def default_variation_set() -> VariationSet:
    """プロセス内で共有する :class:`VariationSet` を返す。

    実証する主張: 「速度」。登録簿の構築は1回だけで済ませ、
    1通あたりの照合コストだけを支払う。
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = VariationSet()
    return _DEFAULT
