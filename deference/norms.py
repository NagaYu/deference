"""規範データベース：文化審議会答申「敬語の指針」（平成19年2月2日）に基づく。

このモジュールが Deference の唯一の「正しさの源」である。Generator も Corrector も
RuleCitation も、すべてここに書かれた分類・語形・条件だけを参照する。

引用の扱い（データセットカードにも同内容を明記する）:
    - 出典は文化審議会答申「敬語の指針」（文化庁ウェブサイト掲載）。
    - 文化庁ウェブサイトのコンテンツは「文部科学省ウェブサイト利用規約」に従う。
      同規約は「公共データ利用規約（第1.0版）に準拠」と定め、「CC BY と互換性が
      あり、本利用ルールが適用されるコンテンツは CC BY に従うことでも利用できます」
      としている。出典の記載を条件に「複製、公衆送信、翻訳・変形等の翻案等、
      自由に利用できます。商用利用も可能です」。編集・加工した場合は
      「加工して作成」した旨の明記が求められる。
    - 本リポジトリは規約に従い出典を明記した上で、判定の根拠として必要な最小限の
      短い引用（定義文・語例・要点）のみを保持する。**答申本文の全文またはそれに
      準ずる分量の再配布は行わない。** PDF 自体も同梱しない。
    - 引用箇所にはすべて章・節・問い番号・ページを付す。

実証する主張との対応:
    - 「根拠提示」: :data:`CITATIONS` が全指摘の裏付けを持つ。Deference が出す指摘は
      必ずこの表のどれかに紐付き、利用者は原典のページまで辿れる。
    - 「過剰指摘の少なさ」: :data:`ESTABLISHED_DOUBLE_KEIGO`、
      :data:`ACCEPTABLE_KEIGO_LINKS`、:data:`SPLIT_ACCEPTABILITY` は、指針自身が
      「習慣として定着している」「許容される」「個人差が大きい」と述べている表現の
      一覧である。これらを誤りにしないことが実用性の要である。
    - 「向きの誤り検出」: :data:`SPECIAL_FORMS` は各語形がどの分類かを持ち、
      分類は〈誰を立てるか〉を規定する。表層文字列ではなく分類で判定する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .types import Citation, ErrorType, KeigoClass

__all__ = [
    "SOURCE",
    "SOURCE_URL",
    "LICENSE_NOTE",
    "CITATIONS",
    "CITATION_FOR_ERROR",
    "cite",
    "cite_for",
    "KEIGO_CLASS_DEFINITION",
    "Verb",
    "VERBS",
    "verb",
    "SpecialForm",
    "SPECIAL_FORMS",
    "special_forms_for",
    "special_form_index",
    "ESTABLISHED_DOUBLE_KEIGO",
    "ACCEPTABLE_KEIGO_LINKS",
    "INAPPROPRIATE_KEIGO_LINKS",
    "SPLIT_ACCEPTABILITY",
    "NO_OGO_VERBS",
    "masu_stem",
    "a_stem",
    "e_stem",
    "te_form",
    "polite",
    "potential",
    "negative_polite",
    "HONORIFIC_PREFIX",
]


# ---------------------------------------------------------------------------
# 出典
# ---------------------------------------------------------------------------

SOURCE = "文化審議会答申「敬語の指針」（平成19年2月2日）"
SOURCE_URL = (
    "https://www.bunka.go.jp/seisaku/bunkashingikai/kokugo/hokoku/pdf/keigo_tosin.pdf"
)
SURVEY_SOURCE = "文化庁「国語に関する世論調査」"

LICENSE_NOTE = """\
出典: 文化審議会答申「敬語の指針」（文化庁）を加工して作成
{url}

文化庁ウェブサイトのコンテンツは「文部科学省ウェブサイト利用規約」
（https://www.mext.go.jp/b_menu/1351168.htm）に従う。同規約は「公共データ利用規約
（第1.0版）」に準拠し、CC BY と互換性があるとされ、出典を記載すれば複製・公衆送信・
翻訳・変形等の翻案等を自由に行える（商用利用も可）。編集・加工した場合は、その旨を
明記することが求められている。

本リポジトリは同規約に従い出典を明記した上で、判定の根拠に必要な短い引用
（定義文・語例・要点）のみを保持する。答申本文の全文またはそれに準ずる分量の
再配布は行わず、PDF も同梱しない。生成したコーパスは、答申の分類・語形・条件を
規則として実装したうえで機械的に構成したものであり、答申本文の文章ではない。\
""".format(url=SOURCE_URL)


def _c(section: str, page: str, quote: str, note: str = "", source: str = SOURCE) -> Citation:
    return Citation(
        source=source,
        section=section,
        page=page,
        quote=quote,
        url=SOURCE_URL,
        note=note,
    )


# ---------------------------------------------------------------------------
# 引用表：指摘の根拠はすべてここから引く
# ---------------------------------------------------------------------------

CITATIONS: Dict[str, Citation] = {
    # --- 5分類の定義 ---------------------------------------------------------
    "class.sonkeigo": _c(
        "第2章 第1-1 尊敬語（「いらっしゃる・おっしゃる」型）",
        "p.14",
        "相手側又は第三者の行為・ものごと・状態などについて，その人物を立てて述べるもの。",
    ),
    "class.kenjougo1": _c(
        "第2章 第1-2 謙譲語Ⅰ（「伺う・申し上げる」型）",
        "p.15",
        "自分側から相手側又は第三者に向かう行為・ものごとなどについて，"
        "その向かう先の人物を立てて述べるもの。",
    ),
    "class.kenjougo2": _c(
        "第2章 第1-3 謙譲語Ⅱ（丁重語）（「参る・申す」型）",
        "p.18",
        "自分側の行為・ものごとなどを，話や文章の相手に対して丁重に述べるもの。",
        note="謙譲語Ⅱは〈話の中の第三者〉を立てる働きを持たない（第3章第2-3【13】）。",
    ),
    "class.teineigo": _c(
        "第2章 第1-4 丁寧語（「です・ます」型）",
        "p.20",
        "話や文章の相手に対して丁寧に述べるもの。",
    ),
    "class.bikago": _c(
        "第2章 第1-5 美化語（「お酒・お料理」型）",
        "p.21",
        "ものごとを，美化して述べるもの。",
    ),
    # --- 自分側は立てない（向きの原則） ---------------------------------------
    "principle.self_not_raised": _c(
        "第2章 第1-6 尊敬語・謙譲語Ⅰの働きに関する留意点【解説1】",
        "p.22",
        "「自分側は立てない」というのが，尊敬語や謙譲語Ⅰを使う場合の基本的な原則である。",
        note="「父は来週海外へいらっしゃいます」「明日父のところに伺います」は"
        "いずれも自分側を立ててしまうため不適切。「参ります」なら問題ない。",
    ),
    "principle.third_party": _c(
        "第2章 第1-6（3）イ",
        "p.22-23",
        "自分から見れば，立てるのがふさわしいように見えても，"
        "「相手から見れば，立てる対象とは認識されないだろう」と思われる第三者については，"
        "立てない配慮が必要である。",
    ),
    # --- 敬語の形 -----------------------------------------------------------
    "form.sonkeigo": _c(
        "第2章 第2-1 尊敬語の形",
        "p.24-25",
        "「行く→いらっしゃる」のように特定の語形（特定形）による場合と，"
        "「お(ご)……になる」のように広くいろいろな語に適用できる一般的な語形（一般形）"
        "を使う場合とがある。",
    ),
    "form.kenjougo1": _c(
        "第2章 第2-2 謙譲語Ⅰの形",
        "p.26-27",
        "「訪ねる→伺う」のように特定の語形（特定形）による場合と，"
        "「お(ご)……する」のように広くいろいろな語に適用できる一般的な語形（一般形）"
        "を使う場合とがある。",
    ),
    "form.kenjougo1_condition": _c(
        "第2章 第2-2【補足イ－1】",
        "p.27",
        "これらの語は＜向かう先＞を立てる謙譲語Ⅰなので，"
        "＜向かう先＞の人物がある動詞に限って，これらの形を作ることができる。",
        note="「食べる」「乗車する」は＜向かう先＞が想定できないため"
        "「お食べする」「ご乗車する」は作れない。",
    ),
    "form.ogo_choice": _c(
        "第2章 第2-1【補足ア－1】",
        "p.25",
        "動詞が和語の場合は「読む→お読みになる」…漢語サ変動詞の場合は"
        "「利用する→御利用になる」…となる。",
    ),
    "form.sonkeigo_potential": _c(
        "第2章 第2-1（1）②",
        "p.25",
        "動詞に可能の意味を添えて，かつ尊敬語にするには，まず尊敬語の形にした上で"
        "可能の形にする。",
        note="「お(ご)……できる」は謙譲語Ⅰの可能形であり、尊敬語の可能形として"
        "使うのは適切ではない。",
    ),
    # --- (i) 二重敬語 --------------------------------------------------------
    "double_keigo": _c(
        "第2章 第2-6（2）「二重敬語」とその適否",
        "p.30",
        "一つの語について，同じ種類の敬語を二重に使ったものを「二重敬語」という。"
        "…「二重敬語」は，一般に適切ではないとされている。"
        "ただし，語によっては，習慣として定着しているものもある。",
    ),
    "double_keigo.established": _c(
        "第2章 第2-6（2）【習慣として定着している二重敬語の例】",
        "p.30",
        "（尊敬語）お召し上がりになる，お見えになる／"
        "（謙譲語Ⅰ）お伺いする，お伺いいたす，お伺い申し上げる",
        note="これらは二重敬語だが指針が定着を認めているため、Deference は誤りにしない。",
    ),
    "keigo_link": _c(
        "第2章 第2-6（3）「敬語連結」とその適否",
        "p.30",
        "「敬語連結」は，多少の冗長感が生じる場合もあるが，個々の敬語の使い方が適切で"
        "あり，かつ敬語同士の結び付きに意味的な不合理がない限りは，基本的に許容される"
        "ものである。",
        note="二つ以上の語をそれぞれ敬語にして「て」でつないだものは二重敬語ではない。",
    ),
    "keigo_link.bad": _c(
        "第2章 第2-6（3）【不適切な敬語連結の例】",
        "p.30-31",
        "伺ってくださる・伺っていただく",
        note="「先生が私の家を訪ねる」ことを謙譲語Ⅰ「伺う」で述べているため，"
        "「私」を立てることになる点が不適切。",
    ),
    # --- (ii) 尊敬語と謙譲語Ⅰの取り違え（向きの誤り） --------------------------
    "direction.q10": _c(
        "第3章 第2-2【10】",
        "p.37",
        "「担当者に伺ってください」の「伺う」は謙譲語Ⅰである。"
        "したがって，客の動作に用いる敬語ではない。",
        note="客を立てるには尊敬語が要る。「担当者にお聞きください」が適切。",
    ),
    "direction.q11": _c(
        "第3章 第2-2【11】",
        "p.37",
        "「お持ちする」は，謙譲語Ⅰである。したがって，自分が持っていくかどうかを"
        "上司である課長に尋ねたことになってしまう。",
        note="相手の動作を尋ねるなら「お持ちになりますか」。",
    ),
    "direction.q12": _c(
        "第3章 第2-2【12】",
        "p.38",
        "「御在宅する」に問題がある。「ご……する」は謙譲語Ⅰを作る形式だからである。",
        note="相手を立てるなら「御在宅なさる」あるいは「御在宅の」。",
    ),
    "direction.q13": _c(
        "第3章 第2-3【13】",
        "p.38",
        "「参る」は謙譲語Ⅱである。つまり相手に対して改まって伝えるための敬語であって，"
        "話の中に出てくる第三者を立てるための敬語ではない。",
    ),
    # --- (iii) 身内に尊敬語 / ウチ・ソト ---------------------------------------
    "uchi_soto.q23": _c(
        "第3章 第3-2【23】",
        "p.43-44",
        "「ウチ・ソト」の意識に基づけば，同僚の田中教諭は「ウチ」の人であり，"
        "保護者を相手とする場合には…「田中はおりません。」と伝えた方が良い。",
        note="ただし指針は同じ問いで、世論調査では「田中先生」を支持する人が多いことも"
        "併記している。学校場面は揺れとして扱う。",
    ),
    "uchi_soto.q24": _c(
        "第3章 第3-2【24】",
        "p.44",
        "「田中部長」をウチ扱いにする（自分側の人物として扱う）ときには，"
        "「田中」と呼ぶことに問題はない。…"
        "ただし，「田中部長」と呼ぶことは，ウチ扱いにした呼び方にはならないので，不適切である。",
    ),
    "uchi_soto.q25": _c(
        "第3章 第3-2【25】",
        "p.44-45",
        "社外の人が多くいる場合には，会社のウチ・会社のソトといった関係が生じるので，"
        "「ウチ」の社長は立てない方が良い。",
        note="宛先が社内か社外かで適切な形が入れ替わる。Deference が宛先を必須の"
        "入力にしているのはこのため。",
    ),
    "uchi_soto.q26": _c(
        "第3章 第3-2【26】",
        "p.45",
        "課長を立てずに，相手である部長に対して改まった表現を用いて，"
        "「課長は，このように申しておりました。」と言えば良いことになる。",
        note="指針は同じ問いで、課長を立ててよいとする考え方にも理があるとしている"
        "（＝この場面は揺れ）。",
    ),
    "self_sonkeigo.q16": _c(
        "第3章 第2-4【16】",
        "p.39",
        "「お」や「御」を自分のことに付けてはいけないのは，例えば，"
        "「私のお考え」「私の御旅行」など，自分側の動作やものごとを立ててしまう場合である。",
        note="逆に「お待ちしています」「御説明をしたい」は謙譲語Ⅰなので問題ない。",
    ),
    # --- (iv) さ入れ言葉 -----------------------------------------------------
    "sa_insertion": Citation(
        section="Causative of godan verbs / 五段動詞の使役形（休む→休ませる）",
        page="—",
        quote="",
        # 指針を根拠にしていないので、指針の URL は載せない。
        # 根拠の出所を偽らないための措置。
        url="",
        note="五段動詞の使役は「休む→休ませる」「読む→読ませる」であり、"
        "「休まさせる」「読まさせる」は活用形として規範から外れる（いわゆる"
        "「さ入れ言葉」）。「敬語の指針」はこの語形を扱っていないため、"
        "Deference は指針を根拠として引かず、活用規則を根拠として提示する。",
        source="活用規則（「敬語の指針」に該当記述なし）",
    ),
    "sa_insertion.survey": Citation(
        section="平成8年度「国語に関する世論調査」",
        page="—",
        quote="",
        url="https://www.bunka.go.jp/tokei_hakusho_shuppan/tokeichosa/kokugo_yoronchosa/h08/",
        note="「休まさせていただきます」を「気にならない」と答えた人が6割を超えるなど、"
        "話し言葉では広く用いられている。書き言葉のビジネス文書では規範形を案内する。",
        source=SURVEY_SOURCE,
    ),
    # --- (v) させていただく --------------------------------------------------
    "sasete_itadaku": _c(
        "第3章 第2-6【18】【解説1】",
        "p.40",
        "「（お・ご）……(さ)せていただく」といった敬語の形式は，基本的には，"
        "自分側が行うことを，ア）相手側又は第三者の許可を受けて行い，"
        "イ）そのことで恩恵を受けるという事実や気持ちのある場合に使われる。",
    ),
    "sasete_itadaku.tolerance": _c(
        "第3章 第2-6【18】【解説2】",
        "p.40-41",
        "その見立てをどの程度自然なものとして受け入れるかということが，"
        "その個人にとっての「…(さ)せていただく」に対する「許容度」を決めている"
        "のだと考えられる。",
        note="指針は一律に不適切とはしていない。Deference は許可・恩恵の条件が"
        "立たない用法のみを、しかも密度が高い場合にだけ情報提示する。",
    ),
    # --- (vi) 敬体と常体の混在 -----------------------------------------------
    "style_mixing": _c(
        "第1章 第2-5",
        "p.11",
        "「丁寧な言葉と普通の言葉」や「敬体と常体」という2分類…",
        note="指針は敬体・常体の混在自体を可否として論じてはいない。"
        "Deference は文体の一貫性という文章作法上の観点として提示し、"
        "規範違反とは呼ばない。",
    ),
    # --- (vii) 敬意の程度の不整合 --------------------------------------------
    "deference_inversion": _c(
        "第2章 第1-6（3）イ／第3章 第3-2【25】",
        "p.22-23, p.44-45",
        "社外の人が多くいる場合には，会社のウチ・会社のソトといった関係が生じるので，"
        "「ウチ」の社長は立てない方が良い。",
        note="身内を相手側より高く述べると敬意の向きが逆転する。"
        "宛先の情報がなければ判定できない類型。",
    ),
    # --- 派生型 -------------------------------------------------------------
    "go_sareru.q7": _c(
        "第3章 第2-1【7】",
        "p.36",
        "規範的には，「適切な敬語」だとは位置付けられてこなかった形である。"
        "したがって，現時点では，「利用される・利用なさる・御利用になる・御利用なさる」"
        "などが適切な形だと言える。",
    ),
    "ogo_dekiru.q8": _c(
        "第3章 第2-1【8】",
        "p.36",
        "「お(ご)……できる」というのは，謙譲語Ⅰの形である「お(ご)……する」の可能形である。"
        "…ここは，相手（＝乗客）の行為なので尊敬語を使うべきところである。",
    ),
    "adj_keigo.q9": _c(
        "第3章 第2-1【9】",
        "p.37",
        "「動詞＋形容詞」の形を取るものを尊敬語にする場合には，"
        "動詞の部分だけを尊敬語にすれば良い。",
    ),
    # --- 揺れとして扱う根拠 ---------------------------------------------------
    "variation.itadaku_kudasaru": _c(
        "第3章 第2-5【17】",
        "p.40",
        "基本的には，どちらもほぼ同じように使える敬語だと言ってよい。",
        note="「ご利用いただく」（謙譲語Ⅰ）と「ご利用くださる」（尊敬語）は"
        "どちらも適切。受け止め方に個人差がある。",
    ),
    "variation.jisan": _c(
        "第3章 第2-3【14】",
        "p.38-39",
        "これらの表現の中に含まれる「参る」や「申す」は，謙譲語Ⅱとしての働きは"
        "持っていないと言ってよい。したがって，これらの表現を「相手側」の行為に"
        "用いるのは問題ない。",
        note="「御持参ください」「お申し出ください」は誤りではない。",
    ),
    "variation.adj_desu": _c(
        "第2章 第2-4 丁寧語",
        "p.28",
        "（「高いです。」のように形容詞に「です」を付けることについては抵抗を感じる人も"
        "あろうが，既にかなりの人が許容するようになってきている。）",
    ),
    "variation.ageru": _c(
        "第1章 第2-2 世代や性による敬語意識の多様性",
        "p.8",
        "「植木に水をあげる」という場合の「あげる」は，旧来の規範からすれば誤用と"
        "されるものであるが，この語の謙譲語から美化語に向かう意味的な変化は既に進行し，"
        "定着しつつあると言ってよい。",
    ),
    "variation.generational": _c(
        "第1章 第2-2 世代や性による敬語意識の多様性",
        "p.8",
        "男女の違いや世代の違いなどによって画一的に考える態度は避けるべきである。",
        note="Deference が揺れを赤くしない方針の根拠。",
    ),
    "principle.moderate": _c(
        "第3章 第1-4 敬語は過剰でなく適度に使う",
        "p.35",
        "敬語は過剰でなく適度に使う",
    ),
    "principle.self_expression": _c(
        "第1章 第1-3 「自己表現」としての敬語使用",
        "p.7",
        "",
        note="敬語は「自己表現」として自ら選ぶもの。Deference が断定的に"
        "「間違い」と言わず情報提示に留める方針の根拠。",
    ),
}


def cite(key: str) -> Citation:
    """引用キーから :class:`Citation` を引く。未知キーは例外。

    実証する主張: 「根拠提示」。存在しない引用キーを例外にすることで、根拠のない指摘をコードとして書けなくする。
    """
    if key not in CITATIONS:
        raise KeyError(f"未知の引用キーです: {key!r}")
    return CITATIONS[key]


#: 誤り種別 → 既定の引用キー
CITATION_FOR_ERROR: Dict[ErrorType, str] = {
    ErrorType.DOUBLE_KEIGO: "double_keigo",
    ErrorType.DIRECTION_SWAP: "direction.q10",
    ErrorType.UCHI_SONKEIGO: "principle.self_not_raised",
    ErrorType.SA_INSERTION: "sa_insertion",
    ErrorType.SASETE_ITADAKU_OVERUSE: "sasete_itadaku",
    ErrorType.STYLE_MIXING: "style_mixing",
    ErrorType.DEFERENCE_INVERSION: "deference_inversion",
    ErrorType.GO_SARERU: "go_sareru.q7",
    ErrorType.OGO_DEKIRU: "ogo_dekiru.q8",
    ErrorType.BAD_KEIGO_LINK: "keigo_link.bad",
    ErrorType.SELF_SONKEIGO: "self_sonkeigo.q16",
    ErrorType.NONE: "principle.self_expression",
}


def cite_for(error_type: ErrorType) -> Citation:
    """誤り種別から既定の引用を引く。

    実証する主張: 「根拠提示」。すべての誤り種別が原典の該当箇所に対応づく
    ことを、この関数の全数テストで保証する。
    """
    return cite(CITATION_FOR_ERROR[error_type])


#: 5分類の定義（表示用）
KEIGO_CLASS_DEFINITION: Dict[KeigoClass, str] = {
    KeigoClass.SONKEIGO: CITATIONS["class.sonkeigo"].quote,
    KeigoClass.KENJOUGO_1: CITATIONS["class.kenjougo1"].quote,
    KeigoClass.KENJOUGO_2: CITATIONS["class.kenjougo2"].quote,
    KeigoClass.TEINEIGO: CITATIONS["class.teineigo"].quote,
    KeigoClass.BIKAGO: CITATIONS["class.bikago"].quote,
    KeigoClass.KENJOUGO_1_AND_2: "「お(ご)……いたす」は謙譲語Ⅰ兼謙譲語Ⅱの一般形。",
    KeigoClass.PLAIN: "敬語形ではない素の形。",
}


# ---------------------------------------------------------------------------
# 活用（規則ベースの語形生成の土台）
# ---------------------------------------------------------------------------

#: 五段活用の連用形（ます形の語幹）への写像
_GODAN_MASU = {
    "う": "い", "く": "き", "ぐ": "ぎ", "す": "し", "つ": "ち",
    "ぬ": "に", "ぶ": "び", "む": "み", "る": "り",
}
#: 五段活用の未然形（〜ない／〜せる が付く形）への写像
_GODAN_A = {
    "う": "わ", "く": "か", "ぐ": "が", "す": "さ", "つ": "た",
    "ぬ": "な", "ぶ": "ば", "む": "ま", "る": "ら",
}
#: 五段活用の仮定形／可能動詞の語幹
_GODAN_E = {
    "う": "え", "く": "け", "ぐ": "げ", "す": "せ", "つ": "て",
    "ぬ": "ね", "ぶ": "べ", "む": "め", "る": "れ",
}
#: 五段活用のテ形
_GODAN_TE = {
    "う": "って", "つ": "って", "る": "って",
    "む": "んで", "ぶ": "んで", "ぬ": "んで",
    "く": "いて", "ぐ": "いで", "す": "して",
}

#: ラ行五段だが連用形が「い」になる敬語動詞（いらっしゃる型）
_IRREGULAR_MASU_STEM = {
    "いらっしゃる": "いらっしゃい",
    "おっしゃる": "おっしゃい",
    "なさる": "なさい",
    "くださる": "ください",
    "下さる": "ください",
    "ござる": "ござい",
}

HONORIFIC_PREFIX = {"wago": "お", "kango": "ご"}


def masu_stem(form: str, kind: str) -> str:
    """連用形（「ます」が付く形）を返す。

    実証する主張: 「修正候補の妥当性」。修正候補はすべてこの規則的な活用から
    組み立てられ、自由生成を含まない。
    """
    if form in _IRREGULAR_MASU_STEM:
        return _IRREGULAR_MASU_STEM[form]
    if kind == "ichidan":
        return form[:-1]
    if kind == "sahen":
        return form[:-2] + "し" if form.endswith("する") else form
    if kind == "kahen":
        return "来"
    if kind == "copula":
        # 「ご存じだ」型。「だ」を落とした形が語幹に当たる。
        return form[:-1]
    if kind == "godan":
        return form[:-1] + _GODAN_MASU[form[-1]]
    raise ValueError(f"未知の活用種別です: {kind!r}")


def a_stem(form: str, kind: str) -> str:
    """未然形（「ない」「せる」が付く形）を返す。さ入れ言葉の生成に使う。

    実証する主張: 「根拠提示」。さ入れ言葉の判定根拠が五段動詞の活用規則そのものであることを、この関数が体現する。
    """
    if kind == "ichidan":
        return form[:-1]
    if kind == "sahen":
        return form[:-2] + "さ" if form.endswith("する") else form
    if kind == "kahen":
        return "来"
    if kind == "godan":
        return form[:-1] + _GODAN_A[form[-1]]
    raise ValueError(f"未知の活用種別です: {kind!r}")


def e_stem(form: str, kind: str) -> str:
    """仮定形の語幹（五段の可能動詞を作るのに使う）。

    実証する主張: 「修正候補の妥当性」。可能形を規則から作るための語幹。
    """
    if kind == "godan":
        return form[:-1] + _GODAN_E[form[-1]]
    raise ValueError("e_stem は五段活用にのみ適用できます")


def te_form(form: str, kind: str) -> str:
    """テ形を返す。敬語連結の生成・判定に使う。

    実証する主張: 「過剰指摘の少なさ」。敬語連結（て でつないだ形）は二重敬語ではないため、テ形を正しく作れることが誤検出の抑制に効く。
    """
    if kind == "ichidan":
        return form[:-1] + "て"
    if kind == "sahen":
        return form[:-2] + "して" if form.endswith("する") else form + "て"
    if kind == "kahen":
        return "来て"
    if kind == "godan":
        if form == "行く":
            return "行って"
        return form[:-1] + _GODAN_TE[form[-1]]
    raise ValueError(f"未知の活用種別です: {kind!r}")


def polite(form: str, kind: str) -> str:
    """「ます」形を返す（丁寧語 = 敬体）。「……だ」型は「です」になる。

    実証する主張: 「修正候補の妥当性」。候補は必ず規則的な活用から組み立てる。
    """
    if kind == "copula":
        return form[:-1] + "です"
    return masu_stem(form, kind) + "ます"


def negative_polite(form: str, kind: str) -> str:
    """「ません」形を返す。

    実証する主張: 「修正候補の妥当性」。否定形の候補も規則から作る。
    """
    return masu_stem(form, kind) + "ません"


def potential(form: str, kind: str) -> str:
    """可能形を返す。

    指針 第2章第2-1（1）②「まず尊敬語の形にした上で可能の形にする」に従い、
    敬語形を作ってからこの関数を適用するのが正しい順序である。

    実証する主張: 「修正候補の妥当性」。指針 第2章第2-1（1）② の順序（敬語形にしてから可能形）を関数の適用順で守る。
    """
    if kind == "godan":
        return e_stem(form, kind) + "る"
    if kind == "ichidan":
        return form[:-1] + "られる"
    if kind == "sahen":
        return form[:-2] + "できる" if form.endswith("する") else form
    if kind == "kahen":
        return "来られる"
    raise ValueError(f"未知の活用種別です: {kind!r}")


# ---------------------------------------------------------------------------
# 動詞辞書
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verb:
    """一般動詞1語ぶんの規範情報。

    実証する主張: 「修正候補の妥当性」。``has_target`` と ``ogo`` は
    指針 第2章第2-2【補足イ】に基づく制約であり、これを守ることで
    「お食べする」「ご乗車する」のような作れない形を生成しない。
    """

    plain: str  # 素の終止形（読む）
    kind: str  # godan / ichidan / sahen / kahen
    gloss: str  # 意味の短い説明
    ogo: str = "お"  # 「お」/「ご」/ ""（なじまない）
    has_target: bool = True  # ＜向かう先＞の人物があるか
    can_ogo_ni_naru: bool = True  # 「お(ご)……になる」が作れるか
    functions: Tuple[str, ...] = ()

    # --- 一般形（指針 第2章第2） ----------------------------------------
    @property
    def stem(self) -> str:
        """連用形（「お読みになる」の「読み」）。

        実証する主張: 「修正候補の妥当性」。
        """
        return masu_stem(self.plain, self.kind)

    @property
    def ogo_stem(self) -> str:
        """「お(ご)……」に挟まる部分。

        指針 第2章第2-1【ア－1】に従い、和語は連用形（読む→読み）、
        漢語サ変動詞は語基（利用する→利用）を用いる。連用形をそのまま使うと
        「ご利用しになる」のような作れない形になるため、ここで分岐する。

        実証する主張: 「修正候補の妥当性」。ここを連用形にすると「ご利用しになる」という作れない形が出る。
        """
        if self.kind == "sahen":
            return self.plain[:-2]
        return masu_stem(self.plain, self.kind)

    def sonkeigo_general(self) -> List[str]:
        """尊敬語の一般形（お(ご)……になる／……(ら)れる／……なさる 等）。

        実証する主張: 「向きの誤り検出」。相手側の行為に当てる形の集合であり、自分側に当たっていれば向きの誤りになる。
        """
        out: List[str] = []
        if self.ogo and self.can_ogo_ni_naru:
            out.append(f"{self.ogo}{self.ogo_stem}になる")
        out.append(self._rareru())
        if self.kind == "sahen":
            base = self.plain[:-2]
            out.append(f"{base}なさる")
            if self.ogo == "ご":
                out.append(f"ご{base}なさる")
        if self.ogo and self.can_ogo_ni_naru:
            out.append(f"{self.ogo}{self.ogo_stem}だ")
        return out

    def _rareru(self) -> str:
        """「……(ら)れる」形。"""
        if self.kind == "godan":
            return a_stem(self.plain, self.kind) + "れる"
        if self.kind == "ichidan":
            return self.plain[:-1] + "られる"
        if self.kind == "sahen":
            return self.plain[:-2] + "される"
        return "来られる"

    def sonkeigo_kudasaru(self) -> Optional[str]:
        """「お(ご)……くださる」形。

        実証する主張: 「修正候補の妥当性」。指針【17】が「いただく」と並んで適切とする形。
        """
        if self.ogo and self.can_ogo_ni_naru:
            return f"{self.ogo}{self.ogo_stem}くださる"
        return None

    def kenjougo1_general(self) -> List[str]:
        """謙譲語Ⅰの一般形（お(ご)……する／申し上げる／いただく）。

        ＜向かう先＞が想定できない動詞では空を返す（指針【補足イ－1】）。

        実証する主張: 「修正候補の妥当性」。＜向かう先＞が無い動詞には作らない（指針【補足イ－1】）ので、「お食べする」のような形を提示しない。
        """
        if not (self.has_target and self.ogo):
            return []
        return [
            f"{self.ogo}{self.ogo_stem}する",
            f"{self.ogo}{self.ogo_stem}申し上げる",
            f"{self.ogo}{self.ogo_stem}いただく",
        ]

    def kenjougo2_general(self) -> List[str]:
        """謙譲語Ⅱの一般形（……いたす）。サ変動詞のみ。

        実証する主張: 「向きの誤り検出」。自分側の行為に使える形。身内敬語の修正候補はここから出る。
        """
        if self.kind != "sahen":
            return []
        base = self.plain[:-2]
        out = [f"{base}いたす"]
        if self.ogo and self.has_target:
            # 「お(ご)……いたす」は謙譲語Ⅰ兼Ⅱ。謙譲語Ⅰの側面を持つため
            # ＜向かう先＞のある動詞にしか作れない（指針【補足イ－1】）。
            out.append(f"{self.ogo}{base}いたす")
        return out


def _v(plain, kind, gloss, ogo="お", has_target=True, can_ogo=True, functions=()):
    return Verb(plain, kind, gloss, ogo, has_target, can_ogo, tuple(functions))


#: 一般動詞辞書。ビジネス文書で頻出する語を中心に構成する。
VERBS: Tuple[Verb, ...] = (
    # --- 和語（お＋連用形） --------------------------------------------------
    _v("読む", "godan", "読む", "お", True, True, ("report",)),
    _v("書く", "godan", "書く", "お", True, True, ("report",)),
    _v("送る", "godan", "送る", "お", True, True, ("report", "giving")),
    _v("届ける", "ichidan", "届ける", "お", True, True, ("giving",)),
    _v("待つ", "godan", "待つ", "お", True, True, ("scheduling",)),
    _v("使う", "godan", "使う", "お", True, True, ()),
    _v("呼ぶ", "godan", "呼ぶ", "お", True, True, ()),
    _v("渡す", "godan", "渡す", "お", True, True, ("giving",)),
    _v("預かる", "godan", "預かる", "お", True, True, ()),
    _v("持つ", "godan", "持つ", "お", True, True, ()),
    _v("知らせる", "ichidan", "知らせる", "お", True, True, ("report", "notice")),
    _v("願う", "godan", "願う", "お", True, True, ("request",)),
    _v("勧める", "ichidan", "勧める", "お", True, True, ("request",)),
    _v("誘う", "godan", "誘う", "お", True, True, ("request",)),
    _v("受け取る", "godan", "受け取る", "お", True, True, ("receiving",)),
    _v("決める", "ichidan", "決める", "お", False, True, ()),
    _v("選ぶ", "godan", "選ぶ", "お", False, True, ()),
    _v("調べる", "ichidan", "調べる", "お", False, True, ()),
    _v("帰る", "godan", "帰る", "お", False, True, ()),
    _v("休む", "godan", "休む", "お", False, True, ("notice",)),
    _v("急ぐ", "godan", "急ぐ", "お", False, True, ()),
    _v("考える", "ichidan", "考える", "お", False, True, ()),
    _v("使いこなす", "godan", "使いこなす", "", False, False, ()),
    # 「お(ご)……になる」が作れない例（指針【ア－3】）
    _v("死ぬ", "godan", "死ぬ", "", False, False, ()),
    _v("失敗する", "sahen", "失敗する", "", False, False, ()),
    _v("運転する", "sahen", "運転する", "", False, False, ()),
    # --- 漢語サ変（ご＋語幹） ------------------------------------------------
    _v("利用する", "sahen", "利用する", "ご", False, True, ()),
    _v("出席する", "sahen", "出席する", "ご", False, True, ("scheduling",)),
    _v("案内する", "sahen", "案内する", "ご", True, True, ("giving",)),
    _v("説明する", "sahen", "説明する", "ご", True, True, ("report",)),
    _v("連絡する", "sahen", "連絡する", "ご", True, True, ("report",)),
    _v("確認する", "sahen", "確認する", "ご", True, True, ("inquiry",)),
    _v("検討する", "sahen", "検討する", "ご", False, True, ("request",)),
    _v("報告する", "sahen", "報告する", "ご", True, True, ("report",)),
    _v("相談する", "sahen", "相談する", "ご", True, True, ("inquiry",)),
    _v("対応する", "sahen", "対応する", "ご", True, True, ("request",)),
    _v("参加する", "sahen", "参加する", "ご", False, True, ("scheduling",)),
    _v("送付する", "sahen", "送付する", "ご", True, True, ("giving",)),
    _v("提案する", "sahen", "提案する", "ご", True, True, ("report",)),
    _v("返信する", "sahen", "返信する", "ご", True, True, ("report",)),
    _v("訪問する", "sahen", "訪問する", "ご", True, True, ("scheduling",)),
    _v("確保する", "sahen", "確保する", "ご", False, True, ()),
    _v("承知する", "sahen", "承知する", "ご", False, True, ("thanks",)),
    _v("在宅する", "sahen", "在宅する", "ご", False, True, ()),  # 指針【12】
    _v("乗車する", "sahen", "乗車する", "ご", False, True, ()),  # 指針【8】
    _v("持参する", "sahen", "持参する", "ご", True, True, ("request",)),
    _v("記入する", "sahen", "記入する", "ご", False, True, ("request",)),
    _v("指導する", "sahen", "指導する", "ご", True, True, ()),
    _v("協力する", "sahen", "協力する", "ご", True, True, ("request",)),
)

_VERB_INDEX: Dict[str, Verb] = {v.plain: v for v in VERBS}


def verb(plain: str) -> Verb:
    """素の終止形から :class:`Verb` を引く。

    実証する主張: 「修正候補の妥当性」。辞書に無い語で当て推量の形を作らないよう、例外で知らせる。
    """
    if plain not in _VERB_INDEX:
        raise KeyError(f"辞書にない動詞です: {plain!r}")
    return _VERB_INDEX[plain]


#: 「お(ご)……になる」が作れない動詞（指針 第2章第2-1【ア－3】）
NO_OGO_VERBS: FrozenSet[str] = frozenset(
    v.plain for v in VERBS if not v.ogo or not v.can_ogo_ni_naru
)


# ---------------------------------------------------------------------------
# 特定形（指針 第2章第2 の【特定形の主な例】）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecialForm:
    """特定形1件。指針が明示的に挙げている語形のみを収録する。

    実証する主張: 「根拠提示」。指針が明示的に挙げている語形だけを収録し、出典を語形ごとに保持する。
    """

    base: str  # 素の動詞（言う）
    form: str  # 特定形（おっしゃる）
    keigo_class: KeigoClass
    kind: str  # 活用種別
    citation_key: str
    note: str = ""

    @property
    def polite_form(self) -> str:
        """この特定形の「ます」形。

        実証する主張: 「修正候補の妥当性」。特定形も規則的な活用から敬体を作る。
        """
        return polite(self.form, self.kind)


def _sf(base, form, cls, kind, ck, note=""):
    return SpecialForm(base, form, cls, kind, ck, note)


SPECIAL_FORMS: Tuple[SpecialForm, ...] = (
    # --- 尊敬語（指針 p.24） ------------------------------------------------
    _sf("行く", "いらっしゃる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("来る", "いらっしゃる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("いる", "いらっしゃる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("行く", "おいでになる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("来る", "おいでになる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("いる", "おいでになる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("来る", "見える", KeigoClass.SONKEIGO, "ichidan", "form.sonkeigo"),
    _sf("言う", "おっしゃる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("する", "なさる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("食べる", "召し上がる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("飲む", "召し上がる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("くれる", "くださる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("見る", "ご覧になる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("寝る", "お休みになる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("着る", "お召しになる", KeigoClass.SONKEIGO, "godan", "form.sonkeigo"),
    _sf("知る", "ご存じだ", KeigoClass.SONKEIGO, "copula", "form.sonkeigo",
        "「ご存じだ（ご存じです）」は「知っている」の尊敬語。"),
    # --- 謙譲語Ⅰ（指針 p.26） ----------------------------------------------
    _sf("訪ねる", "伺う", KeigoClass.KENJOUGO_1, "godan", "form.kenjougo1"),
    _sf("尋ねる", "伺う", KeigoClass.KENJOUGO_1, "godan", "form.kenjougo1"),
    _sf("聞く", "伺う", KeigoClass.KENJOUGO_1, "godan", "form.kenjougo1"),
    _sf("言う", "申し上げる", KeigoClass.KENJOUGO_1, "ichidan", "form.kenjougo1"),
    _sf("知る", "存じ上げる", KeigoClass.KENJOUGO_1, "ichidan", "form.kenjougo1"),
    _sf("上げる", "差し上げる", KeigoClass.KENJOUGO_1, "ichidan", "form.kenjougo1"),
    _sf("もらう", "いただく", KeigoClass.KENJOUGO_1, "godan", "form.kenjougo1"),
    _sf("会う", "お目にかかる", KeigoClass.KENJOUGO_1, "godan", "form.kenjougo1"),
    _sf("見せる", "お目にかける", KeigoClass.KENJOUGO_1, "ichidan", "form.kenjougo1"),
    _sf("見せる", "ご覧に入れる", KeigoClass.KENJOUGO_1, "ichidan", "form.kenjougo1"),
    _sf("見る", "拝見する", KeigoClass.KENJOUGO_1, "sahen", "form.kenjougo1"),
    _sf("借りる", "拝借する", KeigoClass.KENJOUGO_1, "sahen", "form.kenjougo1"),
    # --- 謙譲語Ⅱ（丁重語）（指針 p.28） -------------------------------------
    _sf("行く", "参る", KeigoClass.KENJOUGO_2, "godan", "class.kenjougo2"),
    _sf("来る", "参る", KeigoClass.KENJOUGO_2, "godan", "class.kenjougo2"),
    _sf("言う", "申す", KeigoClass.KENJOUGO_2, "godan", "class.kenjougo2"),
    _sf("する", "いたす", KeigoClass.KENJOUGO_2, "godan", "class.kenjougo2"),
    _sf("いる", "おる", KeigoClass.KENJOUGO_2, "godan", "class.kenjougo2"),
    _sf("知る", "存じる", KeigoClass.KENJOUGO_2, "ichidan", "class.kenjougo2"),
    _sf("思う", "存じる", KeigoClass.KENJOUGO_2, "ichidan", "class.kenjougo2"),
)


def special_forms_for(base: str, keigo_class: Optional[KeigoClass] = None) -> List[SpecialForm]:
    """素の動詞から特定形を引く。分類で絞り込める。

    実証する主張: 「向きの誤り検出」。素の動詞から分類ごとの語形を引けることが、向きの判定と修正の前提になる。
    """
    return [
        sf
        for sf in SPECIAL_FORMS
        if sf.base == base and (keigo_class is None or sf.keigo_class is keigo_class)
    ]


def special_form_index() -> Dict[str, List[SpecialForm]]:
    """特定形の表層 → :class:`SpecialForm` 群。検出側で使う。

    実証する主張: 「向きの誤り検出」。表層から分類を逆引きし、その述語が誰を立てているかを観測する。
    """
    idx: Dict[str, List[SpecialForm]] = {}
    for sf in SPECIAL_FORMS:
        idx.setdefault(sf.form, []).append(sf)
    return idx


#: 特定形を持つ素の動詞の集合
SPECIAL_BASES: FrozenSet[str] = frozenset(sf.base for sf in SPECIAL_FORMS)


# ---------------------------------------------------------------------------
# 揺れ・許容の一覧（過剰指摘を避けるための要）
# ---------------------------------------------------------------------------

#: 習慣として定着している二重敬語（指針 p.30）。誤りにしない。
ESTABLISHED_DOUBLE_KEIGO: Tuple[str, ...] = (
    "お召し上がりになる",
    "お見えになる",
    "お伺いする",
    "お伺いいたす",
    "お伺い申し上げる",
)

#: 許容される敬語連結（指針 p.30）。誤りにしない。
ACCEPTABLE_KEIGO_LINKS: Tuple[str, ...] = (
    "お読みになっていらっしゃる",
    "お読みになってくださる",
    "お読みになっていただく",
    "ご案内してさしあげる",
)

#: 不適切な敬語連結（指針 p.30-31）。
INAPPROPRIATE_KEIGO_LINKS: Tuple[str, ...] = (
    "伺ってくださる",
    "伺っていただく",
    "伺ってください",
)


@dataclass(frozen=True)
class SplitCase:
    """許容度が割れる表現1件。誤りではなく「揺れ」として扱う。

    実証する主張: 「過剰指摘の少なさ」。指針が定着・許容・個人差と述べた表現を、誤りと分けて保持する。
    """

    surface: str
    reason: str
    citation_key: str
    acceptability: str = "split"  # established / split / shifting


#: 指針自身が「定着」「許容」「個人差が大きい」と述べている表現。
#: Deference は既定でこれらに指摘を出さない。
SPLIT_ACCEPTABILITY: Tuple[SplitCase, ...] = (
    SplitCase("お伺いする", "指針が習慣として定着した二重敬語と明記している。",
              "double_keigo.established", "established"),
    SplitCase("お伺いいたします", "指針が習慣として定着した二重敬語と明記している。",
              "double_keigo.established", "established"),
    SplitCase("お伺い申し上げます", "指針が習慣として定着した二重敬語と明記している。",
              "double_keigo.established", "established"),
    SplitCase("お召し上がりになる", "指針が習慣として定着した二重敬語と明記している。",
              "double_keigo.established", "established"),
    SplitCase("お見えになる", "指針が習慣として定着した二重敬語と明記している。",
              "double_keigo.established", "established"),
    SplitCase("ご利用いただき", "「いただく」も「くださる」もどちらも適切（指針【17】）。",
              "variation.itadaku_kudasaru", "established"),
    SplitCase("ご利用いただきまして", "「いただく」も「くださる」もどちらも適切（指針【17】）。",
              "variation.itadaku_kudasaru", "established"),
    SplitCase("ご持参ください", "含まれる「参る」は謙譲語Ⅱとして働かない（指針【14】）。",
              "variation.jisan", "established"),
    SplitCase("お申し出ください", "含まれる「申す」は謙譲語Ⅱとして働かない（指針【14】）。",
              "variation.jisan", "established"),
    SplitCase("お申し込みください", "含まれる「申す」は謙譲語Ⅱとして働かない（指針【14】）。",
              "variation.jisan", "established"),
    SplitCase("申し伝えます", "＜向かう先＞である部下を立てる働きはない（指針【15】）。",
              "variation.jisan", "established"),
    SplitCase("お待ちしています", "＜向かう先＞を立てる謙譲語Ⅰ。自分側の「お」は問題ない（指針【16】）。",
              "self_sonkeigo.q16", "established"),
    SplitCase("ご説明したい", "＜向かう先＞を立てる謙譲語Ⅰ。自分側の「ご」は問題ない（指針【16】）。",
              "self_sonkeigo.q16", "established"),
    SplitCase("高いです", "形容詞＋です。既にかなりの人が許容している（指針 第2章第2-4）。",
              "variation.adj_desu", "shifting"),
    SplitCase("水をあげる", "謙譲語から美化語への変化が定着しつつある（指針 第1章第2-2）。",
              "variation.ageru", "shifting"),
    SplitCase("させていただきます", "許可と恩恵の「見立て」をどこまで受け入れるかは個人差（指針【18】）。",
              "sasete_itadaku.tolerance", "split"),
    SplitCase("お疲れ様でございました", "上司に対するねぎらいの言い換えとして指針が挙げる形（指針【27】）。",
              "principle.moderate", "established"),
)

SPLIT_SURFACES: FrozenSet[str] = frozenset(c.surface for c in SPLIT_ACCEPTABILITY)
