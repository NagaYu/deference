"""RuleCitation — 誤り種別を規範の該当分類・該当説明に対応づける。

利用者が学べる出力にすることが製品価値の中心である。指摘だけを返すツールは
「直された」という体験しか残さないが、根拠まで返せば次から自分で選べるようになる。

文言についての方針（製品要件。`tests/test_tone.py` が守る）:
    利用者の日本語を否定する物言いをしない。出力は「規範上はこうなる」という
    情報提示に留め、断定的に「間違い」と責める表現を避ける。根拠は
    指針 第1章第1-3「『自己表現』としての敬語使用」および
    第3章第1-4「敬語は過剰でなく適度に使う」。

実証する主張との対応:
    - 「根拠提示」: :meth:`RuleCitation.for_error` は全 :class:`ErrorType` に対して
      原典の章・問い番号・ページを返す。根拠のない指摘を出せない構造にしている。
    - 「過剰指摘の少なさ」: :meth:`RuleCitation.explain` は断定を避け、
      揺れについては「どちらも使える」と明示する。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from . import norms
from .types import (
    Audience,
    Citation,
    ErrorType,
    KeigoClass,
    MailContext,
    Party,
    Suggestion,
    ERROR_TYPE_JA,
    KEIGO_CLASS_JA,
)

__all__ = [
    "RuleCitation",
    "FORBIDDEN_PHRASES",
    "FORBIDDEN_PHRASES_EN",
    "KEIGO_CLASS_DEFINITION_EN",
]


#: 利用者に見せる文言に現れてはならない断定的な否定表現。
#: `tests/test_tone.py` がこの一覧で全出力を検査する。
FORBIDDEN_PHRASES: tuple = (
    "間違い",
    "間違っています",
    "誤用",
    "誤りです",
    "正しくありません",
    "ダメ",
    "だめです",
    "べきではありません",
    "してはいけません",
    "不適切です",
    "おかしいです",
)


#: 誤り種別が関わる敬語分類（UI の色分けと解説に使う）
_CLASSES: Dict[ErrorType, List[KeigoClass]] = {
    ErrorType.DOUBLE_KEIGO: [KeigoClass.SONKEIGO, KeigoClass.KENJOUGO_1],
    ErrorType.DIRECTION_SWAP: [KeigoClass.SONKEIGO, KeigoClass.KENJOUGO_1],
    ErrorType.UCHI_SONKEIGO: [KeigoClass.SONKEIGO, KeigoClass.KENJOUGO_2],
    ErrorType.SELF_SONKEIGO: [KeigoClass.SONKEIGO, KeigoClass.KENJOUGO_1],
    ErrorType.SA_INSERTION: [KeigoClass.KENJOUGO_1],
    ErrorType.SASETE_ITADAKU_OVERUSE: [KeigoClass.KENJOUGO_1],
    ErrorType.STYLE_MIXING: [KeigoClass.TEINEIGO],
    ErrorType.DEFERENCE_INVERSION: [KeigoClass.SONKEIGO, KeigoClass.KENJOUGO_2],
    ErrorType.GO_SARERU: [KeigoClass.SONKEIGO, KeigoClass.KENJOUGO_1],
    ErrorType.OGO_DEKIRU: [KeigoClass.SONKEIGO, KeigoClass.KENJOUGO_1],
    ErrorType.BAD_KEIGO_LINK: [KeigoClass.KENJOUGO_1, KeigoClass.SONKEIGO],
    ErrorType.NONE: [],
}


#: 誤り種別ごとの「仕組みの説明」（学習用）。5分類の定義を引きながら書く。
_TEACH: Dict[ErrorType, str] = {
    ErrorType.DOUBLE_KEIGO: (
        "指針は、一つの語について同じ種類の敬語を二重に使ったものを「二重敬語」と呼び、"
        "「一般に適切ではないとされている」と説明しています（第2章第2-6（2）, p.30）。"
        "たとえば「お読みになる」は既に尊敬語なので、そこへさらに「れる」を足すと"
        "同じ働きが重なります。"
        "ただし同じ箇所は「語によっては、習慣として定着しているものもある」とも述べ、"
        "「お召し上がりになる」「お見えになる」「お伺いする」を定着した例に挙げています。"
        "なお、二つの語をそれぞれ敬語にして「て」でつないだ「敬語連結」"
        "（お読みになっていらっしゃる など）は二重敬語には当たらず、基本的に許容されます。"
    ),
    ErrorType.DIRECTION_SWAP: (
        "尊敬語は「相手側又は第三者の行為……について、その人物を立てて述べるもの」、"
        "謙譲語Ⅰは「自分側から相手側又は第三者に向かう行為……について、"
        "その向かう先の人物を立てて述べるもの」です（第2章第1, p.14-15）。"
        "つまり尊敬語は〈行為をする人〉を立て、謙譲語Ⅰは〈行為の向かう先〉を立てます。"
        "同じ「お持ちする」でも、自分が持つなら向かう先の相手を立てる形になり、"
        "相手が持つ場合には相手を立てる形になりません（第3章第2-2【11】, p.37）。"
        "どちらを選ぶかは、誰の行為かで決まります。"
    ),
    ErrorType.UCHI_SONKEIGO: (
        "指針は「自分側は立てない」を尊敬語・謙譲語Ⅰの基本的な原則としています"
        "（第2章第1-6, p.22）。ここでの「自分側」には自分の家族や、"
        "自分にとって「ウチ」と認識すべき人物が含まれます。"
        "社外の方が相手のとき、自社の上位者は「ウチ」の側になるため、"
        "立てずに述べる形（謙譲語Ⅱの「申しております」など）が案内されています"
        "（第3章第3-2【25】, p.44-45）。"
        "同じ場面でも社内が相手なら、社長を立てる形が適切になります。"
        "宛先によって適切な形が入れ替わるのが、この項目の特徴です。"
    ),
    ErrorType.SELF_SONKEIGO: (
        "自分側の動作やものごとを立ててしまう形については、指針が"
        "「私のお考え」「私の御旅行」を例に挙げています（第3章第2-4【16】, p.39）。"
        "一方で「お待ちしています」「ご説明をしたいのですが」のように、"
        "＜向かう先＞を立てる謙譲語Ⅰであれば、自分の動作に「お」「ご」が付いても"
        "問題はないとされています。"
        "同じ「お」でも、立てている相手が誰かで働きが変わります。"
    ),
    ErrorType.SA_INSERTION: (
        "五段活用の動詞の使役形は「休む→休ませる」「読む→読ませる」のように"
        "「〜せる」になります。「休まさせる」「読まさせる」は、"
        "一段活用の「食べさせる」の形が五段活用にも及んだものと説明されることが多く、"
        "いわゆる「さ入れ言葉」と呼ばれます。"
        "なお「敬語の指針」はこの語形を扱っていないため、ここでの根拠は"
        "指針ではなく活用の規則です。話し言葉では広く使われており、"
        "文化庁「国語に関する世論調査」でも「気にならない」という回答が多数を占めています。"
    ),
    ErrorType.SASETE_ITADAKU_OVERUSE: (
        "指針は「（お・ご）……(さ)せていただく」について、"
        "「相手側又は第三者の許可を受けて行い」「そのことで恩恵を受ける」"
        "という二つの条件を挙げています（第3章第2-6【18】, p.40）。"
        "そのうえで、条件を満たしているかのように「見立てて」使う用法があり、"
        "「その見立てをどの程度自然なものとして受け入れるかということが、"
        "その個人にとっての『許容度』を決めている」と述べています。"
        "つまり一律に適否が決まる項目ではありません。"
        "簡潔にしたい場合は「いたします」という選択もあります。"
    ),
    ErrorType.STYLE_MIXING: (
        "「です・ます」で書かれた文章に常体の文が混じると、読み手が文体の"
        "切り替わりを手掛かりとして受け取ってしまうことがあります。"
        "「敬語の指針」は敬体と常体を分類として挙げるにとどまり（第1章第2-5, p.11）、"
        "混在の可否そのものは論じていません。"
        "ここでの案内は規範の問題ではなく、文章全体の読みやすさの観点によるものです。"
    ),
    ErrorType.DEFERENCE_INVERSION: (
        "誰をどれだけ立てるかは、その場の人間関係で決まります。"
        "指針は、社外の方が多い場では「ウチ」の社長を立てない方がよいとし"
        "（第3章第3-2【25】, p.44-45）、また「相手から見れば、立てる対象とは"
        "認識されないだろう」と思われる第三者は立てない配慮が必要だとしています"
        "（第2章第1-6（3）イ, p.22-23）。"
        "自社の人物を相手側より高く述べると、敬意の向きが入れ替わって読めます。"
        "この項目は一文だけでは判断できず、本文全体の並びを見て初めて分かります。"
    ),
    ErrorType.GO_SARERU: (
        "「ご利用される」型について、指針は"
        "「規範的には、『適切な敬語』だとは位置付けられてこなかった形である」"
        "と述べ、「利用される・利用なさる・ご利用になる・ご利用なさる」を"
        "適切な形として挙げています（第3章第2-1【7】, p.36）。"
        "「ご……する」が謙譲語Ⅰを作る形なので、そこに尊敬語の「れる」が付くと"
        "謙譲語Ⅰと尊敬語の組合せに見える、というのが理由です。"
        "同じ箇所は「ご利用＋される」という成り立ちで受け止めるなら"
        "尊敬語としてあり得る形だ、という見方も紹介しています。"
    ),
    ErrorType.OGO_DEKIRU: (
        "「お（ご）……できる」は、謙譲語Ⅰの「お（ご）……する」の可能形です"
        "（第3章第2-1【8】, p.36）。自分が届けられるなら「お届けできる」で問題ありませんが、"
        "相手の行為について可能を述べる場面では尊敬語の可能形が使われます。"
        "指針は、まず尊敬語の形にしてから可能の形にする"
        "（「ご乗車になれる」）と説明しています（第2章第2-1（1）②, p.25）。"
        "「ご乗車いただけません」「ご乗車はできません」という言い方も挙げられています。"
    ),
    ErrorType.BAD_KEIGO_LINK: (
        "「伺う」は謙譲語Ⅰで、〈向かう先〉を立てる働きを持ちます"
        "（第2章第1-2, p.15）。そのため相手の動作に使うと、"
        "相手ではなく向かう先の側を立てることになります。"
        "指針は「担当者に伺ってください」を例に挙げ、"
        "「担当者にお聞きください」「担当者にお尋ねください」を案内しています"
        "（第3章第2-2【10】, p.37）。"
        "なお「伺ってくださる」が問題なく使える文脈もあり、指針は"
        "「田中さんが先生のところに伺ってくださいました」を挙げています（p.31）。"
    ),
    ErrorType.NONE: (
        "指針は敬語を、尊敬語・謙譲語Ⅰ・謙譲語Ⅱ（丁重語）・丁寧語・美化語の"
        "5種類に分けて説明しています（第2章第1）。"
        "尊敬語は〈行為をする人〉を、謙譲語Ⅰは〈行為の向かう先〉を立て、"
        "謙譲語Ⅱは誰も立てずに相手へ丁重に述べます。"
    ),
}


#: 利用者に見える英語の文言に現れてはならない断定的な否定表現。
#: 日本語側と同じ方針（利用者の言葉遣いを否定しない）を英語でも守る。
FORBIDDEN_PHRASES_EN: tuple = (
    "is wrong",
    "is incorrect",
    "you must not",
    "you should not",
    "never use",
    "bad Japanese",
    "mistake",
    "error in your",
)


#: 誤り種別ごとの英語解説。原典からの引用は日本語のまま残し、
#: 英訳を添える。引用を英語に置き換えてしまうと、原典に当たれなくなるため。
_TEACH_EN: Dict[ErrorType, str] = {
    ErrorType.DOUBLE_KEIGO: (
        "The guidelines define *nijuu keigo* as using two honorifics of the same "
        "kind on a single word, and note that it \"is generally regarded as not "
        "appropriate\" (Ch.2 Sec.2-6(2), p.30). "
        "\"O-yomi ni naru\" is already respectful, so adding \"-reru\" on top repeats "
        "the same function. "
        "The same passage adds that \"depending on the word, some forms have become "
        "established by custom\", listing o-meshiagari ni naru, o-mie ni naru and "
        "o-ukagai suru as established examples. "
        "Note also that an *honorific chain* — two words each made honorific and "
        "joined by \"-te\" (o-yomi ni natte irassharu) — is not nijuu keigo and is "
        "generally acceptable."
    ),
    ErrorType.DIRECTION_SWAP: (
        "Sonkeigo describes \"the actions ... of the other party or a third party, "
        "speaking of that person in a raised manner\"; kenjougo I describes \"actions "
        "directed from one's own side toward the other party or a third party, "
        "raising the person the action is directed to\" (Ch.2 Sec.1, pp.14-15). "
        "So sonkeigo raises **the one who acts**, and kenjougo I raises **the one the "
        "act is directed to**. "
        "The very same string \"o-mochi shimasu\" raises the reader when the writer "
        "carries something, but does not raise anyone when the reader carries it "
        "(Ch.3 Sec.2-2 Q11, p.37). "
        "Which one fits depends on whose action it is."
    ),
    ErrorType.UCHI_SONKEIGO: (
        "The guidelines state the basic principle as \"one does not raise one's own "
        "side\" (Ch.2 Sec.1-6, p.22). \"One's own side\" includes your family and "
        "anyone you would treat as *uchi* (in-group). "
        "When writing to someone outside your company, your own senior colleagues "
        "fall on the *uchi* side, so the guidelines suggest a form that does not "
        "raise them — kenjougo II such as \"moushite orimashita\" (Ch.3 Sec.3-2 Q25, "
        "pp.44-45). "
        "In the same situation addressed internally, raising the president is "
        "appropriate. **The appropriate form flips with the audience**, which is why "
        "Deference takes the audience as a required input."
    ),
    ErrorType.SELF_SONKEIGO: (
        "For raising one's own actions or belongings, the guidelines give "
        "\"watashi no o-kangae\" and \"watashi no go-ryokou\" as examples "
        "(Ch.3 Sec.2-4 Q16, p.39). "
        "By contrast, \"o-machi shite imasu\" and \"go-setsumei wo shitai no desu ga\" "
        "are kenjougo I, raising the person the act is directed to, and attaching "
        "\"o-\"/\"go-\" to your own action there is described as no problem at all. "
        "The same prefix behaves differently depending on who is being raised."
    ),
    ErrorType.SA_INSERTION: (
        "The causative of a godan verb is formed with \"-seru\": yasumu → yasumaseru, "
        "yomu → yomaseru. Forms like \"yasumasaseru\" and \"yomasaseru\" are usually "
        "explained as the ichidan pattern (tabesaseru) spreading to godan verbs, and "
        "are known as *sa-ire kotoba*. "
        "**The Keigo no Shishin does not discuss this form**, so Deference does not "
        "cite the guidelines here; the basis is the conjugation rule itself. "
        "In speech the form is widespread — the Agency for Cultural Affairs' public "
        "opinion survey found a majority were \"not bothered\" by it."
    ),
    ErrorType.SASETE_ITADAKU_OVERUSE: (
        "The guidelines describe \"(o/go-)...(sa)sete itadaku\" as basically used when "
        "(a) you act with the permission of the other party or a third party, and "
        "(b) you receive some benefit from doing so (Ch.3 Sec.2-6 Q18, p.40). "
        "They then note a usage in which those conditions are merely *treated as if* "
        "they held, and say that \"how naturally one accepts that framing is what "
        "determines an individual's tolerance\" for the expression. "
        "In other words, this is not a matter with a single verdict. "
        "If you want something more concise, \"itashimasu\" is an option."
    ),
    ErrorType.STYLE_MIXING: (
        "When plain-form sentences appear inside a desu/masu text, readers may take "
        "the switch itself as a signal. "
        "The Keigo no Shishin mentions polite and plain style only as a classification "
        "(Ch.1 Sec.2-5, p.11) and does not discuss whether mixing them is acceptable. "
        "What is offered here is a readability observation, not a question of norm."
    ),
    ErrorType.DEFERENCE_INVERSION: (
        "How high you place each person follows from the relationships in the "
        "situation. The guidelines say that where many people from outside the "
        "company are present it is better not to raise your own president "
        "(Ch.3 Sec.3-2 Q25, pp.44-45), and that a third party who \"would probably not "
        "be recognised as someone to raise, from the other party's point of view\" "
        "calls for the care of not raising them (Ch.2 Sec.1-6(3)-i, pp.22-23). "
        "Placing your own colleagues above the reader reads as the direction of "
        "deference being reversed. "
        "**This cannot be judged from a single sentence** — it only becomes visible "
        "across the whole message."
    ),
    ErrorType.GO_SARERU: (
        "On the \"go-...-sareru\" pattern the guidelines say it \"has not been "
        "positioned as an appropriate honorific in normative terms\", and give "
        "riyou sareru / riyou nasaru / go-riyou ni naru / go-riyou nasaru as the "
        "appropriate forms (Ch.3 Sec.2-1 Q7, p.36). "
        "The reason is that \"go-...-suru\" forms kenjougo I, so adding the respectful "
        "\"-reru\" looks like a kenjougo I + sonkeigo combination. "
        "The same passage also presents the alternative reading \"go-riyou + sareru\", "
        "under which the form could stand as sonkeigo."
    ),
    ErrorType.OGO_DEKIRU: (
        "\"O(go)-...-dekiru\" is the potential of the kenjougo I form "
        "\"o(go)-...-suru\" (Ch.3 Sec.2-1 Q8, p.36). \"O-todoke dekiru\" is fine when "
        "*you* can deliver, but where the potential concerns the reader's action a "
        "respectful potential is used. "
        "The guidelines explain that you first make the respectful form and then the "
        "potential — \"go-jousha ni nareru\" (Ch.2 Sec.2-1(1)-2, p.25). "
        "\"Go-jousha itadakemasen\" and \"go-jousha wa dekimasen\" are also listed."
    ),
    ErrorType.BAD_KEIGO_LINK: (
        "\"Ukagau\" is kenjougo I and raises the person the act is directed to "
        "(Ch.2 Sec.1-2, p.15). Used for the reader's action, it therefore raises the "
        "destination rather than the reader. "
        "The guidelines take \"tantousha ni ukagatte kudasai\" as their example and "
        "suggest \"tantousha ni o-kiki kudasai\" or \"o-tazune kudasai\" "
        "(Ch.3 Sec.2-2 Q10, p.37). "
        "Note that \"ukagatte kudasaru\" is perfectly usable in some contexts; the "
        "guidelines give \"Tanaka-san ga sensei no tokoro ni ukagatte kudasaimashita\" "
        "(p.31)."
    ),
    ErrorType.NONE: (
        "The guidelines describe Japanese honorifics in five classes: sonkeigo "
        "(respectful), kenjougo I (humble, raising the target), kenjougo II / "
        "teichougo (courteous), teineigo (polite) and bikago (beautifying) "
        "(Ch.2 Sec.1). "
        "Sonkeigo raises **the one who acts**, kenjougo I raises **the one the act is "
        "directed to**, and kenjougo II raises no one — it simply speaks courteously "
        "to the reader."
    ),
}

#: 5分類の英語定義（原文の訳。原文そのものは norms.KEIGO_CLASS_DEFINITION）
KEIGO_CLASS_DEFINITION_EN: Dict[KeigoClass, str] = {
    KeigoClass.SONKEIGO: (
        "Describes the actions, things or states of the other party or a third "
        "party, speaking of that person in a raised manner."
    ),
    KeigoClass.KENJOUGO_1: (
        "Describes actions or things directed from one's own side toward the other "
        "party or a third party, raising the person they are directed to."
    ),
    KeigoClass.KENJOUGO_2: (
        "Describes one's own actions or things courteously, addressed to the person "
        "one is speaking or writing to."
    ),
    KeigoClass.TEINEIGO: "Speaks politely to the person addressed.",
    KeigoClass.BIKAGO: "Speaks of things in a beautified manner.",
    KeigoClass.KENJOUGO_1_AND_2: (
        "\"O(go)-...-itasu\" serves as both kenjougo I and kenjougo II."
    ),
    KeigoClass.PLAIN: "Not an honorific form.",
}

class RuleCitation:
    """誤り種別 → 規範の該当箇所、および利用者向けの説明文を作る。

    実証する主張: 「根拠提示」。すべての指摘が原典のページに辿れることを、
    このクラスの全数テストで保証する。
    """

    def __init__(self, lang: str = "en") -> None:
        """引用の解決器を作る。言語は "en" / "ja"。

        実証する主張: 「根拠提示」。全ての指摘がこの器を通って原典の該当箇所に解決される。
        """
        self.lang = "ja" if str(lang).lower().startswith("ja") else "en"
        self._cache: Dict[str, Citation] = {}

    # ------------------------------------------------------------------
    def for_error(
        self,
        error_type: ErrorType,
        *,
        context: Optional[MailContext] = None,
        role: object = None,
    ) -> Citation:
        """誤り種別（＋文脈）に最も適した引用を返す。

        実証する主張: 「根拠提示」と「向きの誤り検出」。文脈で引用が変わること
        自体が、この種の誤りが文脈依存であることの証拠になっている。
        身内敬語は宛先が社外なら指針【25】、社内なら【26】が該当する。
        """
        key = norms.CITATION_FOR_ERROR.get(error_type)
        if error_type is ErrorType.UCHI_SONKEIGO and context is not None:
            key = (
                "uchi_soto.q25"
                if context.audience is Audience.EXTERNAL
                else "uchi_soto.q26"
            )
        elif error_type is ErrorType.DIRECTION_SWAP and role is not None:
            actor = getattr(role, "actor", None)
            keigo = getattr(role, "keigo_class", None)
            if actor is not None and getattr(actor, "is_self_side", False):
                key = "principle.self_not_raised"
            elif keigo is KeigoClass.KENJOUGO_2:
                key = "direction.q13"
            else:
                key = "direction.q11"
        if key is None:
            key = "principle.self_expression"
        return norms.cite(key)

    # ------------------------------------------------------------------
    def explain(
        self,
        error_type: ErrorType,
        *,
        context: Optional[MailContext] = None,
        role: object = None,
        suggestions: Optional[Sequence[Suggestion]] = None,
        surface: str = "",
    ) -> str:
        """利用者に見せる説明文を組み立てる。

        断定的に「間違い」と責めない。「指針では〜と説明されています」
        「規範上は〜の形が案内されています」という情報提示の文体にする
        （指針 第1章第1-3「自己表現」としての敬語使用）。

        実証する主張: 「根拠提示」と「過剰指摘の少なさ」。
        """
        cit = self.for_error(error_type, context=context, role=role)
        if self.lang == "en":
            head = self._head_en(
                error_type, context=context, role=role, surface=surface
            )
            body = f"The relevant passage is {cit.section_label('en')} ({cit.page})."
            if cit.quote:
                # 引用は日本語のまま残す。英訳に置き換えると原典に当たれなくなる。
                body += f" It reads: 「{cit.quote}」"
            tail = ""
            if suggestions:
                forms = " / ".join(dict.fromkeys(s.text for s in suggestions[:4]))
                tail = f"Forms in line with the guidelines include 「{forms}」."
            return " ".join(x for x in (head, body, tail) if x)

        head = self._head(error_type, context=context, role=role, surface=surface)
        body = f"指針の該当箇所は{cit.section}（{cit.page}）です。"
        if cit.quote:
            body += f"「{cit.quote}」と説明されています。"
        tail = ""
        if suggestions:
            forms = "／".join(dict.fromkeys(s.text for s in suggestions[:4]))
            tail = f"規範に沿った形としては「{forms}」などが挙げられます。"
        return " ".join(x for x in (head, body, tail) if x)

    def _head_en(
        self,
        error_type: ErrorType,
        *,
        context: Optional[MailContext],
        role: object,
        surface: str,
    ) -> str:
        """英語版の1文目。日本語版と同じく、責めずに何が起きているかを述べる。

        実証する主張: 「過剰指摘の少なさ」。英語でも断定的な否定を避ける。
        `tests/test_tone.py` が FORBIDDEN_PHRASES_EN で検査する。
        """
        from .types import PARTY_EN

        q = f"「{surface}」" if surface else "This passage"
        aud = context.audience if context is not None else None
        actor = getattr(role, "actor", None)
        actor_en = PARTY_EN.get(actor, "") if actor is not None else ""

        if error_type is ErrorType.DOUBLE_KEIGO:
            return f"{q} stacks two honorifics of the same kind."
        if error_type is ErrorType.DIRECTION_SWAP:
            who = f" (the action is by {actor_en})" if actor_en else ""
            return (
                f"{q} points deference in the opposite direction from the "
                f"context{who}."
            )
        if error_type is ErrorType.UCHI_SONKEIGO:
            where = (
                "Because the message is addressed outside the company, "
                if aud is Audience.EXTERNAL
                else ""
            )
            return f"{where}{q} raises someone on your own side."
        if error_type is ErrorType.SELF_SONKEIGO:
            return f"{q} raises the writer's own action."
        if error_type is ErrorType.SA_INSERTION:
            return (
                f"{q} carries an extra 「さ」 in the causative of a godan verb — "
                "the pattern known as sa-ire kotoba."
            )
        if error_type is ErrorType.SASETE_ITADAKU_OVERUSE:
            return (
                f"{q} uses 「させていただく」. The guidelines set out conditions of "
                "permission and benefit, while also noting that tolerance for it "
                "varies between individuals."
            )
        if error_type is ErrorType.STYLE_MIXING:
            return f"{q} is in a different style from the rest of the message."
        if error_type is ErrorType.DEFERENCE_INVERSION:
            return (
                f"{q} places someone on your own side above the reader's side."
            )
        if error_type is ErrorType.GO_SARERU:
            return f"{q} is the 「ご……される」 pattern."
        if error_type is ErrorType.OGO_DEKIRU:
            return (
                f"{q} uses the potential of a kenjougo I form for the reader's "
                "action."
            )
        if error_type is ErrorType.BAD_KEIGO_LINK:
            return (
                f"{q} chains a kenjougo I form onto the reader's action."
            )
        return f"{q} is worth a second look."

    def _head(
        self,
        error_type: ErrorType,
        *,
        context: Optional[MailContext],
        role: object,
        surface: str,
    ) -> str:
        """種別ごとの1文目。何が起きているかを、責めずに述べる。"""
        q = f"「{surface}」は、" if surface else "この箇所は、"
        aud = context.audience if context is not None else None
        actor = getattr(role, "actor", None)
        actor_ja = ""
        if actor is not None:
            from .types import PARTY_JA

            actor_ja = PARTY_JA.get(actor, "")

        if error_type is ErrorType.DOUBLE_KEIGO:
            return f"{q}同じ種類の敬語が二つ重なった形になっています。"
        if error_type is ErrorType.DIRECTION_SWAP:
            who = f"（動作をするのは{actor_ja}側です）" if actor_ja else ""
            return f"{q}立てる相手の向きが、文脈と入れ替わって読めます{who}。"
        if error_type is ErrorType.UCHI_SONKEIGO:
            where = "社外の方が宛先のため、" if aud is Audience.EXTERNAL else ""
            return f"{where}{q}自社の方を立てる形になっています。"
        if error_type is ErrorType.SELF_SONKEIGO:
            return f"{q}書き手ご自身の動作を立てる形になっています。"
        if error_type is ErrorType.SA_INSERTION:
            return f"{q}五段活用の使役形に「さ」が入った、いわゆる「さ入れ言葉」の形です。"
        if error_type is ErrorType.SASETE_ITADAKU_OVERUSE:
            return (
                f"{q}「させていただく」が使われています。"
                "指針は許可と恩恵という条件を挙げていますが、許容度には個人差があるとも"
                "述べています。"
            )
        if error_type is ErrorType.STYLE_MIXING:
            return f"{q}本文の他の部分と文体が異なっています。"
        if error_type is ErrorType.DEFERENCE_INVERSION:
            return f"{q}自社側の方を、相手側より高く述べる並びになっています。"
        if error_type is ErrorType.GO_SARERU:
            return f"{q}「ご……される」の形です。"
        if error_type is ErrorType.OGO_DEKIRU:
            return f"{q}謙譲語Ⅰの可能形が、相手の行為について使われています。"
        if error_type is ErrorType.BAD_KEIGO_LINK:
            return f"{q}謙譲語Ⅰを相手の動作に当てた敬語連結になっています。"
        return f"{q}気づいた点があります。"

    # ------------------------------------------------------------------
    def teach(self, error_type: ErrorType) -> str:
        """その誤り種別が関わる敬語の仕組みを解説する（学習用）。

        実証する主張: 「根拠提示」。指摘を直すだけでなく、次から自分で選べる
        ようにすることが製品価値の中心である。
        """
        table = _TEACH_EN if self.lang == "en" else _TEACH
        return table.get(error_type, table[ErrorType.NONE])

    def classes_involved(self, error_type: ErrorType) -> List[KeigoClass]:
        """その誤り種別が関わる敬語分類。UI の色分けに使う。

        実証する主張: 「根拠提示」。UI の色分けと解説を、指摘と同じ分類体系に揃える。
        """
        return list(_CLASSES.get(error_type, []))

    def class_definition(self, keigo_class: KeigoClass) -> str:
        """5分類の定義文（指針 第2章第1）。

        実証する主張: 「根拠提示」。分類の定義は指針 第2章第1 の原文から引く。
        """
        if self.lang == "en":
            return KEIGO_CLASS_DEFINITION_EN.get(keigo_class, "")
        return norms.KEIGO_CLASS_DEFINITION.get(keigo_class, "")

    # ------------------------------------------------------------------
    def variation_note(self, reason: str, citation: Optional[Citation]) -> str:
        """揺れについての説明文。指摘ではなく参考として示す。

        実証する主張: 「過剰指摘の少なさ」。揺れは「直すべき箇所」ではなく
        「許容度が割れる箇所」として提示する。
        """
        if self.lang == "en":
            head = (
                "This form is treated as one where acceptability varies by "
                "situation and generation, so it is not reported as an issue."
            )
            src = ""
            if citation is not None:
                src = f"({citation.source} {citation.section} {citation.page})"
            return " ".join(x for x in (head, reason or "", src) if x).strip()

        head = "この形は、場面や世代で受け止め方が分かれるものとして整理しています。"
        body = reason or ""
        src = ""
        if citation is not None:
            src = f"（{citation.source} {citation.section} {citation.page}）"
        return " ".join(x for x in (head, body, src) if x).strip()

    # ------------------------------------------------------------------
    def disclaimer(self) -> str:
        """出力の末尾に添える注記。UI・CLI の両方で使う。

        実証する主張: 「過剰指摘の少なさ」。出力が規範上の整理であって書き手の否定ではないことを、毎回明示する。
        """
        if self.lang == "en":
            return (
                "Honorifics are chosen as a form of self-expression "
                "(Keigo no Shishin, Ch.1 Sec.1-3). What is shown here is how the "
                "guidelines organise the matter; it is not a judgement on the "
                "writer's Japanese. Depending on the situation and your "
                "relationship with the reader, other choices are equally available."
            )
        return (
            "敬語は「自己表現」として選ぶものです（敬語の指針 第1章第1-3）。"
            "ここに示したのは規範上の整理であって、書き手の言葉遣いを否定するものでは"
            "ありません。場面やお相手との関係に応じて、別の選び方もあり得ます。"
        )

    def source_note(self) -> str:
        """出典表記。

        実証する主張: 「根拠提示」。出典表記を CLI・UI・データセットカードで共有する。
        """
        return norms.LICENSE_NOTE
