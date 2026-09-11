"""Deference の共有データ契約（全モジュールがこの型に対してコードを書く）。

実証する主張との対応:
    - 「敬意の向きの誤り検出」: :class:`Party` / :class:`MailContext` が
      〈誰の行為か〉〈誰を立てるか〉を一級の情報として持つ。表層文字列だけを
      見る規則ベースの校正ツールにはこの情報が無いため、向きの誤りを原理的に
      判定できない。この型定義がその差の出発点である。
    - 「過剰指摘の少なさ」: :class:`Verdict` が ``NORM_DIVERGENCE`` と
      ``VARIATION`` を別ラベルとして持つ。揺れを誤りに混ぜない構造を型で強制する。
    - 「根拠提示」: :class:`Finding` は :class:`Citation` を必須フィールドとして
      持つ。根拠なしの指摘を型レベルで作れないようにしている。
    - 「速度」: :class:`CheckResult` に ``elapsed_ms`` を持たせ、CPU 応答時間を
      常に計測・記録できるようにする。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Audience",
    "Party",
    "KeigoClass",
    "ErrorType",
    "Verdict",
    "FunctionTag",
    "Span",
    "Person",
    "MailContext",
    "Citation",
    "Suggestion",
    "Finding",
    "CheckResult",
    "GeneratedSentence",
    "InjectedError",
    "InjectedSample",
    "VariationCase",
    "ERROR_TYPE_JA",
    "PARTY_JA",
    "KEIGO_CLASS_JA",
    "AUDIENCE_JA",
    "ERROR_TYPE_EN",
    "PARTY_EN",
    "KEIGO_CLASS_EN",
    "AUDIENCE_EN",
    "error_type_name",
    "to_jsonable",
]


# ---------------------------------------------------------------------------
# 列挙型
# ---------------------------------------------------------------------------


class Audience(str, Enum):
    """文章の宛先。敬意の向きは宛先が決まらないと決まらない。

    実証する主張: 「向きの誤り検出」。同じ文字列でも宛先が社内か社外かで
    規範上の適否が反転する（指針 第3章第3-2【25】: 社内の忘年会なら
    「社長からごあいさつを頂きます」、社外の人が多い会なら
    「社長からごあいさつを申し上げます」）。この列挙が反転を表現する。
    """

    INTERNAL = "internal"  # 社内宛（同じ組織の中）
    EXTERNAL = "external"  # 社外宛（取引先・顧客など）
    PUBLIC = "public"  # 不特定多数（告知・掲示など）

    @classmethod
    def parse(cls, value: "str | Audience") -> "Audience":
        """文字列から宛先区分を解釈する（日本語表記も受け付ける）。

        実証する主張: 「向きの誤り検出」。宛先は必須の入力なので、CLI からも UI からも同じ規則で解釈する。
        """
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower()
        aliases = {
            "internal": cls.INTERNAL,
            "in": cls.INTERNAL,
            "社内": cls.INTERNAL,
            "external": cls.EXTERNAL,
            "ext": cls.EXTERNAL,
            "社外": cls.EXTERNAL,
            "public": cls.PUBLIC,
            "不特定": cls.PUBLIC,
        }
        if key not in aliases:
            raise ValueError(
                f"未知の宛先区分です: {value!r}（internal / external / public）"
            )
        return aliases[key]


class Party(str, Enum):
    """動作主・被動作主の立場。RoleTagger の出力語彙。

    実証する主張: 「向きの誤り検出」。指針 第2章第1 は尊敬語を
    「相手側又は第三者の行為…について、その人物を立てて述べるもの」、
    謙譲語Ⅰを「自分側から相手側又は第三者に向かう行為…について、その向かう先の
    人物を立てて述べるもの」と定義する。つまり適否の判定には最低でも
    〈自分側 / 相手側 / 第三者〉の区別が要る。この列挙がその区別である。
    """

    SELF = "self"  # 書き手自身
    SELF_GROUP = "self_group"  # 身内（自社・自部署・自分の家族）
    ADDRESSEE = "addressee"  # 読み手本人
    ADDRESSEE_GROUP = "addressee_group"  # 相手側の人物（先方の上司など）
    THIRD_PARTY = "third_party"  # 第三者
    UNKNOWN = "unknown"

    @property
    def is_self_side(self) -> bool:
        """指針の言う「自分側」（＝立ててはいけない側）か。

        実証する主張: 「向きの誤り検出」。指針の「自分側は立てない」原則（第2章第1-6, p.22）を述語として表現する。向きの判定はこの述語から始まる。
        """
        return self in (Party.SELF, Party.SELF_GROUP)

    @property
    def is_other_side(self) -> bool:
        """指針の言う「相手側又は第三者」（＝立てうる側）か。

        実証する主張: 「向きの誤り検出」。尊敬語・謙譲語Ⅰが立てうる側かどうかを判定する。
        """
        return self in (Party.ADDRESSEE, Party.ADDRESSEE_GROUP, Party.THIRD_PARTY)


class KeigoClass(str, Enum):
    """「敬語の指針」第2章第1 の5分類（+ 敬語でない素の形）。

    実証する主張: 「根拠提示」。5分類は指針 第2章第1 の枠組みそのもので、各分類が誰を立てるかを規定する。
    """

    SONKEIGO = "sonkeigo"  # 尊敬語（「いらっしゃる・おっしゃる」型）
    KENJOUGO_1 = "kenjougo_1"  # 謙譲語Ⅰ（「伺う・申し上げる」型）
    KENJOUGO_2 = "kenjougo_2"  # 謙譲語Ⅱ（丁重語）（「参る・申す」型）
    TEINEIGO = "teineigo"  # 丁寧語（「です・ます」型）
    BIKAGO = "bikago"  # 美化語（「お酒・お料理」型）
    KENJOUGO_1_AND_2 = "kenjougo_1_and_2"  # 「お(ご)……いたす」= 謙譲語Ⅰ兼Ⅱ
    PLAIN = "plain"  # 敬語形でない


class ErrorType(str, Enum):
    """注入・検出する誤り種別。

    (i)〜(vii) は依頼仕様の 7 種に対応し、末尾は指針が具体的に論じている
    派生型（規則ベースでは特に取りこぼしやすいもの）である。
    ``NONE`` は「誤りでない」、``VARIATION`` はラベルとしては
    :class:`Verdict` 側で扱うためここには置かない。

    実証する主張: 「向きの誤り検出」。種別を互いに素に保つことで、種別分類の正解率が測れる。
    """

    NONE = "none"

    # (i) 二重敬語
    DOUBLE_KEIGO = "double_keigo"
    # (ii) 尊敬語と謙譲語Ⅰの取り違え（＝敬意の「向き」の誤り）
    DIRECTION_SWAP = "direction_swap"
    # (iii) 身内に尊敬語
    UCHI_SONKEIGO = "uchi_sonkeigo"
    # (iv) さ入れ言葉
    SA_INSERTION = "sa_insertion"
    # (v) 「させていただく」の過剰使用
    SASETE_ITADAKU_OVERUSE = "sasete_itadaku_overuse"
    # (vi) 敬体と常体の混在
    STYLE_MIXING = "style_mixing"
    # (vii) 敬意の程度の不整合（相手と身内で高さが逆転）
    DEFERENCE_INVERSION = "deference_inversion"

    # --- 指針が個別に論じている派生型 -------------------------------------
    # 「ご利用される」型（謙譲語Ⅰ＋尊敬語）— 指針【7】
    GO_SARERU = "go_sareru"
    # 「ご乗車できません」型（謙譲語Ⅰ可能形を尊敬語に流用）— 指針【8】
    OGO_DEKIRU = "ogo_dekiru"
    # 「伺ってください」型（不適切な敬語連結）— 指針 第2章第2-6(3)
    BAD_KEIGO_LINK = "bad_keigo_link"
    # 自分側に尊敬語（「私が申されました」など）
    SELF_SONKEIGO = "self_sonkeigo"

    @property
    def is_direction_error(self) -> bool:
        """「敬意の向き」に関する誤りか。

        実証する主張: 「向きの誤り検出」。図表と評価はこの述語で
        向き系の誤り種別だけを切り出し、規則ベースとの差を示す。
        """
        return self in _DIRECTION_ERRORS

    @property
    def requires_context(self) -> bool:
        """文脈（誰の行為か・宛先は誰か）が無ければ判定できない誤りか。

        実証する主張: 「向きの誤り検出」。**この述語が Deference の主張の核**である。
        ここに入る種別は、同じ文字列が立場によって適否を変えるため、
        表層パターンだけを見るツールには原理的に到達できない。
        逆に、ここに入らない種別（二重敬語・さ入れ言葉・「ご利用される」型など）は
        規則ベースでも拾えるので、そこで優位を主張してはならない。
        """
        return self in _CONTEXT_REQUIRED


_DIRECTION_ERRORS = frozenset(
    {
        ErrorType.DIRECTION_SWAP,
        ErrorType.UCHI_SONKEIGO,
        ErrorType.DEFERENCE_INVERSION,
        ErrorType.BAD_KEIGO_LINK,
        ErrorType.SELF_SONKEIGO,
        ErrorType.OGO_DEKIRU,
    }
)

#: 〈誰の行為か〉〈誰に向かうか〉が分からなければ**原理的に判定できない**誤り。
#:
#: これは :data:`_DIRECTION_ERRORS` より狭い。たとえば「ご乗車できません」
#: （OGO_DEKIRU）は敬意の向きの問題だが、「ご……できません」という表層形が
#: 特徴的なので正規表現でも拾える。一方「お持ちします」は、自分が持つなら適切で
#: 相手が持つなら不適切であり、**同じ文字列のまま適否が反転する**。
#: 表層規則との差を主張できるのはこちらの集合であって、
#: 評価の図はこの区別を明示しなければ誠実な比較にならない。
_CONTEXT_REQUIRED = frozenset(
    {
        ErrorType.DIRECTION_SWAP,
        ErrorType.UCHI_SONKEIGO,
        ErrorType.SELF_SONKEIGO,
        ErrorType.DEFERENCE_INVERSION,
        ErrorType.BAD_KEIGO_LINK,
    }
)


class Verdict(str, Enum):
    """指摘の性格。誤りと揺れを分けるための最重要フィールド。

    実証する主張: 「過剰指摘の少なさ」。指針 第1章第2-2 は世代や性による
    敬語意識の多様性を指摘し、画一的に断ずる態度を避けるべきだと述べる。
    したがって許容度が割れる表現は ``VARIATION`` として別扱いにし、
    既定では赤く出さない。
    """

    NORM_DIVERGENCE = "norm_divergence"  # 規範上は別の形が案内されているもの
    VARIATION = "variation"  # 許容度が割れる「揺れ」。指摘しない
    OK = "ok"


class FunctionTag(str, Enum):
    """ビジネス文書の機能タグ。データの網羅性を測るために使う。

    実証する主張: 「過剰指摘の少なさ」。ビジネス文書の機能を網羅することで、偽陽性を測る負例が実務の分布に近づく。
    """

    REQUEST = "request"  # 依頼
    REFUSAL = "refusal"  # 断り
    APOLOGY = "apology"  # 謝罪
    THANKS = "thanks"  # 感謝
    REPORT = "report"  # 報告・連絡
    INQUIRY = "inquiry"  # 問い合わせ
    SCHEDULING = "scheduling"  # 日程調整
    NOTICE = "notice"  # 告知
    GIVING = "giving"  # 授受（差し上げる）
    RECEIVING = "receiving"  # 授受（いただく／くださる）
    GREETING = "greeting"  # あいさつ


# ---------------------------------------------------------------------------
# 値オブジェクト
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """文字オフセットで表す範囲。``text[start:end]`` と一致することを保証する。

    実証する主張: 「向きの誤り検出」の評価可能性。誤りは文単位ではなく
    スパン単位で持つため、注入位置と検出位置を厳密に突き合わせられる。
    """

    start: int
    end: int
    text: str = ""

    def __post_init__(self) -> None:
        """不正なスパンを構築時に弾く。

        実証する主張: 「向きの誤り検出」の評価可能性。壊れたスパンを持ち回らない。
        """
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"不正なスパンです: [{self.start}, {self.end})")

    def __len__(self) -> int:
        """スパンの文字数。

        実証する主張: 「向きの誤り検出」の評価可能性。スパンの大きさは重なり判定の基礎になる。
        """
        return self.end - self.start

    def overlaps(self, other: "Span") -> bool:
        """2つのスパンが重なるか。

        実証する主張: 「向きの誤り検出」の評価可能性。注入位置と検出位置の突き合わせに使う。
        """
        return self.start < other.end and other.start < self.end

    def iou(self, other: "Span") -> float:
        """Jaccard 係数。部分一致の評価に使う。

        実証する主張: 「向きの誤り検出」の評価可能性。注入位置と検出位置の重なりを数値化し、部分一致を含めて再現率を測る。
        """
        inter = max(0, min(self.end, other.end) - max(self.start, other.start))
        union = max(self.end, other.end) - min(self.start, other.start)
        return inter / union if union else 0.0

    def shifted(self, delta: int) -> "Span":
        """オフセットをずらしたスパンを返す（本文単位への埋め込みに使う）。

        実証する主張: 「向きの誤り検出」。敬意の逆転は本文全体でしか判定できないため、文単位の位置を本文基準に移す操作が要る。
        """
        return Span(self.start + delta, self.end + delta, self.text)

    def bind(self, source: str) -> "Span":
        """``source`` から実テキストを取り直した Span を返す。

        実証する主張: 「向きの誤り検出」の評価可能性。スパンと本文の不整合を持ち込まないための正規化。
        """
        return Span(self.start, self.end, source[self.start : self.end])


@dataclass(frozen=True)
class Person:
    """文中に現れうる人物と、その立場。

    実証する主張: 「向きの誤り検出」。人物名だけでは適否が決まらず、その人物が自分側か相手側かで決まる。
    """

    name: str
    side: Party
    title: str = ""  # 部長・課長・先生 など
    org: str = ""

    @property
    def display(self) -> str:
        """表示用の氏名（役職込み）。

        実証する主張: 「向きの誤り検出」。社外宛では身内に役職を付けない（指針【24】）ため、表示と立場は別に持つ。
        """
        return f"{self.name}{self.title}" if self.title else self.name


@dataclass(frozen=True)
class MailContext:
    """メール1通ぶんのメタ情報。

    実証する主張: 「向きの誤り検出」。単文だけを見ても
    「田中がおっしゃいました」が誤りかどうかは決まらない。田中が自社の人間か
    先方の人間か、そして宛先が社内か社外かで決まる。この文脈を明示的に
    受け取ることが Deference と表層規則ツールの構造的な差である。
    """

    audience: Audience = Audience.EXTERNAL
    writer_name: str = "山田"
    writer_title: str = ""
    writer_org: str = "弊社"
    recipient_name: str = "佐藤"
    recipient_title: str = "様"
    recipient_org: str = "貴社"
    persons: Tuple[Person, ...] = field(default_factory=tuple)

    def side_of(self, name: str) -> Party:
        """人物名から立場を引く。未知なら ``Party.UNKNOWN``。

        実証する主張: 「向きの誤り検出」。文中の人物名を立場に解決する。表層規則が持たない情報はここに集約される。
        """
        for p in self.persons:
            if p.name and p.name in name:
                return p.side
        if self.writer_name and self.writer_name in name:
            return Party.SELF
        if self.recipient_name and self.recipient_name in name:
            return Party.ADDRESSEE
        return Party.UNKNOWN

    def with_persons(self, persons: Sequence[Person]) -> "MailContext":
        """人物表を差し替えた文脈を返す。

        実証する主張: 「向きの誤り検出」。人物の立場が変われば同じ本文の適否も変わる。
        """
        return MailContext(
            audience=self.audience,
            writer_name=self.writer_name,
            writer_title=self.writer_title,
            writer_org=self.writer_org,
            recipient_name=self.recipient_name,
            recipient_title=self.recipient_title,
            recipient_org=self.recipient_org,
            persons=tuple(persons),
        )


#: 出典名の英訳。原典は日本語なので、英語表記のあとに原題を残す。
_SOURCE_EN: Dict[str, str] = {
    "文化審議会答申「敬語の指針」（平成19年2月2日）": (
        "Council for Cultural Affairs, Keigo no Shishin (2007-02-02)"
    ),
    "文化庁「国語に関する世論調査」": (
        "Agency for Cultural Affairs, Public Opinion Survey on the Japanese Language"
    ),
    "活用規則（「敬語の指針」に該当記述なし）": (
        "Conjugation rule (not covered by the Keigo no Shishin)"
    ),
}


@dataclass(frozen=True)
class Citation:
    """規範の該当箇所。

    実証する主張: 「根拠提示」。指摘には必ず、指針のどの章・どの問い・
    何ページに基づくのかを添える。引用は要点のみ・短く、出典 URL を併記する。
    全文の再配布はしない。
    """

    source: str  # 例: 文化審議会答申「敬語の指針」（平成19年2月2日）
    section: str  # 例: 第2章 第2 6(2)「二重敬語」とその適否
    page: str  # 例: p.30
    quote: str  # 要点の短い引用（原文）
    url: str = "https://www.bunka.go.jp/seisaku/bunkashingikai/kokugo/hokoku/pdf/keigo_tosin.pdf"
    note: str = ""  # 指針本文ではない補足（Deference 側の解説）

    def source_label(self, lang: str = "ja") -> str:
        """出典名を、言語に合わせた表記で返す。

        実証する主張: 「根拠提示」。英語の読者にも、何に基づく指摘なのかが
        一目で伝わるようにする。原典が日本語であることは括弧で明示する。
        """
        if lang != "en":
            return self.source
        return _SOURCE_EN.get(self.source, self.source)

    def section_label(self, lang: str = "ja") -> str:
        """章・節・問い番号を、言語に合わせた表記に直す。

        「第3章 第3-2【25】」→「Ch.3 Sec.3-2 Q25」。英語圏の読者が原典の
        どこを見ればよいか分かるようにするための変換で、41件の引用に
        英語見出しを手で持たせるより取り違えが起きにくい。

        実証する主張: 「根拠提示」。英語で読んでも原典の該当箇所に辿り着ける。
        """
        if lang != "en":
            return self.section
        import re as _re

        out = self.section
        out = _re.sub(r"第(\d+)章", r"Ch.\1", out)
        out = _re.sub(r"第(\d+)-(\d+)", r"Sec.\1-\2", out)
        out = _re.sub(r"第(\d+)（(\d+)）", r"Sec.\1(\2)", out)
        out = _re.sub(r"第(\d+)", r"Sec.\1", out)
        out = _re.sub(r"【(\d+)】", r" Q\1", out)
        out = _re.sub(r"（(\d+)）", r"(\1)", out)
        return _re.sub(r"\s{2,}", " ", out.replace("　", " ")).strip()

    def render(self, lang: str = "ja") -> str:
        """引用を一行の文字列にする。

        実証する主張: 「根拠提示」。CLI・UI・データセットカードが同じ書式で出典を示す。
        """
        if lang == "en":
            head = f"{self.source} {self.section_label('en')} ({self.page})"
        else:
            head = f"{self.source} {self.section}（{self.page}）"
        body = f"「{self.quote}」" if self.quote else ""
        return f"{head} {body}".strip()


@dataclass(frozen=True)
class Suggestion:
    """修正候補。

    実証する主張: 「修正候補の妥当性」。``generated_by`` が示すとおり候補は
    必ず Generator（規則）が作れる形の集合から来る。自由生成しないので、
    修正案そのものが新たな規範逸脱になることがない。
    """

    text: str  # 置換後の文字列（スパンを置き換える）
    keigo_class: KeigoClass = KeigoClass.PLAIN
    reason: str = ""
    generated_by: str = "rule"  # "rule" のみを許す運用（自由生成を混ぜない）


@dataclass(frozen=True)
class Finding:
    """1件の指摘。

    実証する主張: 4つすべて。向き（``error_type``）・過剰指摘の抑制
    （``verdict``）・根拠（``citation``）を1つの構造体に束ね、
    UI と CLI と評価スクリプトが同じ対象を見るようにする。
    """

    span: Span
    error_type: ErrorType
    verdict: Verdict = Verdict.NORM_DIVERGENCE
    confidence: float = 1.0
    suggestions: Tuple[Suggestion, ...] = field(default_factory=tuple)
    citation: Optional[Citation] = None
    message: str = ""  # 利用者に見せる非断定的な説明
    detector: str = "deference"  # 由来（deference / textlint / rule_baseline / llm）
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_reportable(self) -> bool:
        """既定で利用者に提示するか。揺れは提示しない。

        実証する主張: 「過剰指摘の少なさ」。揺れを既定の出力から外すのはこの述語一つで、UI・CLI・評価が同じ基準を共有する。
        """
        return self.verdict is Verdict.NORM_DIVERGENCE


@dataclass
class CheckResult:
    """1通ぶんの検査結果。

    実証する主張: 「速度」。elapsed_ms を必ず持たせ、応答時間を条件間で横並びに測れるようにする。
    """

    text: str
    context: MailContext
    findings: List[Finding] = field(default_factory=list)
    elapsed_ms: float = 0.0
    engine: str = "deference"
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def reportable(self) -> List[Finding]:
        """既定で利用者に提示する指摘（揺れを含まない）。

        実証する主張: 「過剰指摘の少なさ」。
        """
        return [f for f in self.findings if f.is_reportable]

    @property
    def variations(self) -> List[Finding]:
        """揺れとして扱った箇所。指摘ではない。

        実証する主張: 「過剰指摘の少なさ」。
        """
        return [f for f in self.findings if f.verdict is Verdict.VARIATION]


# ---------------------------------------------------------------------------
# データ生成側の型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedSentence:
    """Generator が規範から構成した「正しい」文。

    実証する主張: 「過剰指摘の少なさ」。ここで作った文は定義上すべて規範に
    沿っているので、これらに指摘が出れば偽陽性である。テストと評価の
    負例集合として使う。
    """

    text: str
    context: MailContext
    actor: Party
    target: Party
    predicate: str  # 元の素の動詞（例: 言う）
    keigo_class: KeigoClass
    politeness: int  # 0=常体, 1=です・ます, 2=より改まった形
    function: FunctionTag = FunctionTag.REPORT
    predicate_span: Optional[Span] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InjectedError:
    """注入した誤り1件の台帳。位置と種別を厳密に保持する。

    実証する主張: 「向きの誤り検出」の評価可能性。ラベルは規則で作った
    ものなので、LLM による教師バイアスが入らない。
    """

    span: Span
    error_type: ErrorType
    original_text: str  # 注入前の元の表層
    gold_suggestions: Tuple[str, ...] = field(default_factory=tuple)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InjectedSample:
    """誤りを注入した学習・評価サンプル。

    実証する主張: 「向きの誤り検出」の評価可能性。誤りの位置と種別が既知なので、種別ごとの再現率を出せる。
    """

    text: str
    context: MailContext
    errors: Tuple[InjectedError, ...] = field(default_factory=tuple)
    source_text: str = ""  # 注入前の正しい文
    meta: Dict[str, Any] = field(default_factory=dict)

    def verify(self) -> None:
        """スパンが実テキストと一致することを検査する。

        実証する主張: 「向きの誤り検出」の評価可能性。スパンと本文がずれた台帳は評価を壊すので、生成時に必ず検査する。
        """
        for e in self.errors:
            actual = self.text[e.span.start : e.span.end]
            if e.span.text and actual != e.span.text:
                raise AssertionError(
                    f"スパン不整合: 期待 {e.span.text!r} / 実際 {actual!r}"
                )


@dataclass(frozen=True)
class VariationCase:
    """揺れの事例。誤りではない。

    実証する主張: 「過剰指摘の少なさ」。指針が「習慣として定着している」
    「許容される」「個人差が大きい」と述べている表現、および世代差・場面差で
    許容度が割れる表現をここに集め、既定で指摘を出さないことを検証する。
    """

    text: str
    context: MailContext
    focus: Span
    reason: str  # なぜ揺れなのか
    citation: Optional[Citation] = None
    acceptability: str = "split"  # established / split / shifting
    related_error_type: ErrorType = ErrorType.NONE
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 日本語表示名
# ---------------------------------------------------------------------------

ERROR_TYPE_JA: Dict[ErrorType, str] = {
    ErrorType.NONE: "誤りなし",
    ErrorType.DOUBLE_KEIGO: "二重敬語",
    ErrorType.DIRECTION_SWAP: "尊敬語と謙譲語Ⅰの取り違え",
    ErrorType.UCHI_SONKEIGO: "身内に尊敬語",
    ErrorType.SA_INSERTION: "さ入れ言葉",
    ErrorType.SASETE_ITADAKU_OVERUSE: "「させていただく」の過剰使用",
    ErrorType.STYLE_MIXING: "敬体と常体の混在",
    ErrorType.DEFERENCE_INVERSION: "敬意の程度の不整合",
    ErrorType.GO_SARERU: "「ご利用される」型",
    ErrorType.OGO_DEKIRU: "「ご乗車できません」型",
    ErrorType.BAD_KEIGO_LINK: "不適切な敬語連結",
    ErrorType.SELF_SONKEIGO: "自分側に尊敬語",
}

#: 誤り種別の英語表示名。GitHub / Hugging Face 向けの英語 UI で使う。
#: 敬語の分類名（尊敬語・謙譲語Ⅰ…）は原典が日本語なので、英語側では
#: ローマ字＋訳語を併記して、原典に当たれる状態を保つ。
ERROR_TYPE_EN: Dict[ErrorType, str] = {
    ErrorType.NONE: "no issue",
    ErrorType.DOUBLE_KEIGO: "Doubled honorific (nijuu keigo)",
    ErrorType.DIRECTION_SWAP: "Respectful/humble mix-up (direction)",
    ErrorType.UCHI_SONKEIGO: "Respectful form for one's own side (uchi)",
    ErrorType.SA_INSERTION: "Inserted 'sa' (sa-ire kotoba)",
    ErrorType.SASETE_ITADAKU_OVERUSE: "Overuse of 'sasete itadaku'",
    ErrorType.STYLE_MIXING: "Mixed polite/plain style",
    ErrorType.DEFERENCE_INVERSION: "Inverted deference level",
    ErrorType.GO_SARERU: "'go-...-sareru' form",
    ErrorType.OGO_DEKIRU: "'o/go-...-dekiru' form",
    ErrorType.BAD_KEIGO_LINK: "Ill-formed honorific chain",
    ErrorType.SELF_SONKEIGO: "Respectful form for oneself",
}

PARTY_EN: Dict[Party, str] = {
    Party.SELF: "the writer",
    Party.SELF_GROUP: "the writer's side (in-group)",
    Party.ADDRESSEE: "the reader",
    Party.ADDRESSEE_GROUP: "the reader's side",
    Party.THIRD_PARTY: "a third party",
    Party.UNKNOWN: "unknown",
}

KEIGO_CLASS_EN: Dict[KeigoClass, str] = {
    KeigoClass.SONKEIGO: "sonkeigo (respectful)",
    KeigoClass.KENJOUGO_1: "kenjougo I (humble, raises the target)",
    KeigoClass.KENJOUGO_2: "kenjougo II / teichougo (courteous)",
    KeigoClass.TEINEIGO: "teineigo (polite)",
    KeigoClass.BIKAGO: "bikago (beautifying)",
    KeigoClass.KENJOUGO_1_AND_2: "kenjougo I & II",
    KeigoClass.PLAIN: "not an honorific form",
}

AUDIENCE_EN: Dict[Audience, str] = {
    Audience.INTERNAL: "internal",
    Audience.EXTERNAL: "external",
    Audience.PUBLIC: "public notice",
}


def error_type_name(et: "ErrorType", lang: str = "en") -> str:
    """誤り種別の表示名を言語ごとに返す。

    実証する主張: 「根拠提示」。英語圏の利用者にも、どの類型の指摘なのかが
    伝わる名前を与える。分類名には日本語の術語をローマ字で残し、原典に
    当たれる状態を保つ。
    """
    return (ERROR_TYPE_EN if lang == "en" else ERROR_TYPE_JA).get(et, et.value)


PARTY_JA: Dict[Party, str] = {
    Party.SELF: "自分",
    Party.SELF_GROUP: "自分側（身内）",
    Party.ADDRESSEE: "相手",
    Party.ADDRESSEE_GROUP: "相手側",
    Party.THIRD_PARTY: "第三者",
    Party.UNKNOWN: "不明",
}

KEIGO_CLASS_JA: Dict[KeigoClass, str] = {
    KeigoClass.SONKEIGO: "尊敬語",
    KeigoClass.KENJOUGO_1: "謙譲語Ⅰ",
    KeigoClass.KENJOUGO_2: "謙譲語Ⅱ（丁重語）",
    KeigoClass.TEINEIGO: "丁寧語",
    KeigoClass.BIKAGO: "美化語",
    KeigoClass.KENJOUGO_1_AND_2: "謙譲語Ⅰ兼Ⅱ",
    KeigoClass.PLAIN: "敬語形でない",
}

AUDIENCE_JA: Dict[Audience, str] = {
    Audience.INTERNAL: "社内",
    Audience.EXTERNAL: "社外",
    Audience.PUBLIC: "不特定多数",
}


def to_jsonable(obj: Any) -> Any:
    """dataclass / Enum を JSON 化できる素の構造に落とす。

    実証する主張: 「根拠提示」。CLI の JSON 出力とデータセットが同じ構造を共有し、指摘の根拠を機械可読な形で外へ出せる。
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    return obj


def dumps(obj: Any, **kwargs: Any) -> str:
    """JSON 文字列にする。

    実証する主張: 「根拠提示」。CLI の ``--format json`` が指摘・修正候補・根拠を
    そのまま機械可読な形で外に出せる。
    """
    return json.dumps(to_jsonable(obj), ensure_ascii=False, **kwargs)
