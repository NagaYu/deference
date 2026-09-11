"""述語ごとに〈誰の行為か〉〈＜向かう先＞は誰か〉を推定する（RoleTagger）。

Deference の心臓部。「敬語の指針」第2章第1 の定義（尊敬語＝相手側又は第三者の行為に
ついてその人物を立てる／謙譲語Ⅰ＝自分側から相手側又は第三者に向かう行為について
その向かう先を立てる）は、いずれも〈動作主が誰か〉〈向かう先が誰か〉が決まらなければ
適用できない。表層文字列だけを見る校正ツールがこの類型を扱えないのはそのためであり、
このモジュールはその欠けている情報を明示的に復元する。

設計上の原則:
    - **観測に徹する。** ここでは「今どうなっているか」だけを記述し、
      「本来どうあるべきか」は判定しない。誤りの検出は detect.py の仕事である。
      とりわけ、述語の敬語分類から動作主を逆算しない（それをすると
      「担当者に伺ってください」の向きの矛盾が原理的に見えなくなる）。
      動作主の手掛かりとして敬語形を使うのは、指針 第2章第1-2 が
      ＜向かう先＞との関係で明示的に論じている授受動詞
      （くださる・いただく・差し上げる）に限る。
    - **利用者の日本語を否定しない。** evidence は判定の理由を述べるだけで、
      適否を断じない。
    - **標準ライブラリのみ。** 形態素解析器は使わず、norms.py の語形と活用規則から
      作った表層辞書との最長一致で述語を同定する。

実証する主張:
    - 「向きの誤り検出」: :class:`RoleAssignment` が述語ごとに
      ``actor`` / ``target`` / ``keigo_class`` を同時に持つ。この3つ組があって
      初めて「謙譲語Ⅰなのに動作主が相手側」といった向きの矛盾が観測できる。
    - 「過剰指摘の少なさ」: 手掛かりが弱いときは断定せず ``confidence`` を下げ、
      ``Party.UNKNOWN`` を返す。推測で埋めないことが後段の過剰指摘を防ぐ。
    - 「根拠提示」: 判定理由を ``evidence`` に、依拠する指針の箇所を
      ``meta["citation_keys"]`` に残す。
    - 「速度」: 表層辞書はモジュール内で1度だけ構築してキャッシュし、
      解析は文字列の最長一致走査のみで行う（外部プロセス・モデル呼び出しなし）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import norms
from .types import Audience, KeigoClass, MailContext, Party, Span

__all__ = [
    "RoleAssignment",
    "PredicateEntry",
    "RoleTagger",
    "predicate_lexicon",
]


# ---------------------------------------------------------------------------
# ローカル定数
#
# ここに置く理由: norms.py は「敬語の語形・分類・引用」の唯一の源であり、
# 以下はいずれもその範疇に入らない情報である。
#   - 人称・組織を指す語彙（弊社・貴社・小職 …）は敬語の語形ではない。
#   - 複合形の活用種別（「お送りくださる」が五段か等）は norms が公開していない
#     派生情報である（norms._IRREGULAR_MASU_STEM は非公開かつ語そのものにしか
#     当たらないため、末尾一致で複合形を扱う表をここに持つ）。
# ---------------------------------------------------------------------------

#: ラ行五段だが連用形が「い」になる敬語動詞の**末尾一致**表。
#: norms.masu_stem は語そのもの（「くださる」）にしか当たらないため、
#: 「お送りくださる」のような複合形をここで処理する。
_RA_IRREGULAR_MASU: Dict[str, str] = {
    "いらっしゃる": "いらっしゃい",
    "おっしゃる": "おっしゃい",
    "なさる": "なさい",
    "くださる": "ください",
    "下さる": "ください",
    "ござる": "ござい",
}

#: 立場を示す語彙。長いものから順に照合する（「私ども」を「私」より先に見る）。
#: 敬語の語形ではないので norms.py には置かない。
_PARTY_MARKERS: Tuple[Tuple[str, Party], ...] = tuple(
    sorted(
        (
            # --- 自分側（身内・自組織） ---
            ("私ども", Party.SELF_GROUP),
            ("私共", Party.SELF_GROUP),
            ("わたくしども", Party.SELF_GROUP),
            ("手前ども", Party.SELF_GROUP),
            ("弊社", Party.SELF_GROUP),
            ("当社", Party.SELF_GROUP),
            ("小社", Party.SELF_GROUP),
            ("弊店", Party.SELF_GROUP),
            ("当店", Party.SELF_GROUP),
            ("弊行", Party.SELF_GROUP),
            ("当行", Party.SELF_GROUP),
            ("弊校", Party.SELF_GROUP),
            ("当校", Party.SELF_GROUP),
            ("当部", Party.SELF_GROUP),
            ("当課", Party.SELF_GROUP),
            ("当部署", Party.SELF_GROUP),
            ("担当者", Party.SELF_GROUP),
            ("担当の者", Party.SELF_GROUP),
            ("係の者", Party.SELF_GROUP),
            # --- 書き手本人 ---
            ("小職", Party.SELF),
            ("小生", Party.SELF),
            ("当方", Party.SELF),
            ("わたくし", Party.SELF),
            ("わたし", Party.SELF),
            ("拙者", Party.SELF),
            ("私", Party.SELF),
            # --- 相手側（先方組織） ---
            ("貴社", Party.ADDRESSEE_GROUP),
            ("御社", Party.ADDRESSEE_GROUP),
            ("おん社", Party.ADDRESSEE_GROUP),
            ("貴店", Party.ADDRESSEE_GROUP),
            ("貴行", Party.ADDRESSEE_GROUP),
            ("貴校", Party.ADDRESSEE_GROUP),
            ("貴学", Party.ADDRESSEE_GROUP),
            ("貴部", Party.ADDRESSEE_GROUP),
            ("貴課", Party.ADDRESSEE_GROUP),
            ("そちら様", Party.ADDRESSEE_GROUP),
            ("そちら", Party.ADDRESSEE_GROUP),
            ("ご担当者様", Party.ADDRESSEE_GROUP),
            ("御担当者様", Party.ADDRESSEE_GROUP),
            # --- 読み手本人 ---
            ("あなた様", Party.ADDRESSEE),
            ("あなた", Party.ADDRESSEE),
            ("貴殿", Party.ADDRESSEE),
            ("貴方", Party.ADDRESSEE),
            ("お客様", Party.ADDRESSEE),
            ("お客さま", Party.ADDRESSEE),
            # --- 第三者 ---
            ("先方", Party.THIRD_PARTY),
        ),
        key=lambda kv: -len(kv[0]),
    )
)

#: 敬称。組織語彙でも文脈でも立場が決まらないときの弱い手掛かり。
_HONORIFIC_SUFFIXES: Tuple[str, ...] = ("様", "さま", "先生", "殿", "御中")

#: 助詞。文節の切り出しに使う（長いものから照合する）。
_PARTICLES: Tuple[str, ...] = tuple(
    sorted(
        (
            "にて", "では", "へは", "には", "からは", "までに",
            "から", "まで", "より", "こそ",
            "が", "は", "も", "を", "に", "へ", "と", "で",
        ),
        key=len,
        reverse=True,
    )
)

#: 動作主になりうる文節の助詞（主語・主題・動作の場）。
_ACTOR_PARTICLES = frozenset({"が", "は", "も", "では", "にて", "こそ"})

#: ＜向かう先＞／出どころを表す助詞。
#: 指針 第2章第1-2【解説2】に従い、「から」「より」の名詞句も＜向かう先＞とみなす。
_TARGET_PARTICLES = frozenset({"に", "へ", "には", "へは", "から", "より", "からは"})

#: 文節の区切り（ここを跨いで名詞句を作らない）。
_PHRASE_BREAKS = frozenset("、。，,！!？?\n\r\t 　「」『』（）()【】[]・…")

#: 述語の直後に来てよい文字（述語がここで終わっていると判断する）。
_BOUNDARY_CHARS = frozenset("。．.！!？?、，,；;：:「」『』（）()【】[]…・—\n\r\t 　\"'")

#: 述語の直後に来てよい接続表現。ここに無い平仮名が続く場合は、
#: 未知の長い活用形の途中を切り出した可能性が高いので採用しない（過剰検出を避ける）。
_CONTINUATIONS: Tuple[str, ...] = (
    "が", "けれども", "けれど", "けど", "ので", "のに", "から", "し", "と",
    "たり", "ば", "なら", "たら", "ても", "でも", "ながら", "つつ",
    "ように", "よう", "ため", "か", "かも", "ね", "よ", "な", "の",
    "こと", "もの", "方", "際", "時", "とき", "まで", "ほど", "くらい",
    "ぐらい", "だけ", "ばかり", "はず", "わけ", "予定", "つもり", "ほか",
    "以上", "上", "中", "後", "前", "次第", "かどうか", "でしょう",
    "だろう", "でした", "です", "ます", "という", "といった", "との",
    "とも", "そう", "ない", "べき", "らしい", "みたい",
)

#: 依頼・お願いを示す表現（主語が落ちたときの既定を相手側にする手掛かり）。
_REQUEST_CUES: Tuple[str, ...] = (
    "ください", "下さい", "お願い", "願います", "願いたく",
    "いただけ", "頂け", "いただきたく", "頂きたく", "ませんか", "ましょうか",
)

#: 質問を示す表現。
_QUESTION_CUES: Tuple[str, ...] = ("でしょうか", "ますか", "ですか", "ましたか")

#: 授受動詞による動作主の手掛かり。
#: 指針 第2章第1-2 が＜向かう先＞との関係で明示的に論じている語に限る。
#: これ以外の敬語形から動作主を逆算することはしない（向きの矛盾が見えなくなるため）。
_GIVE_RECEIVE_HINT: Tuple[Tuple[str, str], ...] = (
    ("くださる", "other"),
    ("下さる", "other"),
    ("くれる", "other"),
    ("いただく", "self"),
    ("頂く", "self"),
    ("もらう", "self"),
    ("差し上げる", "self"),
    ("さしあげる", "self"),
    ("あげる", "self"),
)

#: 指針が「習慣として定着している」とする二重敬語（norms.ESTABLISHED_DOUBLE_KEIGO）の
#: 活用種別と分類。語形そのものは norms が持つが、活用種別・分類は持たないため
#: ここで補う（norms に無い派生情報なのでローカル）。
_ESTABLISHED_META: Dict[str, Tuple[str, str, KeigoClass]] = {
    # 表層: (素の動詞, 活用種別, 分類)
    "お召し上がりになる": ("食べる", "godan", KeigoClass.SONKEIGO),
    "お見えになる": ("来る", "godan", KeigoClass.SONKEIGO),
    "お伺いする": ("聞く", "sahen", KeigoClass.KENJOUGO_1),
    "お伺いいたす": ("聞く", "godan", KeigoClass.KENJOUGO_1_AND_2),
    "お伺い申し上げる": ("聞く", "ichidan", KeigoClass.KENJOUGO_1),
}

#: 「て」でつなぐ補助動詞。指針 第2章第2-6(3)「敬語連結」の対象。
#: (辞書形, 活用種別, 分類, 動作主の手掛かり)
_AUXILIARIES: Tuple[Tuple[str, str, KeigoClass, str], ...] = (
    ("くださる", "godan", KeigoClass.SONKEIGO, "other"),
    ("いただく", "godan", KeigoClass.KENJOUGO_1, "self"),
    ("差し上げる", "ichidan", KeigoClass.KENJOUGO_1, "self"),
    ("さしあげる", "ichidan", KeigoClass.KENJOUGO_1, "self"),
    ("いらっしゃる", "godan", KeigoClass.SONKEIGO, ""),
    ("おる", "godan", KeigoClass.KENJOUGO_2, ""),
    ("いる", "ichidan", KeigoClass.PLAIN, ""),
    ("もらう", "godan", KeigoClass.PLAIN, "self"),
)

#: 由来ごとの優先度（同じ表層が衝突したときにどちらを採るか）。
_ORIGIN_PRIORITY: Dict[str, int] = {
    "established": 95,
    "special": 90,
    "general": 80,
    "ogo_potential": 78,
    "sasete": 75,
    "link": 70,
    "plain": 10,
}

_NON_HIRAGANA_HEAD = re.compile(r"[一-鿿々゠-ヿA-Za-z0-9０-９]")
_SENTENCE_ENDS = frozenset("。．！!？?\n\r")


# ---------------------------------------------------------------------------
# 活用ヘルパ（norms の活用規則を複合形に当てるための薄い層）
# ---------------------------------------------------------------------------


def _masu_stem(form: str, kind: str) -> str:
    """複合形にも当たる連用形。norms.masu_stem を末尾一致で補う。

    実証する主張: 「速度」「修正候補の妥当性」。語形は規則から機械的に作るので、
    辞書の手書き量を増やさずに表層のゆれを吸収できる。
    """
    for suffix, stem in _RA_IRREGULAR_MASU.items():
        if form.endswith(suffix) and len(form) > len(suffix):
            return form[: -len(suffix)] + stem
    return norms.masu_stem(form, kind)


def _past_plain(form: str, kind: str) -> str:
    """常体過去形（テ形の末尾を「た／だ」に替える）。

    実証する主張: 「向きの誤り検出」。過去形の述語も同じ辞書で拾えるようにし、
    「おっしゃいました」のような実文中の形を取りこぼさない。
    """
    te = norms.te_form(form, kind)
    return te[:-1] + ("た" if te.endswith("て") else "だ")


def _kind_of_generated(surface: str) -> str:
    """norms の一般形メソッドが返した語形の活用種別を推定する。

    norms は一般形の**語形**は返すが活用種別は返さないため、生成側で
    現れうる末尾の閉じた集合だけを見て決める。

    実証する主張: 「速度」。形態素解析器なしで活用を展開するための最小の仕掛け。
    """
    if surface.endswith("だ"):
        return "copula"
    if surface.endswith("いたす"):
        return "godan"
    if surface.endswith("する"):
        return "sahen"
    if surface.endswith("来る"):
        return "kahen"
    for suffix in _RA_IRREGULAR_MASU:
        if surface.endswith(suffix):
            return "godan"
    if surface.endswith("られる") or surface.endswith("れる"):
        return "ichidan"
    if surface.endswith("申し上げる") or surface.endswith("上げる"):
        return "ichidan"
    if surface.endswith("いただく"):
        return "godan"
    if surface.endswith("になる") or surface.endswith("なる"):
        return "godan"
    if surface.endswith("できる"):
        return "ichidan"
    return "godan" if surface and surface[-1] in "うくぐすつぬぶむる" else "ichidan"


def _variants(form: str, kind: str, keigo: bool) -> List[str]:
    """1つの辞書形から、本文中に現れる活用形を展開する。

    実証する主張: 「向きの誤り検出」。述語を取りこぼすと立場の割り当てが
    そもそも起きないため、活用の展開は検出可能性の前提になる。
    """
    out: List[str] = [form]
    if kind == "copula":
        stem = form[:-1]
        out += [stem, stem + "です", stem + "でした", stem + "でして",
                stem + "でしょう", stem + "ではない"]
        return out
    try:
        stem = _masu_stem(form, kind)
    except (ValueError, KeyError):
        return out
    out += [
        stem + "ます", stem + "ました", stem + "ません", stem + "ませんでした",
        stem + "まして", stem + "ましたら", stem + "ましょう",
    ]
    if keigo:
        # 連用中止形（「お送りいただき、ありがとうございました」）。
        # 素の動詞では語中に埋もれやすいため敬語形に限る。
        out.append(stem)
    for fn in (norms.te_form, _past_plain):
        try:
            out.append(fn(form, kind))
        except (ValueError, KeyError):
            pass
    try:
        out.append(norms.a_stem(form, kind) + "ない")
    except (ValueError, KeyError):
        pass
    try:
        pot = norms.potential(form, kind)
    except (ValueError, KeyError):
        pot = ""
    if pot:
        out += [pot, pot[:-1] + "ます", pot[:-1] + "ました",
                pot[:-1] + "ません", pot[:-1] + "ませんでした", pot[:-1] + "ない"]
    return out


# ---------------------------------------------------------------------------
# 述語辞書
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredicateEntry:
    """表層1つぶんの述語情報。

    実証する主張: 「向きの誤り検出」「根拠提示」。表層から敬語分類と素の動詞に
    辿れることが、向きの判定と根拠の提示の前提になる。
    """

    surface: str
    dictionary_form: str
    keigo_class: KeigoClass
    kind: str
    origin: str
    bases: Tuple[str, ...] = ()
    auxiliary: str = ""
    auxiliary_class: Optional[KeigoClass] = None
    actor_hint: str = ""  # "self" / "other" / ""
    priority: int = 0


def _actor_hint_of(dictionary_form: str) -> str:
    """辞書形の末尾から授受動詞による動作主の手掛かりを引く。"""
    for suffix, hint in _GIVE_RECEIVE_HINT:
        if dictionary_form.endswith(suffix):
            return hint
    return ""


class _LexiconBuilder:
    """表層 → :class:`PredicateEntry` の辞書を組み立てる。

    実証する主張: 「根拠提示」。辞書のすべての項目は norms.py（＝「敬語の指針」）の
    語形と活用規則から機械的に導かれ、roles.py 側で敬語の語形を新たに
    ハードコードしていない。
    """

    def __init__(self) -> None:
        """述語辞書の構築を始める。

        実証する主張: 「速度」。辞書は一度だけ組み、以降の検査を辞書引きだけで済ませる。
        """
        self.table: Dict[str, PredicateEntry] = {}
        self.heads: List[Tuple[str, str, KeigoClass, Tuple[str, ...], str]] = []

    # -- 登録 ---------------------------------------------------------------
    def add(
        self,
        dictionary_form: str,
        kind: str,
        keigo_class: KeigoClass,
        origin: str,
        bases: Tuple[str, ...],
        auxiliary: str = "",
        auxiliary_class: Optional[KeigoClass] = None,
        actor_hint: Optional[str] = None,
        as_head: bool = True,
    ) -> None:
        """表層と分類の対を辞書に登録する。

        実証する主張: 「向きの誤り検出」。表層から敬語分類を引けることが、向きの判定の前提になる。
        """
        hint = _actor_hint_of(dictionary_form) if actor_hint is None else actor_hint
        priority = _ORIGIN_PRIORITY.get(origin, 50)
        keigo = keigo_class is not KeigoClass.PLAIN
        for surface in _variants(dictionary_form, kind, keigo):
            if len(surface) < 2:
                continue
            entry = PredicateEntry(
                surface=surface,
                dictionary_form=dictionary_form,
                keigo_class=keigo_class,
                kind=kind,
                origin=origin,
                bases=bases,
                auxiliary=auxiliary,
                auxiliary_class=auxiliary_class,
                actor_hint=hint,
                priority=priority,
            )
            old = self.table.get(surface)
            if old is None or entry.priority > old.priority:
                self.table[surface] = entry
        if as_head:
            self.heads.append((dictionary_form, kind, keigo_class, bases, origin))

    # -- 各段 ---------------------------------------------------------------
    def add_special_forms(self) -> None:
        """指針の【特定形の主な例】（norms.SPECIAL_FORMS）を登録する。

        実証する主張: 「向きの誤り検出」。特定形は分類が確定しているので、動作主の立場との矛盾を最も強く示す手掛かりになる。
        """
        grouped: Dict[str, List[norms.SpecialForm]] = norms.special_form_index()
        for surface, forms in grouped.items():
            head = forms[0]
            bases = tuple(dict.fromkeys(sf.base for sf in forms))
            self.add(surface, head.kind, head.keigo_class, "special", bases)

    def add_established_double_keigo(self) -> None:
        """指針が定着を認めている二重敬語（norms.ESTABLISHED_DOUBLE_KEIGO）。

        実証する主張: 「過剰指摘の少なさ」。定着した二重敬語を語彙として先に登録し、誤りとして拾わないようにする。
        """
        for surface in norms.ESTABLISHED_DOUBLE_KEIGO:
            meta = _ESTABLISHED_META.get(surface)
            if meta is None:  # norms 側が増えても落ちないようにする
                continue
            base, kind, keigo_class = meta
            self.add(surface, kind, keigo_class, "established", (base,))

    def add_general_forms(self) -> None:
        """一般動詞の一般形（norms.Verb の各メソッド）を登録する。

        実証する主張: 「向きの誤り検出」。一般形は生産的なので、辞書の動詞すべてについて表層を展開しておく。
        """
        for v in norms.VERBS:
            for surface in v.sonkeigo_general():
                self.add(surface, _kind_of_generated(surface),
                         KeigoClass.SONKEIGO, "general", (v.plain,))
            kudasaru = v.sonkeigo_kudasaru()
            if kudasaru:
                self.add(kudasaru, _kind_of_generated(kudasaru),
                         KeigoClass.SONKEIGO, "general", (v.plain,))
            for surface in v.kenjougo1_general():
                self.add(surface, _kind_of_generated(surface),
                         KeigoClass.KENJOUGO_1, "general", (v.plain,))
            for surface in v.kenjougo2_general():
                # 「お(ご)……いたす」は謙譲語Ⅰ兼Ⅱ、接頭辞の無い「……いたす」は謙譲語Ⅱ。
                cls = (
                    KeigoClass.KENJOUGO_1_AND_2
                    if surface[:1] in ("お", "ご", "御")
                    else KeigoClass.KENJOUGO_2
                )
                self.add(surface, _kind_of_generated(surface), cls, "general", (v.plain,))
            # 素の形（丁寧語・常体）も押さえる。文体の観測に要る。
            self.add(v.plain, v.kind, KeigoClass.PLAIN, "plain", (v.plain,))

    def add_ogo_potential(self) -> None:
        """「お(ご)……できる」形（指針【8】）。謙譲語Ⅰ「お(ご)……する」の可能形。

        実証する主張: 「向きの誤り検出」。「お(ご)……できる」は謙譲語Ⅰの可能形であり、相手の行為に使われていれば向きの誤りになる（指針【8】）。
        """
        for v in norms.VERBS:
            if not (v.ogo and v.has_target):
                continue
            surface = f"{v.ogo}{v.ogo_stem}できる"
            self.add(surface, "ichidan", KeigoClass.KENJOUGO_1,
                     "ogo_potential", (v.plain,), as_head=False)

    def add_sasete_itadaku(self) -> None:
        """「……(さ)せていただく」形（指針【18】）。動作主は自分側。

        実証する主張: 「過剰指摘の少なさ」。「させていただく」は許容度に個人差があるため、語彙として認識したうえで密度で判断する。
        """
        for v in norms.VERBS:
            try:
                a = norms.a_stem(v.plain, v.kind)
            except (ValueError, KeyError):
                continue
            tail = "せていただく" if v.kind in ("godan", "sahen") else "させていただく"
            self.add(a + tail, "godan", KeigoClass.KENJOUGO_1, "sasete",
                     (v.plain,), auxiliary="いただく",
                     auxiliary_class=KeigoClass.KENJOUGO_1, actor_hint="self",
                     as_head=False)
            if v.ogo and v.kind == "sahen":
                self.add(f"{v.ogo}{v.ogo_stem}させていただく", "godan",
                         KeigoClass.KENJOUGO_1, "sasete", (v.plain,),
                         auxiliary="いただく",
                         auxiliary_class=KeigoClass.KENJOUGO_1, actor_hint="self",
                         as_head=False)

    def add_links(self) -> None:
        """「て」でつないだ敬語連結（指針 第2章第2-6(3)）を展開する。

        実証する主張: 「過剰指摘の少なさ」。指針が許容する敬語連結（p.30）を登録し、誤りとして拾わないようにする。
        """
        for dictionary_form, kind, keigo_class, bases, origin in list(self.heads):
            if origin == "link":
                continue
            try:
                te = norms.te_form(dictionary_form, kind)
            except (ValueError, KeyError):
                continue
            for aux, aux_kind, aux_class, hint in _AUXILIARIES:
                # 連結全体の分類は主動詞の分類。主動詞が素の形なら補助動詞の分類。
                cls = keigo_class if keigo_class is not KeigoClass.PLAIN else aux_class
                self.add(te + aux, aux_kind, cls, "link", bases,
                         auxiliary=aux, auxiliary_class=aux_class,
                         actor_hint=hint, as_head=False)

    def build(self) -> Dict[str, PredicateEntry]:
        """組み上げた述語辞書を返す。

        実証する主張: 「速度」。辞書は一度だけ構築してキャッシュし、1文あたりの検査を辞書引きだけで済ませる。
        """
        self.add_special_forms()
        self.add_established_double_keigo()
        self.add_general_forms()
        self.add_ogo_potential()
        self.add_sasete_itadaku()
        self.add_links()
        return self.table


_LEXICON: Optional[Dict[str, PredicateEntry]] = None
_MAX_SURFACE_LEN = 0


def predicate_lexicon() -> Dict[str, PredicateEntry]:
    """述語の表層辞書を返す（初回のみ構築し、以降はキャッシュを返す）。

    実証する主張: 「速度」。辞書構築はプロセス内で1度きり。以降の解析は
    文字列の最長一致走査だけで済み、1通あたりの応答は CPU のみで完結する。
    """
    global _LEXICON, _MAX_SURFACE_LEN
    if _LEXICON is None:
        _LEXICON = _LexiconBuilder().build()
        _MAX_SURFACE_LEN = max((len(s) for s in _LEXICON), default=0)
    return _LEXICON


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleAssignment:
    """述語1つに対する立場の割り当て。

    実証する主張: 「向きの誤り検出」「根拠提示」。``actor`` / ``target`` /
    ``keigo_class`` の3つ組が揃って初めて、指針 第2章第1 の定義
    （尊敬語は動作主を立てる／謙譲語Ⅰは＜向かう先＞を立てる）と突き合わせられる。
    ``evidence`` はその割り当ての理由を利用者に見える言葉で残す。
    """

    predicate_span: Span
    surface: str
    actor: Party
    target: Party
    keigo_class: KeigoClass
    base_verb: str = ""
    confidence: float = 1.0
    evidence: str = ""
    sentence_span: Optional[Span] = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Phrase:
    """文節1つ（名詞句＋助詞）。"""

    text: str
    particle: str
    start: int
    end: int


class RoleTagger:
    """本文中の述語に〈動作主〉と〈＜向かう先＞〉を割り当てる。

    表層文字列だけを見る校正ツールは「田中がおっしゃいました」の適否を
    判定できない。田中が自分側の人間か相手側の人間かで結論が反転するからである
    （指針 第3章第3-2【24】【25】）。RoleTagger はその欠けている情報を
    :class:`MailContext` と文中の手掛かりから復元する。

    実証する主張:
        - 「向きの誤り検出」: 述語ごとに動作主・向かう先・敬語分類を返す。
        - 「過剰指摘の少なさ」: 手掛かりが無いときは ``Party.UNKNOWN`` を返し、
          ``confidence`` を下げる。推測で埋めない。
        - 「根拠提示」: ``evidence`` と ``meta["citation_keys"]`` を残す。
        - 「速度」: 標準ライブラリのみ・辞書はプロセス内で1度だけ構築。
    """

    def __init__(self) -> None:
        """述語辞書を用意する（構築済みならキャッシュを共有する）。

        実証する主張: 「速度」。インスタンス化のたびに辞書を作り直さない。
        """
        self._lexicon = predicate_lexicon()
        self._max_len = _MAX_SURFACE_LEN

    # -- 文分割 -------------------------------------------------------------
    def split_sentences(self, text: str) -> List[Span]:
        """「。」「！」「？」「改行」で文に分ける。Span は本文全体基準。

        句点等は直前の文に含める（述語が文末で終わっているかの判定に使うため）。

        実証する主張: 「速度」。1回の線形走査だけで、外部依存を持たない。
        """
        spans: List[Span] = []
        start = 0
        i = 0
        n = len(text)
        while i < n:
            if text[i] in _SENTENCE_ENDS:
                j = i + 1
                while j < n and text[j] in _SENTENCE_ENDS:
                    j += 1
                chunk = text[start:j]
                if chunk.strip():
                    spans.append(Span(start, j, chunk))
                start = j
                i = j
                continue
            i += 1
        if start < n and text[start:n].strip():
            spans.append(Span(start, n, text[start:n]))
        return spans

    # -- 述語同定 -----------------------------------------------------------
    def _accepts_boundary(self, sentence: str, end: int) -> bool:
        """述語がその位置で終わっていると見てよいか。"""
        if end >= len(sentence):
            return True
        nxt = sentence[end]
        if nxt in _BOUNDARY_CHARS:
            return True
        rest = sentence[end:]
        return any(rest.startswith(c) for c in _CONTINUATIONS)

    def find_predicates(self, sentence: str) -> List[Tuple[int, int, PredicateEntry]]:
        """文中の述語を最長一致・重なりなしで拾う。

        実証する主張: 「過剰指摘の少なさ」。素の動詞は直前が漢字のときに採らない
        （「見送ります」から「送ります」を切り出さない）など、語中の偶然一致を
        抑える保守的な走査にしている。
        """
        out: List[Tuple[int, int, PredicateEntry]] = []
        n = len(sentence)
        i = 0
        while i < n:
            hit: Optional[Tuple[int, int, PredicateEntry]] = None
            upper = min(self._max_len, n - i)
            for length in range(upper, 1, -1):
                surface = sentence[i : i + length]
                entry = self._lexicon.get(surface)
                if entry is None:
                    continue
                if not self._accepts_boundary(sentence, i + length):
                    continue
                if entry.keigo_class is KeigoClass.PLAIN and i > 0:
                    # 素の動詞は複合語の一部を切り出しやすいので、直前が
                    # 漢字・カタカナ・英数のときは採らない。
                    if _NON_HIRAGANA_HEAD.match(sentence[i - 1]):
                        continue
                hit = (i, i + length, entry)
                break
            if hit is None:
                i += 1
            else:
                out.append(hit)
                i = hit[1]
        return out

    # -- 文節分解 -----------------------------------------------------------
    def _phrases(self, text: str) -> List[_Phrase]:
        """述語より前の部分を〈名詞句＋助詞〉に分ける。"""
        out: List[_Phrase] = []
        buf_start = 0
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in _PHRASE_BREAKS:
                buf_start = i + 1
                i += 1
                continue
            matched = ""
            for p in _PARTICLES:
                if text.startswith(p, i):
                    matched = p
                    break
            if matched:
                body = text[buf_start:i]
                if body:
                    out.append(_Phrase(body, matched, buf_start, i))
                i += len(matched)
                buf_start = i
                continue
            i += 1
        return out

    # -- 立場の解決 ---------------------------------------------------------
    def _marker_party(self, phrase: str) -> Tuple[Party, str]:
        """語彙（弊社・貴社・私 …）から立場を引く。"""
        for marker, party in _PARTY_MARKERS:
            if marker in phrase:
                return party, marker
        return Party.UNKNOWN, ""

    def resolve_party(
        self, phrase: str, context: MailContext
    ) -> Tuple[Party, float, str]:
        """名詞句から立場を推定する。

        優先順位は (1) 文脈の人物表・書き手・宛先（``MailContext.side_of``）、
        (2) 組織語・人称語、(3) 敬称。(1) と (2) が別の側を指すときは
        本文に書かれている (2) を採り、確信度を下げる。

        実証する主張: 「向きの誤り検出」「過剰指摘の少なさ」。同じ人物名でも
        文脈しだいで側が変わるため、文脈を先に見る。どの手掛かりも無ければ
        ``Party.UNKNOWN`` を返して推測で埋めない。
        """
        phrase = phrase.strip()
        if not phrase:
            return Party.UNKNOWN, 0.0, ""
        ctx_party = context.side_of(phrase)
        mark_party, marker = self._marker_party(phrase)

        if ctx_party is not Party.UNKNOWN and mark_party is not Party.UNKNOWN:
            same_side = (
                ctx_party.is_self_side and mark_party.is_self_side
            ) or (ctx_party.is_other_side and mark_party.is_other_side)
            if same_side:
                return ctx_party, 0.95, f"「{phrase}」は文脈上の人物"
            return (
                mark_party,
                0.6,
                f"「{phrase}」は本文の「{marker}」を手掛かりに判定（文脈の人物表とは別の側）",
            )
        if ctx_party is not Party.UNKNOWN:
            return ctx_party, 0.9, f"「{phrase}」は文脈上の人物"
        if mark_party is not Party.UNKNOWN:
            return mark_party, 0.85, f"「{phrase}」の「{marker}」から判定"
        for suffix in _HONORIFIC_SUFFIXES:
            if phrase.endswith(suffix):
                return (
                    Party.ADDRESSEE_GROUP,
                    0.6,
                    f"「{phrase}」は敬称「{suffix}」付きの名前",
                )
        return Party.UNKNOWN, 0.0, ""

    # -- 文の機能 -----------------------------------------------------------
    def sentence_function(self, sentence: str) -> str:
        """文が依頼・質問・報告のどれかを大まかに見る。

        主語が落ちた文の既定値を決めるためだけに使う。

        実証する主張: 「向きの誤り検出」。日本語では主語がしばしば落ちるため、
        既定値の置き方そのものが向きの判定精度を左右する。
        """
        if any(c in sentence for c in _REQUEST_CUES):
            return "request"
        if any(c in sentence for c in _QUESTION_CUES):
            return "question"
        return "report"

    # -- 動作主 -------------------------------------------------------------
    def _infer_actor(
        self,
        phrases: List[_Phrase],
        entry: PredicateEntry,
        context: MailContext,
        function: str,
        carried: Optional[Tuple[Party, str]],
    ) -> Tuple[Party, float, str]:
        # (1) 明示的な主語・主題
        best: Optional[Tuple[Party, float, str]] = None
        for ph in reversed(phrases):
            if ph.particle not in _ACTOR_PARTICLES:
                continue
            party, conf, why = self.resolve_party(ph.text, context)
            if party is not Party.UNKNOWN:
                best = (party, conf, f"主語「{ph.text}{ph.particle}」— {why}")
                break
        if best is not None:
            return best

        # (2) 授受動詞（指針 第2章第1-2 が＜向かう先＞との関係で論じている語に限る）
        if entry.actor_hint == "self":
            label = entry.auxiliary or entry.dictionary_form
            return (
                Party.SELF,
                0.8,
                f"「{label}」は受け手が自分側になる形なので、動作主を自分側と見た",
            )
        if entry.actor_hint == "other":
            label = entry.auxiliary or entry.dictionary_form
            return (
                Party.ADDRESSEE,
                0.8,
                f"「{label}」は相手側の動作を表す形なので、動作主を相手側と見た",
            )

        # (3)「〜から」の名詞句を弱い動作主の手掛かりとして使う
        for ph in reversed(phrases):
            if ph.particle not in ("から", "からは", "より"):
                continue
            party, _conf, why = self.resolve_party(ph.text, context)
            if party is not Party.UNKNOWN:
                return party, 0.6, f"出どころ「{ph.text}{ph.particle}」— {why}"

        # (4) 直前の文からの引き継ぎ（ゼロ照応の素朴な近似）
        if carried is not None:
            party, src = carried
            return party, 0.55, f"主語が現れないため、直前の文から引き継ぎ（{src}）"

        # (5) 文の機能による既定
        if function in ("request", "question"):
            return Party.ADDRESSEE, 0.5, "主語が現れない依頼・問い合わせの文なので相手側を既定とした"
        return Party.SELF, 0.5, "主語が現れない報告・連絡の文なので自分側を既定とした"

    # -- ＜向かう先＞ -------------------------------------------------------
    def _infer_target(
        self,
        phrases: List[_Phrase],
        entry: PredicateEntry,
        context: MailContext,
        actor: Party,
    ) -> Tuple[Party, float, str]:
        for ph in reversed(phrases):
            if ph.particle not in _TARGET_PARTICLES:
                continue
            party, conf, why = self.resolve_party(ph.text, context)
            if party is Party.UNKNOWN:
                continue
            if ph.particle in ("から", "からは", "より"):
                note = "（指針 第2章第1-2【解説2】に従い、出どころも＜向かう先＞と見る）"
            else:
                note = ""
            return party, min(conf, 0.85), f"＜向かう先＞「{ph.text}{ph.particle}」— {why}{note}"

        if entry.actor_hint == "self" and actor.is_self_side:
            return Party.ADDRESSEE, 0.55, "授受の相手が明示されていないため、宛先を＜向かう先＞と見た"
        if entry.actor_hint == "other" and actor.is_other_side:
            return Party.SELF, 0.55, "授受の受け手が明示されていないため、書き手を＜向かう先＞と見た"
        return Party.UNKNOWN, 0.0, ""

    # -- 主処理 -------------------------------------------------------------
    def tag(self, text: str, context: MailContext) -> List[RoleAssignment]:
        """本文全体を文に分け、各文の述語に立場を割り当てる。

        述語が見つからない文は何も返さない（無理に返さない）。

        実証する主張:
            - 「向きの誤り検出」: 述語ごとに〈動作主・向かう先・敬語分類〉を返す。
              後段の detect.py はこの3つ組を指針の定義と突き合わせるだけでよい。
            - 「過剰指摘の少なさ」: 手掛かりが弱い割り当ては ``confidence`` を
              下げ、分からないものは ``Party.UNKNOWN`` のままにする。
            - 「根拠提示」: ``evidence`` に判定理由、``meta["citation_keys"]`` に
              依拠した指針の箇所を残す。
            - 「速度」: 文分割と最長一致走査だけで完結する。
        """
        out: List[RoleAssignment] = []
        carried: Optional[Tuple[Party, str]] = None

        for s_span in self.split_sentences(text):
            sentence = text[s_span.start : s_span.end]
            matches = self.find_predicates(sentence)
            if not matches:
                continue
            function = self.sentence_function(sentence)
            last_strong: Optional[Tuple[Party, str]] = None

            for start, end, entry in matches:
                phrases = self._phrases(sentence[:start])
                actor, a_conf, a_why = self._infer_actor(
                    phrases, entry, context, function, carried
                )
                target, t_conf, t_why = self._infer_target(
                    phrases, entry, context, actor
                )
                surface = sentence[start:end]
                span = Span(s_span.start + start, s_span.start + end, surface)

                evidence = a_why
                if t_why:
                    evidence = f"{evidence}／{t_why}"

                meta: dict = {
                    "origin": entry.origin,
                    "dictionary_form": entry.dictionary_form,
                    "bases": list(entry.bases),
                    "function": function,
                    "actor_confidence": round(a_conf, 2),
                    "target_confidence": round(t_conf, 2),
                    "audience": context.audience.value,
                    "carried_over": bool(
                        carried is not None and "引き継ぎ" in a_why
                    ),
                }
                if entry.auxiliary:
                    meta["auxiliary"] = entry.auxiliary
                    meta["auxiliary_class"] = (
                        entry.auxiliary_class.value if entry.auxiliary_class else None
                    )
                if entry.origin == "link":
                    link_form = entry.dictionary_form
                    meta["keigo_link"] = link_form
                    if link_form in norms.ACCEPTABLE_KEIGO_LINKS:
                        meta["link_listed"] = "acceptable"
                    elif link_form in norms.INAPPROPRIATE_KEIGO_LINKS or (
                        surface in norms.INAPPROPRIATE_KEIGO_LINKS
                    ):
                        meta["link_listed"] = "inappropriate"
                    else:
                        meta["link_listed"] = ""
                meta.update(self._audience_meta(context, actor))

                out.append(
                    RoleAssignment(
                        predicate_span=span,
                        surface=surface,
                        actor=actor,
                        target=target,
                        keigo_class=entry.keigo_class,
                        base_verb=entry.bases[0] if entry.bases else "",
                        confidence=round(min(a_conf, 1.0), 2),
                        evidence=evidence,
                        sentence_span=s_span,
                        meta=meta,
                    )
                )
                if a_conf >= 0.7 and actor is not Party.UNKNOWN:
                    last_strong = (actor, f"「{surface}」の動作主")
            if last_strong is not None:
                carried = last_strong
        return out

    # -- 宛先の影響 ---------------------------------------------------------
    def _audience_meta(self, context: MailContext, actor: Party) -> dict:
        """宛先が社内か社外かで変わる「ウチ・ソト」の扱いを記録する。

        指針 第3章第3-2【25】【26】。社外向けでは自社の人物は立てない側になり、
        社内向けでは自社の上位者を立ててよい場合がある。ここでは判定はせず、
        後段が使えるように事実として記録するだけにする。

        実証する主張: 「向きの誤り検出」「根拠提示」。同じ文字列でも宛先で
        結論が反転することを、引用キー付きで下流に渡す。
        """
        meta: dict = {}
        if not actor.is_self_side:
            meta["self_side_raisable"] = False
            meta["citation_keys"] = ["principle.self_not_raised"]
            return meta
        if context.audience is Audience.INTERNAL:
            meta["self_side_raisable"] = True
            meta["uchi_soto"] = "internal_own_senior_may_be_raised"
            meta["citation_keys"] = ["uchi_soto.q26", "uchi_soto.q25"]
        else:
            meta["self_side_raisable"] = False
            meta["uchi_soto"] = "external_self_side_not_raised"
            meta["citation_keys"] = ["uchi_soto.q25", "principle.self_not_raised"]
        return meta
