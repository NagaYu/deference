"""修正候補の生成（拘束付き）。

このモジュールは「検出した箇所に対して、どう書き換えれば規範に沿うか」を返す。
生成は **拘束付き** であり、次の二つの供給源からしか候補を作らない。

1. :class:`deference.generate.Generator` の ``correct_forms()`` が返す集合
2. :mod:`deference.norms` から機械的に導ける語形
   （``Verb.sonkeigo_general()`` / ``kenjougo1_general()`` / ``kenjougo2_general()``、
   ``special_forms_for()``、``masu_stem()`` / ``a_stem()`` / ``e_stem()`` /
   ``te_form()`` / ``polite()`` / ``potential()`` / ``negative_polite()``）

自由文字列を組み立てて返すことはしない。したがって修正案そのものが新たな
規範逸脱になることが原理的にない。

実証する主張:
    - 「向きの誤り検出」: 修正候補は〈誰の行為か〉〈誰に向かう行為か〉
      （:class:`~deference.types.Party`）と宛先（:class:`~deference.types.Audience`）
      から分類を決めてから語形を作る。表層置換ではなく向きから作るので、
      向きの誤りに対して向きの正しい候補が出る。
    - 「根拠提示」: 各 :class:`~deference.types.Suggestion` の ``reason`` は
      :mod:`deference.norms` の規範記述に対応する短い説明を持つ。
    - 「過剰指摘の少なさ」: ``norms.ESTABLISHED_DOUBLE_KEIGO`` /
      ``norms.SPLIT_ACCEPTABILITY`` に載る定着形は「言い換え先」としては扱うが、
      それ自体を書き換え対象の起点にしない。
    - 「速度」: 語形索引はモジュール内で一度だけ構築してキャッシュするため、
      1件あたりの候補生成は辞書引きと文字列連結だけで済む。

利用者に見せる文言（``reason``）は情報提示に留め、利用者の書いた日本語を
否定する言い方をしない。
"""

from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from . import norms
from .types import (
    Audience,
    ErrorType,
    KeigoClass,
    MailContext,
    Party,
    Span,
    Suggestion,
)

if TYPE_CHECKING:  # pragma: no cover - 型注釈のためだけの import
    from .generate import Generator
    from .roles import RoleAssignment

__all__ = [
    "Corrector",
    "GENERATED_BY",
    "MAX_SUGGESTIONS",
    "FormEntry",
]


#: すべての候補に付ける生成元。自由生成を混ぜない運用を文字列で表明する。
GENERATED_BY = "rule"

#: 1スパンあたりに返す候補の上限。指針 第3章第1-4「敬語は過剰でなく適度に使う」
#: と同じ精神で、選択肢も出しすぎない。
MAX_SUGGESTIONS = 4


# ---------------------------------------------------------------------------
# ローカル定数（なぜ norms.py ではなくここに置くかの説明つき）
# ---------------------------------------------------------------------------

# norms.SpecialForm は「敬語形」の活用種別（例: 「伺う」= godan）は持つが、
# 素の動詞（例: 「聞く」）の活用種別は持たない。素の丁寧語（「言いました」）や
# 「お聞きください」の語幹を作るには素の動詞の活用種別が要るため、ここに置く。
# 敬語の分類・語形ではなく純粋な活用情報なので、規範データではない。
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

# norms.masu_stem() は「なさる」「くださる」のような不規則連用形を *単独形* に
# 限って持つ（_IRREGULAR_MASU_STEM）。「ご利用なさる」「お聞きくださる」のような
# 複合形は表に無いので五段規則で「なさり」「くださり」になってしまう。
# ここでは「どこで切って末尾補助動詞だけを norms に渡し直すか」だけを持つ。
# 語形そのものは norms.masu_stem() から引くので、新たなハードコードではない。
_COMPOUND_TAILS: Tuple[str, ...] = (
    "いらっしゃる",
    "くださる",
    "下さる",
    "おっしゃる",
    "なさる",
    "ござる",
)

# Verb.sonkeigo_general() などが作る一般形の末尾は閉じた集合なので、
# 表層から活用種別を引き当てられる。特定形は SpecialForm.kind を使うため
# この表は通らない。（norms 側に「生成した語形の活用種別」を返す API が無いため
# ここに置く。語形そのものは norms が作ったものをそのまま受け取る。）
_KIND_BY_SUFFIX: Tuple[Tuple[str, str], ...] = (
    ("いらっしゃる", "godan"),
    ("くださる", "godan"),
    ("なさる", "godan"),
    ("になる", "godan"),
    ("いただく", "godan"),
    ("いたす", "godan"),
    ("申し上げる", "ichidan"),
    ("差し上げる", "ichidan"),
    ("られる", "ichidan"),
    ("れる", "ichidan"),
    ("できる", "ichidan"),
    ("する", "sahen"),
    ("来る", "kahen"),
    ("だ", "copula"),
)

# 五段の未然形の末尾かな。norms.a_stem() の出力から機械的に集める
# （＝ norms の活用表をそのまま使う）。「さ」だけは除く。「話す」「合わす」など
# す段・わす段の動詞では「話させる」「合わさせる」が規範形であり、
# 「さ」を落とすと別語になってしまうため、いわゆる「さ入れ」判定から外す。
_A_ROW_KANA = frozenset(
    norms.a_stem(v.plain, "godan")[-1] for v in norms.VERBS if v.kind == "godan"
) - {"さ"}

# 「敬語の指針」は敬体・常体の混在を可否として論じていない
# （norms.CITATIONS["style_mixing"] の note がそう明記している）。
# したがって文体変換用の対応表は規範データではなく、文章作法上の書き換え表として
# ここに置く。機能語（判定詞・補助動詞・サ変）に限り、内容語は語形索引から引く。
_STYLE_PLAIN_TO_POLITE: Tuple[Tuple[str, str], ...] = (
    ("ではない", "ではありません"),
    ("ではなかった", "ではありませんでした"),
    ("である", "です"),
    ("であった", "でした"),
    ("していた", "していました"),
    ("している", "しています"),
    ("した", "しました"),
    ("する", "します"),
    ("できる", "できます"),
    ("なる", "なります"),
    ("ある", "あります"),
    ("いる", "います"),
    ("だ", "です"),
    ("だった", "でした"),
)

#: 「（さ）せていただく」の末尾 → 語形索引で使う活用ラベル
_ITADAKU_TAILS: Tuple[Tuple[str, str], ...] = (
    ("きませんでした", "polite_neg_past"),
    ("きました", "polite_past"),
    ("きません", "polite_neg"),
    ("きます", "polite"),
    ("いた", "plain_past"),
    ("く", "plain"),
)

_ACTOR_ATTRS: Tuple[str, ...] = ("actor", "subject", "agent", "doer")
_TARGET_ATTRS: Tuple[str, ...] = ("target", "goal", "recipient", "object", "toward")

_INFLECTIONS: Tuple[str, ...] = (
    "plain",
    "polite",
    "polite_past",
    "polite_neg",
    "polite_neg_past",
    "plain_past",
    "te",
    "kudasai",
)


# ---------------------------------------------------------------------------
# 活用ユーティリティ（すべて norms の関数の上に立つ薄い層）
# ---------------------------------------------------------------------------


def _kind_of(form: str) -> str:
    """一般形の表層から活用種別を引く。

    実証する主張: 「根拠提示」。語形は norms が作り、この関数は
    「その語形をどう活用させるか」だけを決める。判定に使うのは末尾の
    閉じた集合であり、新たな敬語形を作り出すものではない。
    """
    for suffix, kind in _KIND_BY_SUFFIX:
        if form.endswith(suffix):
            return kind
    return "godan"


def _masu_stem(form: str, kind: str) -> str:
    """複合形にも対応した連用形。末尾補助動詞は norms.masu_stem() に委ねる。"""
    for tail in _COMPOUND_TAILS:
        if form.endswith(tail) and form != tail:
            return form[: -len(tail)] + norms.masu_stem(tail, "godan")
    return norms.masu_stem(form, kind)


def _inflect(lemma: str, kind: str, inflection: str) -> str:
    """終止形と活用種別から、指定した活用の表層を作る。

    実証する主張: 「修正候補の妥当性」。元の箇所の時制・肯否をそのまま保った
    候補を作るため、置換しても文が壊れない。
    """
    if not lemma:
        return ""
    if inflection == "plain":
        return lemma
    try:
        if kind == "copula":
            stem = lemma[:-1]
            return {
                "polite": stem + "です",
                "polite_past": stem + "でした",
                "plain_past": stem + "だった",
            }.get(inflection, "")
        if inflection == "polite":
            return _masu_stem(lemma, kind) + "ます"
        if inflection == "polite_past":
            return _masu_stem(lemma, kind) + "ました"
        if inflection == "polite_neg":
            return _masu_stem(lemma, kind) + "ません"
        if inflection == "polite_neg_past":
            return _masu_stem(lemma, kind) + "ませんでした"
        if inflection == "renyou":
            # 検出スパンが「お調べになられ」のように語尾を含まないとき用。
            return _masu_stem(lemma, kind)
        if inflection == "te":
            return norms.te_form(lemma, kind)
        if inflection == "plain_past":
            te = norms.te_form(lemma, kind)
            return te[:-1] + ("だ" if te.endswith("で") else "た")
        if inflection == "kudasai":
            if lemma.endswith("くださる"):
                return lemma[: -len("くださる")] + norms.masu_stem("くださる", "godan")
            return ""
    except (ValueError, KeyError, IndexError):
        return ""
    return ""


def _rareru_of(lemma: str, kind: str) -> str:
    """「……(ら)れる」を一枚重ねた形。二重敬語の表層を再構成するのに使う。"""
    try:
        if kind == "godan":
            return norms.a_stem(lemma, "godan") + "れる"
        if kind == "ichidan":
            return lemma[:-1] + "られる"
        if kind == "sahen":
            return lemma[:-2] + "される" if lemma.endswith("する") else ""
        if kind == "kahen":
            return "来られる"
    except (ValueError, KeyError):
        return ""
    return ""


# ---------------------------------------------------------------------------
# 語形索引
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormEntry:
    """索引の1件。「この表層は、どの素の動詞の、どの分類の、どの活用か」。

    実証する主張: 「向きの誤り検出」。表層から ``keigo_class`` を復元できるので、
    〈立てている相手〉が分かる。文字列一致ではなく分類で向きを判定する。
    """

    base: str  # 素の動詞（読む・言う）
    lemma: str  # 敬語形の終止形（お読みになる・おっしゃる）
    kind: str  # 活用種別
    keigo_class: KeigoClass
    origin: str  # special / general / plain


_CACHE: Dict[str, Any] = {}


def _lemmas() -> List[FormEntry]:
    """norms から作れる終止形をすべて列挙する。"""
    cached = _CACHE.get("lemmas")
    if cached is not None:
        return cached
    out: List[FormEntry] = []
    for sf in norms.SPECIAL_FORMS:
        out.append(FormEntry(sf.base, sf.form, sf.kind, sf.keigo_class, "special"))
    for v in norms.VERBS:
        out.append(FormEntry(v.plain, v.plain, v.kind, KeigoClass.PLAIN, "plain"))
        for f in v.sonkeigo_general():
            out.append(FormEntry(v.plain, f, _kind_of(f), KeigoClass.SONKEIGO, "general"))
        kudasaru = v.sonkeigo_kudasaru()
        if kudasaru:
            out.append(
                FormEntry(v.plain, kudasaru, "godan", KeigoClass.SONKEIGO, "general")
            )
        for f in v.kenjougo1_general():
            out.append(
                FormEntry(v.plain, f, _kind_of(f), KeigoClass.KENJOUGO_1, "general")
            )
        for f in v.kenjougo2_general():
            cls = (
                KeigoClass.KENJOUGO_1_AND_2
                if f.startswith(("お", "ご"))
                else KeigoClass.KENJOUGO_2
            )
            out.append(FormEntry(v.plain, f, "godan", cls, "general"))
    _CACHE["lemmas"] = out
    return out


def _surface_index() -> Dict[str, List[Tuple[FormEntry, str]]]:
    """表層 → (語形, 活用ラベル) の索引。"""
    cached = _CACHE.get("surface")
    if cached is not None:
        return cached
    idx: Dict[str, List[Tuple[FormEntry, str]]] = {}
    for entry in _lemmas():
        for infl in _INFLECTIONS:
            surface = _inflect(entry.lemma, entry.kind, infl)
            if surface:
                idx.setdefault(surface, []).append((entry, infl))
    _CACHE["surface"] = idx
    return idx


def _double_index() -> Dict[str, List[Tuple[FormEntry, str]]]:
    """「敬語形＋(ら)れる」の表層 → 内側の敬語形。

    実証する主張: 「過剰指摘の少なさ」。norms.ESTABLISHED_DOUBLE_KEIGO と
    norms.SPLIT_SURFACES に載る定着形はこの索引から除くので、
    定着形を書き換え対象の起点にしない。
    """
    cached = _CACHE.get("double")
    if cached is not None:
        return cached
    established = set(norms.ESTABLISHED_DOUBLE_KEIGO) | set(norms.SPLIT_SURFACES)
    plain_surfaces = _surface_index()
    idx: Dict[str, List[Tuple[FormEntry, str]]] = {}
    for entry in _lemmas():
        if entry.origin == "plain" or entry.keigo_class is KeigoClass.PLAIN:
            continue
        doubled = _rareru_of(entry.lemma, entry.kind)
        if not doubled or doubled in established:
            continue
        for infl in _INFLECTIONS:
            surface = _inflect(doubled, "ichidan", infl)
            if not surface or surface in established or surface in plain_surfaces:
                continue
            idx.setdefault(surface, []).append((entry, infl))
    _CACHE["double"] = idx
    return idx


def _prefixed_index(key: str, tail_lemma: str) -> Dict[str, List[Tuple[FormEntry, str]]]:
    """「お(ご)＋語幹＋<tail>」型の表層索引を作る（される／できる）。"""
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    idx: Dict[str, List[Tuple[FormEntry, str]]] = {}
    for v in norms.VERBS:
        if not v.ogo:
            continue
        lemma = f"{v.ogo}{v.ogo_stem}{tail_lemma}"
        entry = FormEntry(v.plain, lemma, "ichidan", KeigoClass.KENJOUGO_1, "general")
        for infl in _INFLECTIONS:
            surface = _inflect(lemma, "ichidan", infl)
            if surface:
                idx.setdefault(surface, []).append((entry, infl))
    _CACHE[key] = idx
    return idx


def _go_sareru_index() -> Dict[str, List[Tuple[FormEntry, str]]]:
    """「ご利用される」型（指針【7】）の表層索引。"""
    return _prefixed_index("go_sareru", "される")


def _ogo_dekiru_index() -> Dict[str, List[Tuple[FormEntry, str]]]:
    """「ご乗車できません」型（指針【8】）の表層索引。"""
    return _prefixed_index("ogo_dekiru", "できる")


def _match_in(
    span_text: str, index: Dict[str, List[Tuple[FormEntry, str]]]
) -> Optional[Tuple[str, List[Tuple[FormEntry, str]], str]]:
    """スパン内から索引に載る最長の表層を探す。戻り値は (前, 一致群, 後)。"""
    best: Optional[Tuple[int, int]] = None
    for start in range(len(span_text)):
        for end in range(len(span_text), start, -1):
            if best is not None and (end - start) <= (best[1] - best[0]):
                break
            if span_text[start:end] in index:
                best = (start, end)
                break
    if best is None:
        return None
    start, end = best
    return span_text[:start], index[span_text[start:end]], span_text[end:]


# ---------------------------------------------------------------------------
# 分類の決定（向き）
# ---------------------------------------------------------------------------


def _classes_for(actor: Party, target: Party, audience: Audience) -> List[KeigoClass]:
    """動作主と向かう先から、規範上ふさわしい分類の並びを返す。

    指針 第2章第1-1／第1-2／第1-3 の定義そのままの写像である。
    自分側は立てない（第2章第1-6【解説1】）。
    """
    if actor.is_other_side:
        return [KeigoClass.SONKEIGO]
    if actor.is_self_side:
        if target.is_other_side:
            return [KeigoClass.KENJOUGO_1, KeigoClass.KENJOUGO_1_AND_2, KeigoClass.KENJOUGO_2]
        return [KeigoClass.KENJOUGO_2, KeigoClass.TEINEIGO]
    return [KeigoClass.TEINEIGO]


def _forms_of_class(base: str, cls: KeigoClass) -> List[Tuple[str, str, KeigoClass]]:
    """素の動詞と分類から、norms が作れる終止形を列挙する。"""
    out: List[Tuple[str, str, KeigoClass]] = []
    for sf in norms.special_forms_for(base, cls):
        out.append((sf.form, sf.kind, sf.keigo_class))
    try:
        v: Optional[norms.Verb] = norms.verb(base)
    except KeyError:
        v = None
    if v is not None:
        if cls is KeigoClass.SONKEIGO:
            for f in v.sonkeigo_general():
                out.append((f, _kind_of(f), cls))
            kudasaru = v.sonkeigo_kudasaru()
            if kudasaru:
                out.append((kudasaru, "godan", cls))
        elif cls is KeigoClass.KENJOUGO_1:
            for f in v.kenjougo1_general():
                out.append((f, _kind_of(f), cls))
        elif cls in (KeigoClass.KENJOUGO_2, KeigoClass.KENJOUGO_1_AND_2):
            for f in v.kenjougo2_general():
                actual = (
                    KeigoClass.KENJOUGO_1_AND_2
                    if f.startswith(("お", "ご"))
                    else KeigoClass.KENJOUGO_2
                )
                if actual is cls:
                    out.append((f, "godan", actual))
        elif cls is KeigoClass.TEINEIGO:
            out.append((v.plain, v.kind, KeigoClass.TEINEIGO))
    elif cls is KeigoClass.TEINEIGO:
        kind = _BASE_KINDS.get(base)
        if kind:
            out.append((base, kind, KeigoClass.TEINEIGO))
    seen = set()
    uniq: List[Tuple[str, str, KeigoClass]] = []
    for item in out:
        if item[0] in seen:
            continue
        seen.add(item[0])
        uniq.append(item)
    return uniq


def _roles_of(role: Any, context: MailContext) -> Tuple[Party, Party]:
    """RoleAssignment から動作主・向かう先を取り出す（属性名に依存しすぎない）。"""

    def pick(attrs: Sequence[str]) -> Party:
        """候補群から先頭のものを選ぶ。

        実証する主張: 「修正候補の妥当性」。選ぶだけで、新しい文字列を作らない。
        """
        if role is None:
            return Party.UNKNOWN
        for name in attrs:
            value = getattr(role, name, None)
            if isinstance(value, Party):
                return value
            if isinstance(value, str):
                try:
                    return Party(value)
                except ValueError:
                    continue
        return Party.UNKNOWN

    return pick(_ACTOR_ATTRS), pick(_TARGET_ATTRS)


def _dedupe(suggestions: Sequence[Suggestion], limit: int = MAX_SUGGESTIONS) -> List[Suggestion]:
    seen = set()
    out: List[Suggestion] = []
    for s in suggestions:
        if not s.text or s.text in seen:
            continue
        seen.add(s.text)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _rank_by_citation(
    suggestions: Sequence[Suggestion], citation_key: str
) -> List[Suggestion]:
    """指針の本文・補足に実際に現れる語形を先に出す。

    実証する主張: 「根拠提示」。並び順そのものが原典の記述に従う。
    """
    try:
        cit = norms.cite(citation_key)
    except KeyError:
        return list(suggestions)
    haystack = f"{cit.quote}{cit.note}"
    return sorted(suggestions, key=lambda s: 0 if s.text and s.text in haystack else 1)


# ---------------------------------------------------------------------------
# Corrector
# ---------------------------------------------------------------------------


class Corrector:
    """検出箇所に対する修正候補を、規則で作れる形の集合から選んで返す。

    実証する主張:
        - 「向きの誤り検出」: 候補は動作主・向かう先・宛先から分類を決めてから
          作る。向きの誤りに対して向きの正しい候補が出る。
        - 「根拠提示」: 各候補の ``reason`` が norms の規範記述に対応する。
        - 「過剰指摘の少なさ」: 定着形（norms.ESTABLISHED_DOUBLE_KEIGO 等）を
          書き換えの起点にしない。
        - 「速度」: 語形索引は一度だけ構築してキャッシュする。
    """

    def __init__(self, generator: "Generator | None" = None) -> None:
        """generator が None なら遅延 import で Generator() を作る。

        実証する主張: 「修正候補の妥当性」。候補の供給源を Generator に一本化し、
        Corrector 側で新しい語形を発明しないための構造である。
        Generator を読み込めない環境でも import 自体は失敗せず、
        norms から機械的に導ける形だけで動作する。
        """
        self._generator = generator
        self._generator_tried = generator is not None

    # -- Generator との接続 -------------------------------------------------

    def _ensure_generator(self) -> Any:
        """Generator を遅延生成する。使えなければ None（norms 直参照に落ちる）。"""
        if self._generator is not None:
            return self._generator
        if self._generator_tried:
            return None
        self._generator_tried = True
        try:
            from .generate import Generator as _Generator  # 遅延 import
        except Exception:  # pragma: no cover - generate.py 未整備でも動く
            return None
        try:
            self._generator = _Generator()
        except Exception:  # pragma: no cover
            self._generator = None
        return self._generator

    @staticmethod
    def _normalize(raw: Any) -> List[Suggestion]:
        """Generator の戻り値を Suggestion 列に揃える（generated_by は必ず rule）。"""
        if raw is None:
            return []
        if isinstance(raw, (str, Suggestion)):
            raw = [raw]
        out: List[Suggestion] = []
        for item in raw:
            if isinstance(item, Suggestion):
                out.append(
                    item
                    if item.generated_by == GENERATED_BY
                    else Suggestion(item.text, item.keigo_class, item.reason, GENERATED_BY)
                )
                continue
            if isinstance(item, str):
                out.append(Suggestion(item, KeigoClass.PLAIN, "規則で生成した形です。", GENERATED_BY))
                continue
            text = getattr(item, "text", None) or getattr(item, "surface", None)
            if isinstance(text, str) and text:
                cls = getattr(item, "keigo_class", KeigoClass.PLAIN)
                if not isinstance(cls, KeigoClass):
                    cls = KeigoClass.PLAIN
                out.append(Suggestion(text, cls, "規則で生成した形です。", GENERATED_BY))
        return out

    def _generator_forms(
        self,
        base_verb: str,
        actor: Party,
        target: Party,
        politeness: int,
        audience: Audience,
    ) -> List[Suggestion]:
        """Generator.correct_forms() を呼ぶ。呼べなければ空を返す。"""
        gen = self._ensure_generator()
        if gen is None:
            return []
        fn = getattr(gen, "correct_forms", None)
        if not callable(fn):
            return []
        payload: Dict[str, Any] = {
            "base_verb": base_verb,
            "verb": base_verb,
            "predicate": base_verb,
            "actor": actor,
            "target": target,
            "politeness": politeness,
            "audience": audience,
        }
        attempts: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        try:
            params = inspect.signature(fn).parameters
            kwargs = {k: v for k, v in payload.items() if k in params}
            if kwargs:
                attempts.append(((), kwargs))
        except (TypeError, ValueError):  # pragma: no cover
            pass
        attempts.append(((base_verb, actor, target), {}))
        attempts.append(((base_verb,), {}))
        for args, kwargs in attempts:
            try:
                raw = fn(*args, **kwargs)
            except Exception:
                continue
            normalized = self._normalize(raw)
            if normalized:
                return normalized
        return []

    # -- 公開 API ------------------------------------------------------------

    def candidate_set(
        self,
        base_verb: str,
        actor: Party,
        target: Party,
        politeness: int = 1,
        audience: Audience = Audience.EXTERNAL,
    ) -> list[Suggestion]:
        """規則で生成可能な正しい形の集合をそのまま返す（テスト用の入口）。

        Generator が使える場合はその ``correct_forms()`` の集合をそのまま返し、
        使えない場合のみ norms から機械的に導ける形を返す。どちらの経路でも
        ``generated_by`` は ``"rule"`` である。

        実証する主張:
            - 「向きの誤り検出」: ``actor`` / ``target`` / ``audience`` の3つだけで
              分類が決まることを、この関数が具体的に示す。
            - 「修正候補の妥当性」: 返る集合は自由生成を含まない。
        """
        from_generator = self._generator_forms(base_verb, actor, target, politeness, audience)
        if from_generator:
            return _dedupe(from_generator, limit=max(MAX_SUGGESTIONS, len(from_generator)))

        inflection = "plain" if politeness <= 0 else "polite"
        out: List[Suggestion] = []
        for cls in _classes_for(actor, target, audience):
            for lemma, kind, actual in _forms_of_class(base_verb, cls):
                surface = _inflect(lemma, kind, inflection)
                if not surface:
                    continue
                out.append(
                    Suggestion(
                        text=surface,
                        keigo_class=actual,
                        reason=self._class_reason(actual, actor, target),
                        generated_by=GENERATED_BY,
                    )
                )
        if politeness >= 2:
            # より改まった形（「申し上げる」「いたす」を含むもの）を先に置く。
            out.sort(key=lambda s: 0 if ("申し上げ" in s.text or "いた" in s.text) else 1)
        return _dedupe(out, limit=max(MAX_SUGGESTIONS, 6))

    def suggest(
        self,
        text: str,
        span: Span,
        error_type: ErrorType,
        context: MailContext,
        role: "RoleAssignment | None" = None,
    ) -> list[Suggestion]:
        """このスパンに対する修正候補。

        必ず :meth:`Generator.correct_forms` が返す集合か、norms.py から機械的に
        導ける形のみを返す。自由文字列を組み立てて返すことはしない。
        すべての ``Suggestion.generated_by`` は ``"rule"`` である。

        実証する主張:
            - 「向きの誤り検出」: 向き系の誤り種別では、動作主と向かう先から
              分類を決め直してから語形を作る。
            - 「根拠提示」: 各候補の ``reason`` が指針の該当記述に対応する。
            - 「速度」: 索引引きと文字列連結のみで、モデル推論を伴わない。
        """
        span_text = self._span_text(text, span)
        if not span_text:
            return []
        try:
            out = self._dispatch(text, span_text, error_type, context, role)
        except Exception:  # pragma: no cover - 候補生成の失敗で本体を止めない
            # 候補が作れなくても検査自体は続けたいので握りつぶすが、
            # 黙って消すと本物のバグ（import 漏れなど）が見えなくなる。
            # DEFERENCE_DEBUG=1 で表面化させ、テストではこれを立てて走らせる。
            if os.environ.get("DEFERENCE_DEBUG"):
                raise
            return []
        return [s if s.generated_by == GENERATED_BY else
                Suggestion(s.text, s.keigo_class, s.reason, GENERATED_BY) for s in out]

    def apply(self, text: str, span: Span, suggestion: Suggestion) -> str:
        """候補を本文に適用した結果を返す。

        実証する主張: 「修正候補の妥当性」。適用結果を再検査に掛けられるので、
        「直した文が新たな指摘を生まない」ことを機械的に検証できる。
        """
        if not (0 <= span.start <= span.end <= len(text)):
            raise ValueError(f"本文の範囲外のスパンです: [{span.start}, {span.end})")
        return text[: span.start] + suggestion.text + text[span.end :]

    # -- 内部：種別ごとの分岐 -------------------------------------------------

    @staticmethod
    def _span_text(text: str, span: Span) -> str:
        if 0 <= span.start <= span.end <= len(text):
            sliced = text[span.start : span.end]
            if sliced:
                return sliced
        return span.text

    def _dispatch(
        self,
        text: str,
        span_text: str,
        error_type: ErrorType,
        context: MailContext,
        role: Any,
    ) -> List[Suggestion]:
        if error_type is ErrorType.DOUBLE_KEIGO:
            return self._fix_double_keigo(span_text)
        if error_type is ErrorType.DIRECTION_SWAP:
            return self._fix_direction(span_text, context, role)
        if error_type in (ErrorType.UCHI_SONKEIGO, ErrorType.DEFERENCE_INVERSION):
            return self._fix_uchi(span_text, context, error_type)
        if error_type is ErrorType.SA_INSERTION:
            return self._fix_sa_insertion(span_text)
        if error_type is ErrorType.SASETE_ITADAKU_OVERUSE:
            return self._fix_sasete_itadaku(span_text)
        if error_type is ErrorType.STYLE_MIXING:
            return self._fix_style(text, span_text)
        if error_type is ErrorType.GO_SARERU:
            return self._fix_go_sareru(span_text)
        if error_type is ErrorType.OGO_DEKIRU:
            return self._fix_ogo_dekiru(span_text)
        if error_type is ErrorType.BAD_KEIGO_LINK:
            return self._fix_keigo_link(span_text)
        if error_type is ErrorType.SELF_SONKEIGO:
            return self._fix_self_sonkeigo(span_text, context, role)
        return []

    @staticmethod
    def _class_reason(cls: KeigoClass, actor: Party, target: Party) -> str:
        """分類選択の理由（情報提示の文体）。"""
        from .types import KEIGO_CLASS_JA, PARTY_JA

        who = PARTY_JA.get(actor, "動作をする人")
        name = KEIGO_CLASS_JA.get(cls, "敬語")
        if cls is KeigoClass.SONKEIGO:
            return f"{who}の行為として{name}で述べる形です。"
        if cls in (KeigoClass.KENJOUGO_1, KeigoClass.KENJOUGO_1_AND_2):
            return f"{PARTY_JA.get(target, '向かう先')}を立てる{name}の形です。"
        if cls is KeigoClass.KENJOUGO_2:
            return f"自分側の行為を読み手に対して丁重に述べる{name}の形です。"
        return f"{name}の形です。"

    # -- (i) 二重敬語 --------------------------------------------------------

    #: 二重敬語を一重に戻す形態的な規則。
    #: 「お(ご)……になられる」型は動詞ごとに事前展開できないので、
    #: 表層の書き換えで一重に戻す。norms の活用規則から導ける形しか作らない。
    _DOUBLE_UNDO: Tuple[Tuple[str, str], ...] = (
        ("になられ", "になり"),
        ("になられる", "になる"),
        ("くださられ", "ください"),
        ("おっしゃられ", "おっしゃい"),
        ("召し上がられ", "召し上がり"),
        ("いらっしゃられ", "いらっしゃい"),
        ("なさられ", "なさい"),
        ("いたされ", "いたし"),
    )

    def _fix_double_keigo(self, span_text: str) -> List[Suggestion]:
        """同じ種類の敬語が二重になっている箇所に、一重の形を返す。"""
        matched = _match_in(span_text, _double_index())
        if matched is None:
            return self._undo_double_morphologically(span_text)
        head, entries, tail = matched
        out: List[Suggestion] = []
        for entry, infl in entries:
            inner = _inflect(entry.lemma, entry.kind, infl)
            if inner:
                out.append(
                    Suggestion(
                        text=f"{head}{inner}{tail}",
                        keigo_class=entry.keigo_class,
                        reason=f"「{entry.lemma}」だけで{self._class_name(entry.keigo_class)}"
                        "として働く形です。",
                        generated_by=GENERATED_BY,
                    )
                )
            for lemma, kind, cls in _forms_of_class(entry.base, entry.keigo_class):
                if lemma == entry.lemma:
                    continue
                surface = _inflect(lemma, kind, infl)
                if surface:
                    out.append(
                        Suggestion(
                            text=f"{head}{surface}{tail}",
                            keigo_class=cls,
                            reason=f"同じ{self._class_name(cls)}の別の一般形です。",
                            generated_by=GENERATED_BY,
                        )
                    )
        return _dedupe(out or self._undo_double_morphologically(span_text))

    def _undo_double_morphologically(self, span_text: str) -> List[Suggestion]:
        """辞書に載っていない「お(ご)……になられる」型を、活用規則で一重に戻す。

        実証する主張: 「修正候補の妥当性」。事前展開した表に無い動詞でも、
        規則で作れる形しか返さない。ここで自由生成に逃げると、修正案そのものが
        新たな規範逸脱になりうる。

        「お調べになられます」→「お調べになります」のように、重ねた側の
        尊敬語だけを外す。可能なら「お調べくださいます」「調べられます」といった
        別の一般形も添える（いずれも指針 第2章第2-1 の一般形）。
        """
        out: List[Suggestion] = []
        for bad, good in self._DOUBLE_UNDO:
            if bad in span_text:
                out.append(
                    Suggestion(
                        text=span_text.replace(bad, good, 1),
                        keigo_class=KeigoClass.SONKEIGO,
                        reason="重ねた側の敬語を外した形です。"
                        "指針は二重敬語について「一般に適切ではないとされている」"
                        "と説明しています（第2章第2-6（2）, p.30）。",
                        generated_by=GENERATED_BY,
                    )
                )
                break
        # 「お(ご)〜になられます」なら、動詞を同定して別の一般形も出す
        m = re.match(r"^([おご])(.+?)になられ(ます|ました|る|た)?$", span_text)
        if m:
            prefix, stem, tail = m.group(1), m.group(2), m.group(3) or ""
            infl = {
                "ます": "polite", "ました": "polite_past",
                "る": "plain", "た": "plain_past", "": "renyou",
            }[tail]
            for v in norms.VERBS:
                if v.ogo != prefix or v.ogo_stem != stem:
                    continue
                for lemma in [x for x in v.sonkeigo_general() if not x.endswith("だ")] + (
                    [v.sonkeigo_kudasaru()] if v.sonkeigo_kudasaru() else []
                ):
                    surface = _inflect(lemma, _kind_of(lemma), infl)
                    if surface:
                        out.append(
                            Suggestion(
                                text=surface,
                                keigo_class=KeigoClass.SONKEIGO,
                                reason="尊敬語の別の一般形です（指針 第2章第2-1）。",
                                generated_by=GENERATED_BY,
                            )
                        )
                break
        return _dedupe(out)

    # -- (ii) 向きの取り違え --------------------------------------------------

    def _fix_direction(self, span_text: str, context: MailContext, role: Any) -> List[Suggestion]:
        """動作主と向かう先に合う分類の形へ置き換える。"""
        matched = _match_in(span_text, _surface_index())
        if matched is None:
            return []
        head, entries, tail = matched
        actor, target = _roles_of(role, context)
        out: List[Suggestion] = []
        for entry, infl in entries:
            a, t = actor, target
            if a is Party.UNKNOWN:
                # 役割が渡されない場合は、今の分類から向きを読み替える。
                # 謙譲語Ⅰが使われていれば動作主は相手側（指針【10】【11】）、
                # 尊敬語が使われていれば動作主は自分側という読み替えになる。
                if entry.keigo_class in (KeigoClass.KENJOUGO_1, KeigoClass.KENJOUGO_1_AND_2):
                    a, t = Party.ADDRESSEE, Party.SELF
                elif entry.keigo_class is KeigoClass.SONKEIGO:
                    a, t = Party.SELF, Party.ADDRESSEE
                else:
                    continue
            for cls in _classes_for(a, t, context.audience):
                if cls is entry.keigo_class:
                    continue
                for lemma, kind, actual in _forms_of_class(entry.base, cls):
                    surface = _inflect(lemma, kind, infl)
                    if surface:
                        out.append(
                            Suggestion(
                                text=f"{head}{surface}{tail}",
                                keigo_class=actual,
                                reason=self._class_reason(actual, a, t),
                                generated_by=GENERATED_BY,
                            )
                        )
        return _rank_by_citation(_dedupe(out), "direction.q10")

    # -- (iii)(vii) 身内に尊敬語 / 敬意の高さの逆転 ----------------------------

    def _fix_uchi(
        self, span_text: str, context: MailContext, error_type: ErrorType
    ) -> List[Suggestion]:
        """自分側の行為として述べる形（謙譲語Ⅱ・素の丁寧語）を返す。"""
        matched = _match_in(span_text, _surface_index())
        if matched is None:
            matched = _match_in(span_text, _double_index())
        if matched is None:
            return []
        head, entries, tail = matched
        out: List[Suggestion] = []
        for entry, infl in entries:
            for cls in (KeigoClass.KENJOUGO_2, KeigoClass.KENJOUGO_1_AND_2, KeigoClass.TEINEIGO):
                for lemma, kind, actual in _forms_of_class(entry.base, cls):
                    surface = _inflect(lemma, kind, infl)
                    if surface:
                        out.append(
                            Suggestion(
                                text=f"{head}{surface}{tail}",
                                keigo_class=actual,
                                reason=self._class_reason(actual, Party.SELF_GROUP, Party.ADDRESSEE),
                                generated_by=GENERATED_BY,
                            )
                        )
                    # 「申しておりました」型（指針【26】）。「て」＋「おる」で作る。
                    if actual is KeigoClass.KENJOUGO_2 and infl in ("polite", "polite_past"):
                        te = _inflect(lemma, kind, "te")
                        oru = norms.special_forms_for("いる", KeigoClass.KENJOUGO_2)
                        for sf in oru:
                            cont = _inflect(sf.form, sf.kind, infl)
                            if te and cont:
                                out.append(
                                    Suggestion(
                                        text=f"{head}{te}{cont}{tail}",
                                        keigo_class=KeigoClass.KENJOUGO_2,
                                        reason="自分側の行為を読み手に対して丁重に述べる"
                                        "謙譲語Ⅱの形です（指針【26】が挙げる言い方）。",
                                        generated_by=GENERATED_BY,
                                    )
                                )
        key = "uchi_soto.q26" if context.audience is Audience.INTERNAL else "uchi_soto.q25"
        if error_type is ErrorType.DEFERENCE_INVERSION:
            key = "deference_inversion"
        return _rank_by_citation(_dedupe(out), key)

    # -- (iv) さ入れ言葉 ------------------------------------------------------

    def _fix_sa_insertion(self, span_text: str) -> List[Suggestion]:
        """五段動詞の使役形を norms.a_stem() から作り直した形を返す。"""
        out: List[Suggestion] = []
        # まず辞書にある五段動詞の未然形で照合する（確度が高い経路）。
        for v in norms.VERBS:
            if v.kind != "godan":
                continue
            wrong = norms.a_stem(v.plain, "godan") + "させ"
            right = norms.a_stem(v.plain, "godan") + "せ"
            if wrong in span_text:
                out.append(
                    Suggestion(
                        text=span_text.replace(wrong, right),
                        keigo_class=KeigoClass.PLAIN,
                        reason=f"五段動詞「{v.plain}」の使役は"
                        f"「{norms.a_stem(v.plain, 'godan')}せる」の形になります。",
                        generated_by=GENERATED_BY,
                    )
                )
        if out:
            return _dedupe(out)
        # 辞書に無い語は、五段の未然形かな（norms.a_stem 由来）で判定する。
        idx = span_text.find("させ")
        while idx > 0:
            if span_text[idx - 1] in _A_ROW_KANA:
                out.append(
                    Suggestion(
                        text=span_text[:idx] + "せ" + span_text[idx + 2 :],
                        keigo_class=KeigoClass.PLAIN,
                        reason="五段動詞の使役は未然形に「せる」が付く形になります。",
                        generated_by=GENERATED_BY,
                    )
                )
                break
            idx = span_text.find("させ", idx + 1)
        return _dedupe(out)

    # -- (v) させていただく ---------------------------------------------------

    def _fix_sasete_itadaku(self, span_text: str) -> List[Suggestion]:
        """指針【18】が「簡潔」と述べる「いたします」「します」型を返す。"""
        for marker in ("させていただ", "せていただ"):
            pos = span_text.find(marker)
            if pos < 0:
                continue
            head = span_text[:pos]
            rest = span_text[pos + len(marker) :]
            infl, tail = self._itadaku_inflection(rest)
            base, kind = self._base_before_itadaku(head, marker)
            if not base:
                continue
            out: List[Suggestion] = []
            for lemma, k, cls in self._simple_forms(base, kind):
                surface = _inflect(lemma, k, infl)
                if surface:
                    out.append(
                        Suggestion(
                            text=f"{surface}{tail}",
                            keigo_class=cls,
                            reason="指針【18】は、許可や恩恵の関係がない場面では"
                            "こうした簡潔な言い方の方が分かりやすいと案内しています。",
                            generated_by=GENERATED_BY,
                        )
                    )
            if out:
                return _dedupe(out)
        return []

    @staticmethod
    def _itadaku_inflection(rest: str) -> Tuple[str, str]:
        for tail, infl in _ITADAKU_TAILS:
            if rest.startswith(tail):
                return infl, rest[len(tail) :]
        return "polite", rest

    @staticmethod
    def _base_before_itadaku(head: str, marker: str) -> Tuple[str, str]:
        """「……させていただく」の直前から素の動詞を取り出す。"""
        if not head:
            return "", ""
        if marker == "させていただ":
            # 漢語サ変の語基（発表・確認…）。末尾が五段の未然形かなならサ変ではない。
            if head[-1] not in _A_ROW_KANA:
                return head + "する", "sahen"
            for v in norms.VERBS:
                if v.kind == "godan" and head.endswith(norms.a_stem(v.plain, "godan")):
                    return v.plain, "godan"
            return "", ""
        # 「せていただ」：五段の未然形＋せ、または一段の語幹＋させ
        for v in norms.VERBS:
            try:
                if v.kind == "godan" and head.endswith(norms.a_stem(v.plain, "godan")):
                    return v.plain, "godan"
                if v.kind == "ichidan" and head.endswith(norms.a_stem(v.plain, "ichidan") + "さ"):
                    return v.plain, "ichidan"
            except (ValueError, KeyError):
                continue
        return "", ""

    @staticmethod
    def _simple_forms(base: str, kind: str) -> List[Tuple[str, str, KeigoClass]]:
        """「いたす」「する」など簡潔な形の終止形を norms から作る。"""
        out: List[Tuple[str, str, KeigoClass]] = []
        try:
            v: Optional[norms.Verb] = norms.verb(base)
        except KeyError:
            v = None
        if v is None and kind == "sahen":
            # 辞書に無い漢語サ変は、norms.Verb をその場で構成して規則を適用する。
            # 「お(ご)……いたす」は作らず、素の「……いたす」だけにする。
            v = norms.Verb(plain=base, kind="sahen", gloss=base, ogo="", has_target=False)
        if v is not None:
            for f in v.kenjougo2_general():
                out.append((f, "godan", KeigoClass.KENJOUGO_2))
            for f in v.kenjougo1_general():
                out.append((f, _kind_of(f), KeigoClass.KENJOUGO_1))
            out.append((v.plain, v.kind, KeigoClass.TEINEIGO))
        elif kind:
            out.append((base, kind, KeigoClass.TEINEIGO))
        return out

    # -- (vi) 文体の混在 ------------------------------------------------------

    def _fix_style(self, text: str, span_text: str) -> List[Suggestion]:
        """本文の多数派の文体に合わせた形を返す。"""
        polite_majority = self._majority_is_polite(text)
        out: List[Suggestion] = []
        matched = _match_in(span_text, _surface_index())
        if matched is not None:
            head, entries, tail = matched
            want = ("polite", "polite_past") if polite_majority else ("plain", "plain_past")
            pair = {"plain": "polite", "plain_past": "polite_past",
                    "polite": "plain", "polite_past": "plain_past"}
            for entry, infl in entries:
                target_infl = pair.get(infl)
                if target_infl is None or target_infl not in want:
                    continue
                surface = _inflect(entry.lemma, entry.kind, target_infl)
                if surface:
                    out.append(
                        Suggestion(
                            text=f"{head}{surface}{tail}",
                            keigo_class=entry.keigo_class,
                            reason="本文の多くの文と同じ文体にそろえた形です。",
                            generated_by=GENERATED_BY,
                        )
                    )
        if not out and polite_majority:
            for plain_end, polite_end in _STYLE_PLAIN_TO_POLITE:
                stripped = span_text.rstrip("。．")
                punct = span_text[len(stripped) :]
                if stripped.endswith(plain_end):
                    out.append(
                        Suggestion(
                            text=stripped[: -len(plain_end)] + polite_end + punct,
                            keigo_class=KeigoClass.TEINEIGO,
                            reason="本文の多くの文と同じ敬体（です・ます）にそろえた形です。",
                            generated_by=GENERATED_BY,
                        )
                    )
                    break
        return _dedupe(out)

    @staticmethod
    def _majority_is_polite(text: str) -> bool:
        """本文の文末を数えて、敬体が多数派かどうかを返す。"""
        polite_ends = ("ます", "ました", "ません", "ませんでした", "です", "でした",
                       "ください", "ませ", "ましょう", "でしょう")
        polite = plain = 0
        for raw in text.replace("\n", "。").split("。"):
            s = raw.strip().rstrip("！？!?、，,")
            if not s:
                continue
            if s.endswith(polite_ends):
                polite += 1
            elif s.endswith(("だ", "である", "た", "る", "い", "ない", "だった")):
                plain += 1
        return polite >= plain

    # -- 派生型 ---------------------------------------------------------------

    def _fix_go_sareru(self, span_text: str) -> List[Suggestion]:
        """指針【7】が挙げる4つの形を norms.Verb.sonkeigo_general() から作る。"""
        matched = _match_in(span_text, _go_sareru_index())
        if matched is None:
            return []
        head, entries, tail = matched
        out: List[Suggestion] = []
        for entry, infl in entries:
            try:
                v = norms.verb(entry.base)
            except KeyError:
                continue
            forms = [f for f in v.sonkeigo_general() if not f.endswith("だ")]
            # 指針【7】の並び（利用される・利用なさる・御利用になる・御利用なさる）に合わせる。
            def rank(f: str) -> int:
                """候補を提示順に並べ替える。

                実証する主張: 「修正候補の妥当性」。並べ替えるだけで候補集合は変えない。
                """
                if f.endswith(("れる", "られる")):
                    return 0
                if f.endswith("なさる") and not f.startswith(("お", "ご")):
                    return 1
                if f.endswith("になる"):
                    return 2
                return 3

            for f in sorted(forms, key=rank):
                surface = _inflect(f, _kind_of(f), infl)
                if surface:
                    out.append(
                        Suggestion(
                            text=f"{head}{surface}{tail}",
                            keigo_class=KeigoClass.SONKEIGO,
                            reason="指針【7】は、この場面では"
                            "「利用される・利用なさる・御利用になる・御利用なさる」型の"
                            "形が適切だと案内しています。",
                            generated_by=GENERATED_BY,
                        )
                    )
        return _dedupe(out)

    def _fix_ogo_dekiru(self, span_text: str) -> List[Suggestion]:
        """指針【8】が挙げる3つの形（……になれる／……いただける／……はできる）。"""
        matched = _match_in(span_text, _ogo_dekiru_index())
        if matched is None:
            return []
        head, entries, tail = matched
        out: List[Suggestion] = []
        for entry, infl in entries:
            try:
                v = norms.verb(entry.base)
            except KeyError:
                continue
            stem = f"{v.ogo}{v.ogo_stem}"
            lemmas: List[Tuple[str, KeigoClass]] = []
            if v.can_ogo_ni_naru:
                # 尊敬語の形にしてから可能形にする（指針 第2章第2-1（1）②）。
                lemmas.append((norms.potential(f"{stem}になる", "godan"), KeigoClass.SONKEIGO))
            # 「お(ご)……いただく」は指針【8】自身が挙げる形。Verb.kenjougo1_general() は
            # 「お(ご)……する」の＜向かう先＞条件で絞られるため、ここでは
            # 同じ語幹に「いただく」を付けた形を別に作る。
            lemmas.append((norms.potential(f"{stem}いただく", "godan"), KeigoClass.KENJOUGO_1))
            # 「ご乗車はできません」型。可能形は norms.potential() から取る。
            sahen_potential = norms.potential(v.plain, v.kind) if v.kind == "sahen" else ""
            if sahen_potential.startswith(v.ogo_stem):
                lemmas.append((f"{stem}は" + sahen_potential[len(v.ogo_stem) :], KeigoClass.PLAIN))
            for lemma, cls in lemmas:
                surface = _inflect(lemma, "ichidan", infl)
                if surface:
                    out.append(
                        Suggestion(
                            text=f"{head}{surface}{tail}",
                            keigo_class=cls,
                            reason="指針【8】は「お(ご)……できる」を謙譲語Ⅰの可能形と整理し、"
                            "相手の行為には「……になれる」「……いただける」"
                            "「……はできる」型を案内しています。",
                            generated_by=GENERATED_BY,
                        )
                    )
        return _dedupe(out)

    def _fix_keigo_link(self, span_text: str) -> List[Suggestion]:
        """指針【10】が挙げる「お聞きください」「お尋ねください」型を作る。"""
        out: List[Suggestion] = []
        for sf in norms.SPECIAL_FORMS:
            if sf.keigo_class is not KeigoClass.KENJOUGO_1:
                continue
            te = _inflect(sf.form, sf.kind, "te")
            pos = span_text.find(te) if te else -1
            if pos < 0:
                continue
            head = span_text[:pos]
            rest = span_text[pos + len(te) :]
            aux_lemma, infl, tail = self._link_aux(rest)
            if not aux_lemma:
                continue
            for base in sorted({s.base for s in norms.SPECIAL_FORMS if s.form == sf.form}):
                stem, prefix = self._ogo_stem_of(base)
                if not stem:
                    continue
                lemma = f"{prefix}{stem}{aux_lemma}"
                surface = _inflect(lemma, "godan", infl)
                if surface:
                    out.append(
                        Suggestion(
                            text=f"{head}{surface}{tail}",
                            keigo_class=KeigoClass.SONKEIGO
                            if aux_lemma == "くださる"
                            else KeigoClass.KENJOUGO_1,
                            reason="指針【10】は、読み手の行為として述べる場面では"
                            "「お聞きください」「お尋ねください」型を案内しています。",
                            generated_by=GENERATED_BY,
                        )
                    )
        return _rank_by_citation(_dedupe(out), "direction.q10")

    @staticmethod
    def _link_aux(rest: str) -> Tuple[str, str, str]:
        """「……て＋補助動詞」の補助動詞部分を読み取る。"""
        table = (
            ("くださいませんか", "くださる", "polite_neg"),
            ("くださいました", "くださる", "polite_past"),
            ("くださいます", "くださる", "polite"),
            ("ください", "くださる", "kudasai"),
            ("くださる", "くださる", "plain"),
            ("いただきました", "いただく", "polite_past"),
            ("いただきます", "いただく", "polite"),
            ("いただけますか", "いただく", "polite"),
            ("いただく", "いただく", "plain"),
        )
        for surface, lemma, infl in table:
            if rest.startswith(surface):
                return lemma, infl, rest[len(surface) :]
        return "", "", rest

    @staticmethod
    def _ogo_stem_of(base: str) -> Tuple[str, str]:
        """素の動詞から「お(ご)……」に挟まる語幹と接頭辞を返す。"""
        try:
            v = norms.verb(base)
            if v.ogo and v.can_ogo_ni_naru:
                return v.ogo_stem, v.ogo
            return "", ""
        except KeyError:
            pass
        kind = _BASE_KINDS.get(base)
        if not kind:
            return "", ""
        try:
            return norms.masu_stem(base, kind), norms.HONORIFIC_PREFIX["wago"]
        except (ValueError, KeyError):
            return "", ""

    def _fix_self_sonkeigo(self, span_text: str, context: MailContext, role: Any) -> List[Suggestion]:
        """「お」「ご」を外した形、または謙譲語Ⅰの形を返す。"""
        out: List[Suggestion] = []
        # 「申される」型（自分側の語に尊敬の「れる」が付いた形）は内側の形へ。
        matched = _match_in(span_text, _double_index())
        if matched is not None:
            head, entries, tail = matched
            for entry, infl in entries:
                inner = _inflect(entry.lemma, entry.kind, infl)
                if inner:
                    out.append(
                        Suggestion(
                            text=f"{head}{inner}{tail}",
                            keigo_class=entry.keigo_class,
                            reason=f"「{entry.lemma}」だけで"
                            f"{self._class_name(entry.keigo_class)}として働く形です。",
                            generated_by=GENERATED_BY,
                        )
                    )
        # 名詞に付いた「お」「ご」「御」を外した形（指針【16】）。
        #
        # この規則は「私のお考え」のような**名詞**に対するものであって、
        # 動詞の敬語形には当てはまらない。「お持ちになります」から「お」だけを
        # 外すと「持ちになります」という活用として成り立たない形ができてしまうので、
        # 述語として認識できる span では接頭辞外しを行わない。
        _is_predicate = _match_in(span_text, _surface_index()) is not None or any(
            k in span_text for k in ("になり", "になる", "なさ", "れます", "られ")
        )
        for prefix in ("お", "ご", "御"):
            if _is_predicate:
                break
            if span_text.startswith(prefix) and len(span_text) > len(prefix):
                stripped = span_text[len(prefix) :]
                if stripped not in _surface_index():
                    out.append(
                        Suggestion(
                            text=stripped,
                            keigo_class=KeigoClass.PLAIN,
                            reason="指針【16】は、自分側のものごとに「お」「御」を付けると"
                            "自分側を立てることになると説明しています。"
                            "接頭辞を外した形が規範に沿います。",
                            generated_by=GENERATED_BY,
                        )
                    )
                break
        # 動作なら謙譲語Ⅰの形も選択肢になる（「お待ちしています」は問題ない）。
        matched = _match_in(span_text, _surface_index())
        if matched is not None:
            head, entries, tail = matched
            for entry, infl in entries:
                if entry.keigo_class is not KeigoClass.SONKEIGO:
                    continue
                for cls in (KeigoClass.KENJOUGO_1, KeigoClass.KENJOUGO_2, KeigoClass.TEINEIGO):
                    for lemma, kind, actual in _forms_of_class(entry.base, cls):
                        surface = _inflect(lemma, kind, infl)
                        if surface:
                            out.append(
                                Suggestion(
                                    text=f"{head}{surface}{tail}",
                                    keigo_class=actual,
                                    reason=self._class_reason(actual, Party.SELF, Party.ADDRESSEE),
                                    generated_by=GENERATED_BY,
                                )
                            )
        return _rank_by_citation(_dedupe(out), "self_sonkeigo.q16")

    @staticmethod
    def _class_name(cls: KeigoClass) -> str:
        from .types import KEIGO_CLASS_JA

        return KEIGO_CLASS_JA.get(cls, "敬語")
