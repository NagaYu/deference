"""ErrorInjector — 文書化された誤り型を「規則で」注入する。

**LLM に誤りを作らせない。** 規則で作ることでラベルが厳密になり、教師バイアスも
入らない。注入位置（:class:`Span`）と種別（:class:`ErrorType`）を台帳として残し、
`InjectedError.gold_suggestions` に規範上の正しい形を持たせる。

実証する主張との対応:
    - 「向きの誤り検出」: :meth:`ErrorInjector.inject` の向き系の変換
      （DIRECTION_SWAP / UCHI_SONKEIGO / SELF_SONKEIGO / DEFERENCE_INVERSION）は、
      文脈の立場情報を使って初めて作れる。表層の書き換えだけでは同じ文字列が
      正しい場合と誤りの場合の両方に現れるため、ラベルが定まらない。
      この非対称性そのものが、規則ベースの校正ツールが原理的に捕まえられない
      ことの裏返しである。
    - 「過剰指摘の少なさ」: :data:`_ESTABLISHED_FORBIDDEN` により、指針が
      「習慣として定着している」と認めた二重敬語を誤りとして注入しない。
      規範に反するラベルを作らないことが、過剰指摘を学習させないための前提である。
    - 「根拠提示」: 各 :class:`InjectedError` の meta に、その誤りが指針の
      どの記述に対応するかの引用キーを残す。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import norms
from .types import (
    Audience,
    ErrorType,
    FunctionTag,
    GeneratedSentence,
    InjectedError,
    InjectedSample,
    KeigoClass,
    MailContext,
    Party,
    Person,
    Span,
)

__all__ = ["ErrorInjector", "InjectionSpec"]


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------

#: 指針が「習慣として定着している」と述べている二重敬語（p.30）。
#: これらは誤りとして注入してはならない。注入すると規範に反するラベルになる。
_ESTABLISHED_FORBIDDEN: Tuple[str, ...] = tuple(norms.ESTABLISHED_DOUBLE_KEIGO) + (
    "お伺いします",
    "お伺いいたします",
    "お伺い申し上げます",
    "お召し上がりになります",
    "お見えになります",
)

#: 尊敬語の特定形の連用形 → 二重敬語化した連用形（「られ」を足す）
_SONKEIGO_STEM_DOUBLE: Dict[str, str] = {
    "おっしゃい": "おっしゃられ",
    "召し上がり": "召し上がられ",
    "ご覧になり": "ご覧になられ",
    "なさい": "なさられ",
    "いらっしゃい": "いらっしゃられ",
    "おいでになり": "おいでになられ",
}

_POLITE_TAILS: Tuple[str, ...] = ("ますでしょうか", "ますか", "ました", "ません", "ます")


def _hash_seed(seed: int, key: str) -> random.Random:
    """決定的な乱数源。seed と key が同じなら常に同じ結果になる。"""
    h = hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(h, "big"))


def _verb_or_none(base: str) -> Optional[norms.Verb]:
    try:
        return norms.verb(base)
    except KeyError:
        return None


def _is_mail_level(sentence: GeneratedSentence) -> bool:
    """本文全体（複数文）かどうか。

    実証する主張: 「過剰指摘の少なさ」。文体の混在も「させていただく」の
    過剰使用も、**一文だけを見て判定できる誤りではない**。単文に注入すると、
    原理的に検出不可能なものを正解として数えることになり、評価が歪む。
    したがってこれらは本文単位でのみ注入する。
    """
    return sum(sentence.text.count(c) for c in "。！？") >= 3


def _polite_tail(surface: str) -> Tuple[str, str]:
    """述語表層を〈語幹側, 丁寧語の語尾〉に割る。割れなければ語尾は空。"""
    for tail in _POLITE_TAILS:
        idx = surface.find(tail)
        if idx > 0:
            return surface[:idx], surface[idx:]
    return surface, ""


def _forms(
    generator: Any,
    base: str,
    actor: Party,
    target: Party,
    politeness: int,
    audience: Audience,
    want: KeigoClass,
) -> List[str]:
    """Generator に「その立場なら正しい形」を作らせ、指定分類のものだけ返す。

    注入側でも語形は必ず :mod:`deference.norms` 由来のものを使う。
    誤りは「正しい形を、合わない立場に当てる」ことで作る。これが
    〈向きの誤り〉の定義そのものである。
    """
    out: List[str] = []
    for s in generator.correct_forms(base, actor, target, politeness, audience):
        if s.keigo_class is want and s.text not in out:
            out.append(s.text)
    return out


@dataclass(frozen=True)
class InjectionSpec:
    """1回の注入で置き換える内容。

    実証する主張: 「向きの誤り検出」の評価可能性。置換内容・種別・正しい形を1件ぶんまとめて持つ。
    """

    replacement: str
    error_type: ErrorType
    gold: Tuple[str, ...]
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ErrorInjector
# ---------------------------------------------------------------------------


class ErrorInjector:
    """規則で誤りを注入する。

    実証する主張: 「向きの誤り検出」。注入結果は位置と種別が厳密に既知なので、
    検出器の再現率を種別ごとに測れる。評価の目玉（図1）はこの台帳に依存する。
    """

    def __init__(self, seed: int = 0, generator: Any = None) -> None:
        """seed と Generator を受け取る（Generator は省略可）。

        実証する主張: 「向きの誤り検出」の評価可能性。seed を固定するので、評価が実行ごとにぶれない。
        """
        self.seed = seed
        self._generator = generator

    # -- Generator の遅延取得 ------------------------------------------------
    @property
    def generator(self) -> Any:
        """Generator を遅延生成する（循環 import を避けるため）。

        実証する主張: 「速度」。Generator の構築を遅延させ、注入を使わない経路で余計な初期化をしない。
        """
        if self._generator is None:
            from .generate import Generator

            self._generator = Generator(seed=self.seed)
        return self._generator

    # ------------------------------------------------------------------
    # 適用可能性
    # ------------------------------------------------------------------
    def available(self, sentence: GeneratedSentence) -> List[ErrorType]:
        """この文に注入できる誤り型を返す。

        実証する主張: 「向きの誤り検出」。適用条件そのものが立場に依存する
        （身内敬語は動作主が自分側でなければ作れない）ことを明示する。
        """
        out: List[ErrorType] = []
        for et in (
            ErrorType.DOUBLE_KEIGO,
            ErrorType.DIRECTION_SWAP,
            ErrorType.UCHI_SONKEIGO,
            ErrorType.SELF_SONKEIGO,
            ErrorType.SA_INSERTION,
            ErrorType.SASETE_ITADAKU_OVERUSE,
            ErrorType.STYLE_MIXING,
            ErrorType.GO_SARERU,
            ErrorType.OGO_DEKIRU,
            ErrorType.BAD_KEIGO_LINK,
        ):
            if self._build(sentence, et) is not None:
                out.append(et)
        return out

    # ------------------------------------------------------------------
    # 注入
    # ------------------------------------------------------------------
    def inject(
        self, sentence: GeneratedSentence, error_type: ErrorType
    ) -> Optional[InjectedSample]:
        """指定の誤り型を1つ注入する。適用できなければ None。

        実証する主張: 「向きの誤り検出」の評価可能性。位置と種別が既知の誤りを作る。
        """
        spec = self._build(sentence, error_type)
        if spec is None:
            return None
        return self._apply(sentence, [(sentence.predicate_span, spec)])

    def inject_random(
        self, sentence: GeneratedSentence, n_errors: int = 1
    ) -> Optional[InjectedSample]:
        """適用可能な誤り型からランダムに選んで注入する（決定的）。

        実証する主張: 「向きの誤り検出」の評価可能性。seed 固定で再現するので、評価が実行ごとにぶれない。
        """
        avail = self.available(sentence)
        if not avail:
            return None
        rng = _hash_seed(self.seed, f"rand|{sentence.text}|{n_errors}")
        et = rng.choice(avail)
        return self.inject(sentence, et)

    def inject_into_mail(
        self, mail: GeneratedSentence, n_errors: int = 1
    ) -> Optional[InjectedSample]:
        """メール本文全体の任意の文に注入する。オフセットは本文全体基準。

        実証する主張: 「向きの誤り検出」。敬意の程度の不整合
        （DEFERENCE_INVERSION）は本文全体を見ないと判定できないため、
        本文単位の注入がどうしても要る。
        """
        sentences = self._mail_sentences(mail)
        if not sentences:
            return None
        rng = _hash_seed(self.seed, f"mail|{mail.text}|{n_errors}")

        # 注入候補（文, スパン, spec）を集める
        cands: List[Tuple[Span, InjectionSpec]] = []
        order = list(range(len(sentences)))
        rng.shuffle(order)
        for i in order:
            sub = sentences[i]
            avail = self.available(sub)
            if not avail:
                continue
            et = rng.choice(avail)
            spec = self._build(sub, et)
            if spec is None or sub.predicate_span is None:
                continue
            offset = sub.meta.get("mail_offset", 0)
            gspan = sub.predicate_span.shifted(offset)
            if any(gspan.overlaps(s) for s, _ in cands):
                continue
            cands.append((gspan, spec))
            if len(cands) >= n_errors:
                break
        if not cands:
            return None
        return self._apply(mail, cands)

    # ------------------------------------------------------------------
    # 敬意の逆転（本文単位でしか作れない）
    # ------------------------------------------------------------------
    def inject_deference_inversion(
        self, context: MailContext
    ) -> Optional[InjectedSample]:
        """身内を相手側より高く述べる本文を構成する。

        実証する主張: 「向きの誤り検出」。この誤りは単文では定義できない。
        同じ「おっしゃいました」が、相手側に使えば適切、身内に使えば規範から
        外れる（指針 第3章第3-2【25】, p.44-45）。宛先と人物の立場が分からなければ
        判定できない類型であり、表層規則が最も無力な領域である。
        """
        if context.audience is not Audience.EXTERNAL:
            return None
        insider = next(
            (p for p in context.persons if p.side is Party.SELF_GROUP), None
        )
        outsider = next(
            (p for p in context.persons if p.side is Party.ADDRESSEE_GROUP), None
        )
        if insider is None:
            insider = Person("佐藤", Party.SELF_GROUP, "社長", context.writer_org)
        if outsider is None:
            outsider = Person(
                context.recipient_name, Party.ADDRESSEE, "様", context.recipient_org
            )

        head = f"{context.recipient_name}様\n\nいつもお世話になっております。\n"
        # 身内を尊敬語で高く述べ、相手側を丁重語どまりで低く述べる
        bad_in = f"弊社の{insider.name}{insider.title}が、そのようにおっしゃっておりました。"
        bad_out = f"{outsider.name}様は、その件について申しておりました。"
        text = head + bad_in + "\n" + bad_out + "\n\nよろしくお願いいたします。\n"

        errors: List[InjectedError] = []
        s1 = text.index("おっしゃっておりました")
        errors.append(
            InjectedError(
                span=Span(s1, s1 + len("おっしゃっておりました"), "おっしゃっておりました"),
                error_type=ErrorType.DEFERENCE_INVERSION,
                original_text="申しておりました",
                gold_suggestions=("申しておりました", "申しておりました。"),
                meta={
                    "actor": Party.SELF_GROUP.value,
                    "person": insider.name,
                    "audience": context.audience.value,
                    "role": "身内を尊敬語で高く述べている",
                    "citation_key": "deference_inversion",
                },
            )
        )
        s2 = text.index("申しておりました", s1 + 1)
        errors.append(
            InjectedError(
                span=Span(s2, s2 + len("申しておりました"), "申しておりました"),
                error_type=ErrorType.DEFERENCE_INVERSION,
                original_text="おっしゃっていました",
                gold_suggestions=("おっしゃっていました", "おっしゃっておりました"),
                meta={
                    "actor": Party.ADDRESSEE.value,
                    "person": outsider.name,
                    "audience": context.audience.value,
                    "role": "相手側を謙譲語Ⅱで低く述べている",
                    "citation_key": "deference_inversion",
                },
            )
        )
        sample = InjectedSample(
            text=text,
            context=context,
            errors=tuple(errors),
            source_text=(
                head
                + f"弊社の{insider.name}が、そのように申しておりました。\n"
                + f"{outsider.name}様は、その件についておっしゃっていました。\n"
                + "\nよろしくお願いいたします。\n"
            ),
            meta={"kind": "deference_inversion", "generated_by": "rule"},
        )
        sample.verify()
        return sample

    # ------------------------------------------------------------------
    # 内部：注入の適用
    # ------------------------------------------------------------------
    def _apply(
        self,
        sentence: GeneratedSentence,
        edits: Sequence[Tuple[Optional[Span], InjectionSpec]],
    ) -> Optional[InjectedSample]:
        """スパンを後ろから置換し、オフセットのずれを避ける。"""
        valid = [(s, sp) for s, sp in edits if s is not None]
        if not valid:
            return None
        valid.sort(key=lambda e: e[0].start, reverse=True)

        text = sentence.text
        placed: List[Tuple[int, int, InjectionSpec, str]] = []
        for span, spec in valid:
            original = text[span.start : span.end]
            text = text[: span.start] + spec.replacement + text[span.end :]
            placed.append((span.start, len(spec.replacement), spec, original))

        errors: List[InjectedError] = []
        for start, length, spec, original in placed:
            gold = tuple(dict.fromkeys((original,) + spec.gold))
            errors.append(
                InjectedError(
                    span=Span(start, start + length, text[start : start + length]),
                    error_type=spec.error_type,
                    original_text=original,
                    gold_suggestions=gold,
                    meta=dict(spec.meta),
                )
            )
        errors.sort(key=lambda e: e.span.start)
        sample = InjectedSample(
            text=text,
            context=sentence.context,
            errors=tuple(errors),
            source_text=sentence.text,
            meta={
                "base_verb": sentence.predicate,
                "actor": sentence.actor.value,
                "target": sentence.target.value,
                "function": sentence.function.value,
                "politeness": sentence.politeness,
                "generated_by": "rule",
            },
        )
        sample.verify()
        return sample

    def _mail_sentences(self, mail: GeneratedSentence) -> List[GeneratedSentence]:
        """generate_mail の meta から文単位の GeneratedSentence を復元する。"""
        out: List[GeneratedSentence] = []
        for rec in mail.meta.get("sentences", ()) or ():
            if not isinstance(rec, dict):
                continue
            try:
                span = rec.get("span")
                start = int(span["start"]) if isinstance(span, dict) else int(rec["start"])
                pspan = rec.get("predicate_span")
                if isinstance(pspan, dict):
                    ps = Span(
                        int(pspan["start"]), int(pspan["end"]), pspan.get("text", "")
                    )
                else:
                    continue
                sub_text = rec.get("text") or ""
                out.append(
                    GeneratedSentence(
                        text=sub_text,
                        context=mail.context,
                        actor=Party(rec.get("actor", "unknown")),
                        target=Party(rec.get("target", "unknown")),
                        predicate=rec.get("predicate", ""),
                        keigo_class=KeigoClass(rec.get("keigo_class", "plain")),
                        politeness=int(rec.get("politeness", 1)),
                        function=FunctionTag(rec.get("function", "report")),
                        predicate_span=ps,
                        meta={"mail_offset": start, **(rec.get("meta") or {})},
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        return out

    # ------------------------------------------------------------------
    # 内部：誤り型ごとの変換
    # ------------------------------------------------------------------
    def _build(
        self, s: GeneratedSentence, et: ErrorType
    ) -> Optional[InjectionSpec]:
        builder = self._BUILDERS.get(et)
        if builder is None or s.predicate_span is None:
            return None
        try:
            return builder(self, s)
        except (KeyError, ValueError, IndexError):
            return None

    # (i) 二重敬語 -----------------------------------------------------------
    def _double_keigo(self, s: GeneratedSentence) -> Optional[InjectionSpec]:
        """「お読みになる」に更に尊敬語を重ねる（指針 第2章第2-6（2）, p.30）。

        実証する主張: 「過剰指摘の少なさ」。定着した二重敬語
        （お伺いする・お召し上がりになる・お見えになる）は生成しない。
        指針がそれらを認めている以上、誤りとして注入するとラベルが規範に反する。
        """
        if s.keigo_class is not KeigoClass.SONKEIGO:
            return None
        surface = s.predicate_span.text
        head, tail = _polite_tail(surface)
        if not tail:
            return None

        new_head: Optional[str] = None
        m = re.match(r"^(お|ご)(.+?)になり$", head)
        if m:
            new_head = f"{m.group(1)}{m.group(2)}になられ"
        else:
            for stem, dbl in _SONKEIGO_STEM_DOUBLE.items():
                if head == stem:
                    new_head = dbl
                    break
        if new_head is None:
            return None
        replacement = new_head + tail
        if any(e in replacement for e in _ESTABLISHED_FORBIDDEN):
            return None

        gold = tuple(
            _forms(
                self.generator,
                s.predicate,
                s.actor,
                s.target,
                s.politeness,
                s.context.audience,
                KeigoClass.SONKEIGO,
            )
        )
        return InjectionSpec(
            replacement,
            ErrorType.DOUBLE_KEIGO,
            gold,
            {
                "from_class": KeigoClass.SONKEIGO.value,
                "doubled_with": "……れる/られる",
                "citation_key": "double_keigo",
                "actor": s.actor.value,
            },
        )

    # (ii) 尊敬語と謙譲語Ⅰの取り違え ------------------------------------------
    def _direction_swap(self, s: GeneratedSentence) -> Optional[InjectionSpec]:
        """立場に合わない分類の敬語を当てる（指針【10】【11】【12】）。

        実証する主張: 「向きの誤り検出」。ここで作る誤りは、**同じ文字列が
        別の立場では正しい**。「お持ちします」は自分が持つなら適切、相手が
        持つなら不適切（指針【11】, p.37）。表層規則ではラベルを付けられない。
        """
        if s.actor.is_other_side and s.keigo_class is KeigoClass.SONKEIGO:
            want, wrong_actor, wrong_target = (
                KeigoClass.KENJOUGO_1,
                Party.SELF,
                Party.ADDRESSEE,
            )
            cite_key = "direction.q11"
            note = "相手側の行為に謙譲語Ⅰを当てている"
        else:
            # 「自分側の行為に尊敬語」は、種別としては UCHI_SONKEIGO（身内）と
            # SELF_SONKEIGO（書き手自身）で扱う。ここで一緒に DIRECTION_SWAP として
            # ラベル付けすると、同じ現象に二つの正解ラベルが存在することになり、
            # 種別分類の正解率が測れなくなる。種別は互いに素であるべきなので、
            # DIRECTION_SWAP は「相手側の行為に謙譲語Ⅰ」（指針【10】【11】【12】）に限る。
            return None

        cands = _forms(
            self.generator,
            s.predicate,
            wrong_actor,
            wrong_target,
            s.politeness,
            s.context.audience,
            want,
        )
        if not cands:
            return None
        _, tail = _polite_tail(s.predicate_span.text)
        rng = _hash_seed(self.seed, f"dir|{s.text}")
        replacement = rng.choice(cands)
        # 元の文末（「〜ますか」「〜ますでしょうか」など）を保つ
        if tail and tail != "ます" and replacement.endswith("ます"):
            replacement = replacement[: -len("ます")] + tail

        gold = tuple(
            _forms(
                self.generator,
                s.predicate,
                s.actor,
                s.target,
                s.politeness,
                s.context.audience,
                s.keigo_class,
            )
        )
        return InjectionSpec(
            replacement,
            ErrorType.DIRECTION_SWAP,
            gold,
            {
                "from_class": s.keigo_class.value,
                "to_class": want.value,
                "actor": s.actor.value,
                "target": s.target.value,
                "audience": s.context.audience.value,
                "citation_key": cite_key,
                "note": note,
            },
        )

    # (iii) 身内に尊敬語 ------------------------------------------------------
    def _uchi_sonkeigo(self, s: GeneratedSentence) -> Optional[InjectionSpec]:
        """社外宛の文で自社の人物に尊敬語を当てる（指針【24】【25】）。

        実証する主張: 「向きの誤り検出」。宛先が社内なら同じ形が適切になりうる
        （指針【25】, p.44-45）。宛先を入力に取らないツールには判定できない。
        """
        if s.actor is not Party.SELF_GROUP:
            return None
        if s.context.audience is not Audience.EXTERNAL:
            return None
        if s.keigo_class is KeigoClass.SONKEIGO:
            return None
        # 「〜いたしかねます」のように語尾が否定・可能の意味を担っている述語は、
        # 丸ごと置き換えると文の意味が変わってしまう。誤りのラベルとしては成立
        # するが、不自然な文を学習データに混ぜないために避ける。
        if any(k in s.predicate_span.text for k in ("かね", "ますよう", "幸い")):
            return None
        cands = _forms(
            self.generator,
            s.predicate,
            Party.ADDRESSEE,
            Party.SELF,
            s.politeness,
            s.context.audience,
            KeigoClass.SONKEIGO,
        )
        if not cands:
            return None
        rng = _hash_seed(self.seed, f"uchi|{s.text}")
        gold = tuple(
            _forms(
                self.generator,
                s.predicate,
                s.actor,
                s.target,
                s.politeness,
                s.context.audience,
                s.keigo_class,
            )
        )
        return InjectionSpec(
            rng.choice(cands),
            ErrorType.UCHI_SONKEIGO,
            gold,
            {
                "from_class": s.keigo_class.value,
                "to_class": KeigoClass.SONKEIGO.value,
                "actor": s.actor.value,
                "audience": s.context.audience.value,
                "citation_key": "uchi_soto.q25",
                "note": "社外宛で身内を立てている",
            },
        )

    # 自分側に尊敬語 ---------------------------------------------------------
    def _self_sonkeigo(self, s: GeneratedSentence) -> Optional[InjectionSpec]:
        """書き手自身の行為に尊敬語を当てる（指針 第2章第1-6, p.22）。"""
        if s.actor is not Party.SELF:
            return None
        if s.keigo_class is KeigoClass.SONKEIGO:
            return None
        # 「〜いたしかねます」のように語尾が否定・可能の意味を担っている述語は、
        # 丸ごと置き換えると文の意味が変わってしまう。誤りのラベルとしては成立
        # するが、不自然な文を学習データに混ぜないために避ける。
        if any(k in s.predicate_span.text for k in ("かね", "ますよう", "幸い")):
            return None
        cands = _forms(
            self.generator,
            s.predicate,
            Party.ADDRESSEE,
            Party.SELF,
            s.politeness,
            s.context.audience,
            KeigoClass.SONKEIGO,
        )
        if not cands:
            return None
        rng = _hash_seed(self.seed, f"selfson|{s.text}")
        gold = tuple(
            _forms(
                self.generator,
                s.predicate,
                s.actor,
                s.target,
                s.politeness,
                s.context.audience,
                s.keigo_class,
            )
        )
        return InjectionSpec(
            rng.choice(cands),
            ErrorType.SELF_SONKEIGO,
            gold,
            {
                "from_class": s.keigo_class.value,
                "to_class": KeigoClass.SONKEIGO.value,
                "actor": s.actor.value,
                "citation_key": "self_sonkeigo.q16",
                "note": "自分側を立てている",
            },
        )

    # (iv) さ入れ言葉 --------------------------------------------------------
    def _sa_insertion(self, s: GeneratedSentence) -> Optional[InjectionSpec]:
        """五段動詞の使役に余分な「さ」を入れる。

        実証する主張: 「根拠提示」。この語形は「敬語の指針」が扱っていないため、
        Deference は指針を根拠に引かず、五段動詞の活用規則を根拠として示す
        （norms.CITATIONS["sa_insertion"]）。根拠の出所を偽らないための区別である。
        """
        if not s.actor.is_self_side:
            return None
        v = _verb_or_none(s.predicate)
        if v is None or v.kind != "godan":
            return None
        a = norms.a_stem(v.plain, v.kind)
        correct = f"{a}せていただきます"
        wrong = f"{a}させていただきます"
        _, tail = _polite_tail(s.predicate_span.text)
        if tail in ("ました",):
            correct = f"{a}せていただきました"
            wrong = f"{a}させていただきました"
        return InjectionSpec(
            wrong,
            ErrorType.SA_INSERTION,
            (correct,),
            {
                "verb": v.plain,
                "kind": v.kind,
                "a_stem": a,
                "citation_key": "sa_insertion",
                "note": "五段動詞の使役は「〜せる」であり「〜させる」ではない",
            },
        )

    # (v) 「させていただく」の過剰使用 ----------------------------------------
    def _sasete_itadaku(self, s: GeneratedSentence) -> Optional[InjectionSpec]:
        """許可も恩恵もない文脈で「させていただく」を使う（指針【18】）。

        実証する主張: 「過剰指摘の少なさ」。指針は一律に不適切とはせず、
        許容度は個人差だと述べる。したがって meta に density を必ず持たせ、
        検出側が「1回だけの使用は指摘しない」判断を取れるようにする。
        """
        if not s.actor.is_self_side:
            return None
        if s.function in (FunctionTag.REQUEST,):
            return None
        if not _is_mail_level(s):
            return None  # 「過剰」は本文中の密度でしか定義できない
        v = _verb_or_none(s.predicate)
        if v is None:
            return None
        if v.kind == "godan":
            core = norms.a_stem(v.plain, v.kind) + "せていただきます"
        elif v.kind == "sahen":
            core = v.plain[:-2] + "させていただきます"
        else:
            core = norms.a_stem(v.plain, v.kind) + "させていただきます"
        _, tail = _polite_tail(s.predicate_span.text)
        if tail == "ました":
            core = core[: -len("ます")] + "ました"
        density = s.text.count("させていただ") + s.text.count("せていただ") + 1
        gold = tuple(
            _forms(
                self.generator,
                s.predicate,
                s.actor,
                s.target,
                s.politeness,
                s.context.audience,
                s.keigo_class,
            )
        )
        return InjectionSpec(
            core,
            ErrorType.SASETE_ITADAKU_OVERUSE,
            gold,
            {
                "density": density,
                "actor": s.actor.value,
                "citation_key": "sasete_itadaku",
                "note": "許可と恩恵の条件が立たない文脈での使用",
            },
        )

    # (vi) 敬体と常体の混在 ---------------------------------------------------
    def _style_mixing(self, s: GeneratedSentence) -> Optional[InjectionSpec]:
        """敬体の本文に常体の文末を混ぜる。

        実証する主張: 「根拠提示」。指針は敬体・常体の混在自体の可否を論じて
        いないため、Deference はこれを規範違反とは呼ばず、文章作法上の観点として
        提示する（norms.CITATIONS["style_mixing"] の note を参照）。
        """
        if s.politeness < 1:
            return None
        if not _is_mail_level(s):
            return None  # 一文だけでは「混在」が定義できない
        surface = s.predicate_span.text
        dic = s.meta.get("dictionary_form") or ""
        if not dic:
            return None
        from .correct import _inflect, _kind_of  # 遅延 import（活用の推定を共有する）

        kind = _kind_of(dic)
        # 「お読みくださいますようお願い申し上げます」のように後続句を伴う述語は
        # 文末だけを常体にできない。素の「〜ます／〜ました」形のときだけ適用する。
        if surface not in (
            _inflect(dic, kind, "polite"),
            _inflect(dic, kind, "polite_past"),
        ):
            return None
        if surface.endswith("ました"):
            try:
                plain = norms.te_form(dic, kind)
            except (ValueError, KeyError):
                return None
            replacement = plain[:-1] + ("だ" if plain.endswith("で") else "た")
        elif surface.endswith("ます"):
            replacement = dic
        else:
            return None
        if replacement == surface:
            return None
        return InjectionSpec(
            replacement,
            ErrorType.STYLE_MIXING,
            (surface,),
            {
                "from_style": "敬体",
                "to_style": "常体",
                "citation_key": "style_mixing",
                "note": "本文の多数派の文体から外れている",
            },
        )

    # 「ご利用される」型 -----------------------------------------------------
    def _go_sareru(self, s: GeneratedSentence) -> Optional[InjectionSpec]:
        """指針【7】（p.36）が「適切な敬語だとは位置付けられてこなかった」形。"""
        if not s.actor.is_other_side:
            return None
        v = _verb_or_none(s.predicate)
        if v is None or v.kind != "sahen" or v.ogo != "ご":
            return None
        base = v.plain[:-2]
        _, tail = _polite_tail(s.predicate_span.text)
        # 元の文が常体なら常体のまま誤りにする。ここで勝手に「ます」を足すと、
        # 注入後の本文（敬体）と gold（常体）が食い違い、修正候補の評価が壊れる。
        replacement = f"ご{base}され{tail}" if tail else f"ご{base}される"
        gold = tuple(
            _forms(
                self.generator,
                s.predicate,
                s.actor,
                s.target,
                s.politeness,
                s.context.audience,
                KeigoClass.SONKEIGO,
            )
        )
        return InjectionSpec(
            replacement,
            ErrorType.GO_SARERU,
            gold,
            {"verb": v.plain, "citation_key": "go_sareru.q7"},
        )

    # 「ご乗車できません」型 --------------------------------------------------
    def _ogo_dekiru(self, s: GeneratedSentence) -> Optional[InjectionSpec]:
        """指針【8】（p.36）。謙譲語Ⅰの可能形を尊敬語の可能形に流用する。"""
        if not s.actor.is_other_side:
            return None
        v = _verb_or_none(s.predicate)
        if v is None or v.kind != "sahen" or v.ogo != "ご":
            return None
        base = v.plain[:-2]
        replacement = f"ご{base}できません"
        gold = (
            f"ご{base}になれません",
            f"ご{base}いただけません",
            f"ご{base}はできません",
        )
        return InjectionSpec(
            replacement,
            ErrorType.OGO_DEKIRU,
            gold,
            {"verb": v.plain, "citation_key": "ogo_dekiru.q8"},
        )

    # 不適切な敬語連結 -------------------------------------------------------
    def _bad_keigo_link(self, s: GeneratedSentence) -> Optional[InjectionSpec]:
        """「伺ってください」型（指針 第2章第2-6（3）, p.30-31／【10】）。"""
        if not s.actor.is_other_side:
            return None
        if s.predicate not in ("聞く", "尋ねる", "訪ねる", "相談する", "確認する"):
            return None
        surface = s.predicate_span.text
        if surface.endswith("ください"):
            replacement = "伺ってください"
        elif surface.endswith("ます"):
            replacement = "伺っていただきます"
        else:
            return None
        gold = ("お聞きください", "お尋ねください")
        return InjectionSpec(
            replacement,
            ErrorType.BAD_KEIGO_LINK,
            gold,
            {
                "citation_key": "keigo_link.bad",
                "note": "「伺う」が相手側ではなく自分側を立ててしまう",
            },
        )

    _BUILDERS = {
        ErrorType.DOUBLE_KEIGO: _double_keigo,
        ErrorType.DIRECTION_SWAP: _direction_swap,
        ErrorType.UCHI_SONKEIGO: _uchi_sonkeigo,
        ErrorType.SELF_SONKEIGO: _self_sonkeigo,
        ErrorType.SA_INSERTION: _sa_insertion,
        ErrorType.SASETE_ITADAKU_OVERUSE: _sasete_itadaku,
        ErrorType.STYLE_MIXING: _style_mixing,
        ErrorType.GO_SARERU: _go_sareru,
        ErrorType.OGO_DEKIRU: _ogo_dekiru,
        ErrorType.BAD_KEIGO_LINK: _bad_keigo_link,
    }

    # ------------------------------------------------------------------
    # 指針の教科書例（全種別の被覆を保証する）
    # ------------------------------------------------------------------
    def canonical_examples(self) -> List[InjectedSample]:
        """「敬語の指針」が実際に挙げている事例を、そのまま評価用に構成する。

        実証する主張: 「向きの誤り検出」と「根拠提示」。合成データだけでなく、
        原典が誤りとして論じている実例そのものを評価に含めることで、
        指標が生成器の癖に依存していないことを示す。
        """
        from .generate import default_context

        ext = default_context(Audience.EXTERNAL)
        ext = ext.with_persons(
            [
                Person("佐藤", Party.SELF_GROUP, "社長", ext.writer_org),
                Person("田中", Party.SELF_GROUP, "部長", ext.writer_org),
                Person("鈴木", Party.ADDRESSEE, "様", ext.recipient_org),
            ]
        )
        specs: List[Tuple[str, str, ErrorType, Tuple[str, ...], str]] = [
            # (本文, 誤りの表層, 種別, 正しい形, 引用キー)
            (
                "担当者に伺ってください。",
                "伺ってください",
                ErrorType.BAD_KEIGO_LINK,
                ("お聞きください", "お尋ねください"),
                "direction.q10",
            ),
            (
                "課長、そのファイルも会議室にお持ちしますか。",
                "お持ちします",
                ErrorType.DIRECTION_SWAP,
                ("お持ちになります",),
                "direction.q11",
            ),
            (
                "来週の日曜日に点検に伺いますが、ご在宅する必要はありません。",
                "ご在宅する",
                ErrorType.DIRECTION_SWAP,
                ("ご在宅なさる", "ご在宅の"),
                "direction.q12",
            ),
            (
                "先生もこの店をよくご利用されるんですか。",
                "ご利用される",
                ErrorType.GO_SARERU,
                ("利用される", "利用なさる", "ご利用になる", "ご利用なさる"),
                "go_sareru.q7",
            ),
            (
                "こちらの車両にはご乗車できません。",
                "ご乗車できません",
                ErrorType.OGO_DEKIRU,
                ("ご乗車になれません", "ご乗車いただけません", "ご乗車はできません"),
                "ogo_dekiru.q8",
            ),
            (
                "資料はもうお読みになられましたか。",
                "お読みになられました",
                ErrorType.DOUBLE_KEIGO,
                ("お読みになりました", "読まれました"),
                "double_keigo",
            ),
            (
                "弊社の佐藤社長が、そのようにおっしゃいました。",
                "おっしゃいました",
                ErrorType.UCHI_SONKEIGO,
                ("申しました", "申しておりました"),
                "uchi_soto.q25",
            ),
            (
                "明日は都合により休まさせていただきます。",
                "休まさせて",
                ErrorType.SA_INSERTION,
                ("休ませて",),
                "sa_insertion",
            ),
            (
                "私が資料をお持ちになります。",
                "お持ちになります",
                ErrorType.SELF_SONKEIGO,
                ("お持ちします", "持参いたします"),
                "self_sonkeigo.q16",
            ),
        ]
        out: List[InjectedSample] = []
        for text, surface, et, gold, ckey in specs:
            idx = text.find(surface)
            if idx < 0:
                continue
            err = InjectedError(
                span=Span(idx, idx + len(surface), surface),
                error_type=et,
                original_text=gold[0] if gold else surface,
                gold_suggestions=gold,
                meta={"citation_key": ckey, "source": "敬語の指針の事例", "canonical": True},
            )
            sample = InjectedSample(
                text=text,
                context=ext,
                errors=(err,),
                source_text=text.replace(surface, gold[0]) if gold else text,
                meta={"kind": "canonical", "generated_by": "rule"},
            )
            sample.verify()
            out.append(sample)

        inv = self.inject_deference_inversion(ext)
        if inv is not None:
            out.append(inv)
        return out
