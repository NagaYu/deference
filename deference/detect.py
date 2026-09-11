"""NormDetector — 立場の情報と規範を突き合わせて、規範から外れた箇所を見つける。

これは表層規則ベースラインとは本質的に違う。判定に使うのは文字列パターンではなく、
:class:`RoleTagger` が推定した〈誰の行為か〉〈誰に向かうか〉と、
:mod:`deference.norms` が持つ〈その分類は誰を立てるか〉である。

実証する主張との対応:
    - 「向きの誤り検出」: :meth:`NormDetector._direction` は、述語の敬語分類と
      動作主の立場が噛み合っているかだけを見る。同じ文字列が立場によって
      適否を変えるため、表層パターンではこの判定に到達できない。
    - 「過剰指摘の少なさ」: 検出したものは捨てずに :class:`VariationSet` に通し、
      揺れに当たるものは :attr:`Verdict.VARIATION` として返す。
      既定では利用者に提示しない。
    - 「根拠提示」: 各 :class:`Finding` に引用キーを載せ、pipeline が
      :class:`RuleCitation` で原典の該当箇所に解決する。
    - 「速度」: 依存は標準ライブラリのみ。モデルを読み込まずに動く。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import norms
from .roles import RoleAssignment, RoleTagger
from .types import (
    Audience,
    ErrorType,
    Finding,
    KeigoClass,
    MailContext,
    Party,
    Span,
    Verdict,
)
from .variation import VariationSet

__all__ = ["NormDetector", "SASETE_ITADAKU_THRESHOLD"]


#: 「させていただく」を指摘する下限。指針【18】は許容度に個人差があるとしており、
#: 1回だけの使用を指摘するのは過剰指摘になる。本文中に3回以上現れたときだけ、
#: 「密度」の観点として情報提示する。
SASETE_ITADAKU_THRESHOLD = 3

#: 向きの誤りを指摘するために必要な、動作主の推定の確からしさの下限。
#: RoleTagger が文型から当てただけ（0.5〜0.6）の場合は指摘しない。
_ACTOR_EVIDENCE_MIN = 0.7

#: 同じ位置に複数の診断が当たったときの優先順位。数字が大きいほど具体的。
#: 指針が名指しで論じている型は、一般的な「向きの取り違え」より優先して示す。
_SPECIFICITY: Dict[ErrorType, int] = {
    ErrorType.SA_INSERTION: 5,
    ErrorType.OGO_DEKIRU: 4,
    ErrorType.GO_SARERU: 4,
    ErrorType.BAD_KEIGO_LINK: 4,
    ErrorType.DOUBLE_KEIGO: 3,
    ErrorType.DEFERENCE_INVERSION: 3,
    ErrorType.UCHI_SONKEIGO: 2,
    ErrorType.SELF_SONKEIGO: 2,
    ErrorType.DIRECTION_SWAP: 1,
    ErrorType.SASETE_ITADAKU_OVERUSE: 1,
    ErrorType.STYLE_MIXING: 0,
}

#: 敬語分類ごとの「立てる度合い」。敬意の逆転の判定に使う。
_DEFERENCE_LEVEL: Dict[KeigoClass, int] = {
    KeigoClass.SONKEIGO: 3,
    KeigoClass.KENJOUGO_1: 2,
    KeigoClass.KENJOUGO_1_AND_2: 2,
    KeigoClass.KENJOUGO_2: 1,
    KeigoClass.TEINEIGO: 1,
    KeigoClass.BIKAGO: 1,
    KeigoClass.PLAIN: 0,
}

#: 二重敬語の表層パターン（同じ種類の敬語の重ね）
_DOUBLE_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"[おご].{1,8}?になられ"), "お(ご)……になる＋られる"),
    (re.compile(r"おっしゃられ"), "おっしゃる＋られる"),
    (re.compile(r"召し上がられ"), "召し上がる＋られる"),
    (re.compile(r"ご覧になられ"), "ご覧になる＋られる"),
    (re.compile(r"いらっしゃられ"), "いらっしゃる＋られる"),
    (re.compile(r"なさられ"), "なさる＋られる"),
    (re.compile(r"[おご].{1,8}?くださられ"), "お(ご)……くださる＋られる"),
    (re.compile(r"拝見させていただ"), "拝見する＋させていただく"),
    (re.compile(r"[おご].{1,8}?いたされ"), "お(ご)……いたす＋れる"),
)

#: 「ご……される」型（指針【7】）
_GO_SARERU = re.compile(r"ご([一-龥ァ-ヶー]{1,6})され(ます|ました|ません|る|た|て)")

#: 「お(ご)……できる」型（指針【8】）
_OGO_DEKIRU = re.compile(r"[おご]([一-龥ァ-ヶー]{1,6})でき(ます|ました|ません|ませんでした|る|ない)")

#: 不適切な敬語連結（指針 p.30-31）
_BAD_LINK = re.compile(r"伺って(ください|くださ|いただ)")

_SASETE = re.compile(r"[させ]せていただ")


def _godan_sa_patterns() -> List[Tuple[re.Pattern, str, str]]:
    """さ入れ言葉の検出パターンを norms の五段動詞から機械的に作る。

    実証する主張: 「根拠提示」。パターンを手で列挙せず活用規則から導くことで、
    根拠が「五段動詞の使役は〜せる」という規則そのものであることを保てる。
    """
    out: List[Tuple[re.Pattern, str, str]] = []
    for v in norms.VERBS:
        if v.kind != "godan":
            continue
        a = norms.a_stem(v.plain, v.kind)
        out.append(
            (
                re.compile(re.escape(a + "させ")),
                v.plain,
                a + "せ",
            )
        )
    return out


_SA_PATTERNS = _godan_sa_patterns()


@dataclass(frozen=True)
class _Hit:
    span: Span
    error_type: ErrorType
    confidence: float
    citation_key: str
    meta: dict
    #: 規範側が「どちらの考え方にも理がある」としている場面では、
    #: 揺れとして扱うことを検出側から指定する。
    force_variation: bool = False


class NormDetector:
    """規範と立場の突き合わせによる検出器（非ニューラル）。

    実証する主張: 「向きの誤り検出」。この検出器は文脈（:class:`MailContext`）
    を必須の入力として取る。同じ本文でも宛先が違えば結果が変わることを
    `tests/test_context_dependence.py` が検証する。
    """

    def __init__(
        self,
        tagger: Optional[RoleTagger] = None,
        variations: Optional[VariationSet] = None,
    ) -> None:
        """役割推定器と揺れ集合を受け取る（省略時は既定のものを作る）。

        実証する主張: 「過剰指摘の少なさ」。揺れ集合を差し替え可能にし、運用ごとに許容範囲を調整できるようにする。
        """
        self.tagger = tagger or RoleTagger()
        self.variations = variations or VariationSet()

    # ------------------------------------------------------------------
    def detect(self, text: str, context: MailContext) -> List[Finding]:
        """本文全体を検査して :class:`Finding` を返す。

        揺れに当たるものは捨てずに ``Verdict.VARIATION`` で返す。UI が
        「揺れ」として別表示できるようにするためである。

        実証する主張: 「向きの誤り検出」と「過剰指摘の少なさ」。立場と規範の突き合わせで検出し、揺れは捨てずに Verdict.VARIATION として返す。
        """
        roles = self.tagger.tag(text, context)
        hits: List[_Hit] = []

        hits.extend(self._direction(text, context, roles))
        hits.extend(self._surface_patterns(text, context))
        hits.extend(self._sasete_density(text, context))
        hits.extend(self._style(text, context, roles))
        inversion, suppressed = self._inversion(text, context, roles)
        hits.extend(inversion)

        # 敬意の逆転として報告した位置では、身内敬語の指摘を重ねない
        hits = [h for h in hits if not (h.span in suppressed and h.error_type is not ErrorType.DEFERENCE_INVERSION)]
        hits = self._dedupe(hits)

        findings: List[Finding] = []
        for h in hits:
            # 揺れ判定は必ず「誤り種別を考慮する」suppresses() で行う。
            # 種別を見ない is_variation() に落とすと、たとえば「させていただく」の
            # 揺れ登録が、同じ位置に重なる「さ入れ言葉」の指摘まで消してしまう。
            # 揺れの範囲は種別ごとに違う、というのがここでの要点である。
            var = self.variations.suppresses(text, h.span, h.error_type)
            verdict = (
                Verdict.VARIATION
                if (var is not None or h.force_variation)
                else Verdict.NORM_DIVERGENCE
            )
            meta = dict(h.meta)
            meta["citation_key"] = h.citation_key
            if var is not None:
                meta["variation_reason"] = var.reason
                meta["variation_acceptability"] = var.acceptability
            findings.append(
                Finding(
                    span=h.span.bind(text),
                    error_type=h.error_type,
                    verdict=verdict,
                    confidence=h.confidence,
                    detector="deference-norm",
                    meta=meta,
                )
            )
        findings.sort(key=lambda f: (f.span.start, f.span.end))
        return findings

    # ------------------------------------------------------------------
    # 向きの判定（このモジュールの中心）
    # ------------------------------------------------------------------
    def _direction(
        self, text: str, context: MailContext, roles: Sequence[RoleAssignment]
    ) -> List[_Hit]:
        """述語の敬語分類と動作主の立場が噛み合っているかを見る。

        指針 第2章第1 の定義そのものを条件にしている:
          - 尊敬語 …… 相手側又は第三者の行為について、その人物を立てる
          - 謙譲語Ⅰ … 自分側から相手側又は第三者に向かう行為について、向かう先を立てる
        したがって「自分側の行為に尊敬語」「相手側の行為に謙譲語Ⅰ」は、
        いずれも立てる向きが噛み合わない。

        実証する主張: 「向きの誤り検出」。ここが規則ベースとの差の本体である。
        """
        out: List[_Hit] = []
        for r in roles:
            # 立場が「文型からの既定値」でしかないときは黙る。
            #
            # 実証する主張: 「過剰指摘の少なさ」。RoleTagger は主語が省略された文でも
            # 文型（依頼なら相手側、報告なら自分側）から動作主を推定するが、それは
            # 当て推量である。向きの誤りは動作主が確かでなければ判定できないので、
            # 明示的な根拠（主語・授受動詞・人物表）がある場合に限って指摘する。
            # これを外すと「お願い申し上げます」のような自分側の定型句を
            # 片端から誤検出することになり、ツールとして信頼されない。
            if r.meta.get("actor_confidence", r.confidence) < _ACTOR_EVIDENCE_MIN:
                continue
            if r.confidence < 0.45:
                continue

            # 指針が許容例として挙げている敬語連結は、そもそも指摘しない。
            #
            # 「お読みになっていただく」は尊敬語と謙譲語Ⅰの連結だが、指針は
            # 「立てる対象が一致しているので、意味的に不合理はなく、許容される」
            # と明記している（第2章第2-6（3）, p.30）。この形では、尊敬語の部分は
            # 相手側の「読む」を指し、「いただく」は書き手が恩恵を受けることを表す。
            # 動作主だけを見て「自分側に尊敬語」と読むと、指針が許容した形を
            # 誤りにしてしまう。
            if r.meta.get("link_listed") == "acceptable":
                continue

            # --- 自分側を立ててしまっている ---------------------------------
            if r.actor.is_self_side and r.keigo_class is KeigoClass.SONKEIGO:
                force_var = False
                if r.actor is Party.SELF_GROUP:
                    et = ErrorType.UCHI_SONKEIGO
                    external = context.audience is Audience.EXTERNAL
                    ckey = "uchi_soto.q25" if external else "uchi_soto.q26"
                    conf = 0.9 if external else 0.5
                    # 宛先が社内のときは、身内を立てること自体が問題にならない。
                    # 指針【26】は、係長が部長に対して課長のことを述べる場面について
                    # 「どちらの考え方にも理がある」とし、適用範囲には個人差があるとする。
                    # したがって社内宛では規範違反として出さず、揺れとして扱う。
                    # ここが「同じ本文でも宛先で結果が変わる」ことの実体である。
                    force_var = not external
                else:
                    et = ErrorType.SELF_SONKEIGO
                    ckey = "self_sonkeigo.q16"
                    conf = 0.85
                    force_var = False
                out.append(
                    _Hit(
                        r.predicate_span,
                        et,
                        conf * r.confidence,
                        ckey,
                        {
                            "actor": r.actor.value,
                            "keigo_class": r.keigo_class.value,
                            "audience": context.audience.value,
                            "evidence": r.evidence,
                            "base_verb": r.base_verb,
                            "variation_reason": (
                                "宛先が社内のため、身内を立てる形も選べます。"
                                "指針【26】は二通りの考え方のどちらにも理があるとしています。"
                            )
                            if force_var
                            else "",
                        },
                        force_variation=force_var,
                    )
                )
                continue

            # --- 相手側の行為に謙譲語Ⅰ -------------------------------------
            if r.actor.is_other_side and r.keigo_class is KeigoClass.KENJOUGO_1:
                # 「〜ていただく」「お(ご)〜いただく」は受け手が自分側なので、
                # RoleTagger が actor=self と読む。ここに来るのはそれ以外。
                if _BAD_LINK.search(r.surface):
                    et, ckey = ErrorType.BAD_KEIGO_LINK, "keigo_link.bad"
                else:
                    et, ckey = ErrorType.DIRECTION_SWAP, "direction.q11"
                out.append(
                    _Hit(
                        r.predicate_span,
                        et,
                        0.85 * r.confidence,
                        ckey,
                        {
                            "actor": r.actor.value,
                            "target": r.target.value,
                            "keigo_class": r.keigo_class.value,
                            "evidence": r.evidence,
                            "base_verb": r.base_verb,
                        },
                    )
                )
                continue

            # --- 謙譲語Ⅱで第三者を立てようとしている（指針【13】） -----------
            if (
                r.actor.is_other_side
                and r.keigo_class is KeigoClass.KENJOUGO_2
                and r.actor is not Party.ADDRESSEE
            ):
                out.append(
                    _Hit(
                        r.predicate_span,
                        ErrorType.DIRECTION_SWAP,
                        0.6 * r.confidence,
                        "direction.q13",
                        {
                            "actor": r.actor.value,
                            "keigo_class": r.keigo_class.value,
                            "evidence": r.evidence,
                            "note": "謙譲語Ⅱは話の中の第三者を立てる働きを持たない",
                        },
                    )
                )
        return out

    # ------------------------------------------------------------------
    # 表層に現れる規範逸脱（向きの情報が要らないもの）
    # ------------------------------------------------------------------
    def _surface_patterns(self, text: str, context: MailContext) -> List[_Hit]:
        """二重敬語・さ入れ・ご〜される・お〜できる・不適切な敬語連結。

        これらは表層だけで判定できるため、規則ベースのツールでも原理的には
        拾える種類である。Deference が優位に立つのは向き系であって、
        ここではないことを図で明示する。
        """
        out: List[_Hit] = []

        # 二重敬語（定着したものは除く）
        for pat, desc in _DOUBLE_PATTERNS:
            for m in pat.finditer(text):
                span = Span(m.start(), m.end(), m.group(0))
                if any(e in text[max(0, m.start() - 2) : m.end() + 4] for e in norms.ESTABLISHED_DOUBLE_KEIGO):
                    continue
                out.append(
                    _Hit(span, ErrorType.DOUBLE_KEIGO, 0.9, "double_keigo", {"pattern": desc})
                )

        # さ入れ言葉
        for pat, verb_plain, correct_stem in _SA_PATTERNS:
            for m in pat.finditer(text):
                out.append(
                    _Hit(
                        Span(m.start(), m.end(), m.group(0)),
                        ErrorType.SA_INSERTION,
                        0.9,
                        "sa_insertion",
                        {"verb": verb_plain, "correct_stem": correct_stem},
                    )
                )

        # 「ご……される」型
        for m in _GO_SARERU.finditer(text):
            out.append(
                _Hit(
                    Span(m.start(), m.end(), m.group(0)),
                    ErrorType.GO_SARERU,
                    0.8,
                    "go_sareru.q7",
                    {"base": m.group(1)},
                )
            )

        # 「お(ご)……できる」型：相手の行為についてのときだけ
        for m in _OGO_DEKIRU.finditer(text):
            out.append(
                _Hit(
                    Span(m.start(), m.end(), m.group(0)),
                    ErrorType.OGO_DEKIRU,
                    0.7,
                    "ogo_dekiru.q8",
                    {"base": m.group(1)},
                )
            )

        # 不適切な敬語連結
        for m in _BAD_LINK.finditer(text):
            end = m.end()
            while end < len(text) and text[end] not in "。\n、":
                end += 1
            out.append(
                _Hit(
                    Span(m.start(), min(end, m.start() + 12), text[m.start() : min(end, m.start() + 12)]),
                    ErrorType.BAD_KEIGO_LINK,
                    0.75,
                    "keigo_link.bad",
                    {},
                )
            )
        return out

    # ------------------------------------------------------------------
    def _sasete_density(self, text: str, context: MailContext) -> List[_Hit]:
        """「させていただく」の密度を見る。

        実証する主張: 「過剰指摘の少なさ」。指針【18】は許容度に個人差があると
        述べているため、1回だけの使用は指摘しない。本文中に
        :data:`SASETE_ITADAKU_THRESHOLD` 回以上現れたときにのみ、
        密度の観点として情報提示する。
        """
        ms = list(_SASETE.finditer(text))
        if len(ms) < SASETE_ITADAKU_THRESHOLD:
            return []
        out: List[_Hit] = []
        for m in ms:
            end = m.end()
            while end < len(text) and text[end] not in "。\n":
                end += 1
            out.append(
                _Hit(
                    Span(m.start(), min(end, m.end() + 6), text[m.start() : min(end, m.end() + 6)]),
                    ErrorType.SASETE_ITADAKU_OVERUSE,
                    0.6,
                    "sasete_itadaku",
                    {"density": len(ms), "threshold": SASETE_ITADAKU_THRESHOLD},
                )
            )
        return out

    # ------------------------------------------------------------------
    def _style(
        self, text: str, context: MailContext, roles: Sequence[RoleAssignment]
    ) -> List[_Hit]:
        """敬体と常体の混在。本文の多数派から外れた文末を拾う。

        実証する主張: 「根拠提示」。指針は混在の可否を論じていないため、
        引用は norms.CITATIONS["style_mixing"]（note に射程外である旨を明記）を使う。
        """
        sentences = self.tagger.split_sentences(text)
        if len(sentences) < 3:
            return []
        styles: List[Tuple[Span, str]] = []
        for sp in sentences:
            body = text[sp.start : sp.end].rstrip("。！？\n 　")
            if not body:
                continue
            if re.search(r"(ます|ました|ません|ませんでした|です|でした|ください)$", body):
                styles.append((sp, "敬体"))
            elif re.search(r"(だ|である|った|た|る|ない|う|く|い)$", body):
                styles.append((sp, "常体"))
        if len(styles) < 3:
            return []
        counts = Counter(s for _, s in styles)
        if len(counts) < 2:
            return []
        majority, n_major = counts.most_common(1)[0]
        minority_n = len(styles) - n_major
        # 少数派が半分近くあるなら「混在」とは言えない（意図的な構成の可能性）
        if minority_n == 0 or minority_n > len(styles) / 2:
            return []
        out: List[_Hit] = []
        for sp, st in styles:
            if st == majority:
                continue
            body = text[sp.start : sp.end].rstrip("。！？\n 　")
            tail_start = sp.start + len(body) - min(len(body), 8)
            out.append(
                _Hit(
                    Span(tail_start, sp.start + len(body), text[tail_start : sp.start + len(body)]),
                    ErrorType.STYLE_MIXING,
                    0.6,
                    "style_mixing",
                    {"majority_style": majority, "this_style": st},
                )
            )
        return out

    # ------------------------------------------------------------------
    def _inversion(
        self, text: str, context: MailContext, roles: Sequence[RoleAssignment]
    ) -> Tuple[List[_Hit], set]:
        """敬意の程度の不整合（相手と身内で高さが逆転）。

        実証する主張: 「向きの誤り検出」。この判定は単文では原理的に不可能で、
        本文全体の敬語分類の分布を見る必要がある。宛先が社外であることも要る。
        表層規則ツールが最も届かない領域である。
        """
        if context.audience is not Audience.EXTERNAL:
            return [], set()
        inside = [
            r
            for r in roles
            if r.actor is Party.SELF_GROUP and r.confidence >= 0.5
        ]
        outside = [
            r
            for r in roles
            if r.actor in (Party.ADDRESSEE, Party.ADDRESSEE_GROUP) and r.confidence >= 0.5
        ]
        if not inside or not outside:
            return [], set()
        lvl_in = max(_DEFERENCE_LEVEL.get(r.keigo_class, 0) for r in inside)
        lvl_out = max(_DEFERENCE_LEVEL.get(r.keigo_class, 0) for r in outside)
        if lvl_in <= lvl_out:
            return [], set()

        hits: List[_Hit] = []
        suppressed = set()
        meta_base = {
            "inside_level": lvl_in,
            "outside_level": lvl_out,
            "audience": context.audience.value,
        }
        for r in inside:
            if _DEFERENCE_LEVEL.get(r.keigo_class, 0) != lvl_in:
                continue
            suppressed.add(r.predicate_span)
            hits.append(
                _Hit(
                    r.predicate_span,
                    ErrorType.DEFERENCE_INVERSION,
                    0.8 * r.confidence,
                    "deference_inversion",
                    {**meta_base, "side": "身内", "note": "相手側より高く述べている"},
                )
            )
        for r in outside:
            if _DEFERENCE_LEVEL.get(r.keigo_class, 0) != lvl_out:
                continue
            hits.append(
                _Hit(
                    r.predicate_span,
                    ErrorType.DEFERENCE_INVERSION,
                    0.7 * r.confidence,
                    "deference_inversion",
                    {**meta_base, "side": "相手側", "note": "身内より低く述べている"},
                )
            )
        return hits, suppressed

    # ------------------------------------------------------------------
    #: 表層だけで真偽が確かめられる誤り種別 → その検証パターン。
    #: モデルがこれらの種別を予測しても、規則が成り立たなければ棄却する。
    _VERIFIABLE = {
        ErrorType.SA_INSERTION: lambda text, span: any(
            pat.search(text[max(0, span.start - 4) : span.end + 4])
            for pat, _, _ in _SA_PATTERNS
        ),
        ErrorType.GO_SARERU: lambda text, span: bool(
            _GO_SARERU.search(text[max(0, span.start - 2) : span.end + 6])
        ),
        ErrorType.OGO_DEKIRU: lambda text, span: bool(
            _OGO_DEKIRU.search(text[max(0, span.start - 2) : span.end + 8])
        ),
        ErrorType.DOUBLE_KEIGO: lambda text, span: any(
            pat.search(text[max(0, span.start - 2) : span.end + 6])
            for pat, _ in _DOUBLE_PATTERNS
        ),
    }

    def is_possible(self, text: str, finding: Finding) -> bool:
        """その指摘が、規則の上で成り立ちうるかを検査する。

        学習済みモデルは「ご報告させていただきます」を さ入れ言葉 と読むことがある。
        しかし「報告する」はサ変動詞で、その使役は「報告させる」が規範形である。
        つまりこの予測は活用規則の上でありえない。

        表層だけで真偽が確かめられる種別については、規則で裏を取り、
        成り立たない予測は捨てる。モデルの自由度を規範の側から拘束する仕掛けで、
        修正候補を規則で作れる形に限定しているのと同じ考え方である。

        実証する主張: 「過剰指摘の少なさ」。規則で否定できる指摘を出さない。
        向き系の種別（文脈が要るもの）はここでは検査しない。表層では
        裏を取れないからであり、そここそがモデルの担当領域である。
        """
        check = self._VERIFIABLE.get(finding.error_type)
        if check is None:
            return True
        return bool(check(text, finding.span))

    @staticmethod
    def _dedupe(hits: Sequence[_Hit]) -> List[_Hit]:
        """同じ位置に複数当たったら、より具体的な診断を残す。

        実証する主張: 「根拠提示」。「ご乗車できません」は動作主の向きから見れば
        DIRECTION_SWAP でもあるが、指針【8】が名指しで論じている OGO_DEKIRU の方が
        利用者にとって有用な説明になる（該当ページと言い換え候補が具体的に示せる）。
        したがって確信度ではなく、診断の具体性を優先する。
        """
        out: List[_Hit] = []
        for h in sorted(hits, key=lambda x: (-_SPECIFICITY.get(x.error_type, 0), -x.confidence)):
            if any(h.span.overlaps(o.span) and h.error_type is o.error_type for o in out):
                continue
            if any(h.span.overlaps(o.span) for o in out):
                continue
            out.append(h)
        return sorted(out, key=lambda x: x.span.start)
