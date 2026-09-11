"""Deference — 検査の入口。役割推定 → 検出 → 揺れの切り分け → 修正候補 → 根拠。

実証する主張との対応:
    - 「速度」: :meth:`Deference.check` は ``elapsed_ms`` を必ず埋める。
      engine="norm" のときは torch を一切読み込まないので、CPU で即応する。
    - 「過剰指摘の少なさ」: 揺れは捨てずに :attr:`Verdict.VARIATION` として保持し、
      ``report_variations=False``（既定）では :attr:`CheckResult.reportable` に載せない。
    - 「根拠提示」: 返る :class:`Finding` は必ず ``citation`` と ``message`` を持つ。
    - 「向きの誤り検出」: :class:`MailContext` を受け取り、宛先で結果が変わる。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from .cite import RuleCitation
from .correct import Corrector
from .detect import NormDetector
from .roles import RoleAssignment, RoleTagger
from .types import (
    Audience,
    CheckResult,
    ErrorType,
    Finding,
    MailContext,
    Span,
    Verdict,
)
from .variation import VariationSet

__all__ = ["Deference"]


class Deference:
    """敬語の検査パイプライン。

    Parameters
    ----------
    engine:
        ``"norm"``   規範＋立場による検出のみ（外部依存なし・最速）
        ``"neural"`` 学習済み ErrorSpanClassifier のみ
        ``"hybrid"`` 両方の和集合
        ``"auto"``   ``model_dir`` があれば hybrid、無ければ norm

    実証する主張: 「速度」。学習済みチェックポイントが無くても必ず動く。
    規範エンジンだけで実用的な精度が出ることが、CPU 即応の前提である。
    """

    def __init__(
        self,
        *,
        engine: str = "auto",
        model_dir: "str | Path | None" = None,
        device: str = "cpu",
        threshold: float = 0.5,
        report_variations: bool = False,
        lang: str = "en",
    ) -> None:
        """エンジンとモデルを選んで検査器を作る。

        実証する主張: 「速度」。engine='norm' なら torch を読み込まない。
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device
        self.threshold = threshold
        self.report_variations = report_variations
        self.lang = "ja" if str(lang).lower().startswith("ja") else "en"

        if engine == "auto":
            has_model = bool(self.model_dir) and (
                self.model_dir.exists() or "/" in str(self.model_dir)
            )
            engine = "hybrid" if has_model else "norm"
        if engine not in ("norm", "neural", "hybrid"):
            raise ValueError(
                f"未知のエンジンです: {engine!r}（norm / neural / hybrid / auto）"
            )
        self.engine = engine

        self.tagger = RoleTagger()
        self.variations = VariationSet()
        self.detector = NormDetector(self.tagger, self.variations)
        self.corrector = Corrector()
        self.citer = RuleCitation(self.lang)
        self._model: Any = None

    # ------------------------------------------------------------------
    @property
    def model(self) -> Any:
        """学習済みモデルを遅延読み込みする。

        実証する主張: 「速度」。engine="norm" では決してここに触れないため、
        torch / transformers の import コスト（数百ms〜数秒）を払わずに済む。
        """
        if self._model is None:
            # ローカルのディレクトリでなければ Hub のリポジトリ ID として扱う
            # （Space からは checkpoint を同梱せず Hub から読む）。
            looks_like_repo_id = bool(
                self.model_dir
                and not self.model_dir.exists()
                and "/" in str(self.model_dir)
                and not str(self.model_dir).startswith((".", "/"))
            )
            if not self.model_dir or (
                not self.model_dir.exists() and not looks_like_repo_id
            ):
                raise FileNotFoundError(
                    f"学習済みモデルが見つかりません: {self.model_dir}。"
                    "scripts/train.py で学習するか、engine='norm' を使ってください。"
                )
            from .model import ErrorSpanClassifier

            self._model = ErrorSpanClassifier.from_pretrained(
                self.model_dir, device=self.device
            )
        return self._model

    # ------------------------------------------------------------------
    def check(
        self, text: str, context: Optional[MailContext] = None
    ) -> CheckResult:
        """本文1通を検査する。

        実証する主張: 4つすべて。立場推定→検出→揺れの切り分け→修正候補→根拠を1回の呼び出しで通し、elapsed_ms に応答時間を残す。
        """
        ctx = context or MailContext()
        t0 = time.perf_counter()

        roles = self.tagger.tag(text, ctx)
        findings: List[Finding] = []

        if self.engine in ("norm", "hybrid"):
            findings.extend(self.detector.detect(text, ctx))
        if self.engine in ("neural", "hybrid"):
            try:
                findings.extend(
                    self.model.predict(text, ctx, threshold=self.threshold)
                )
            except FileNotFoundError:
                if self.engine == "neural":
                    raise

        # 規則で否定できる指摘は、この時点で落とす（モデルの予測を規範で拘束する）
        findings = [f for f in findings if self.detector.is_possible(text, f)]
        findings = self._merge(findings)
        findings = [self._apply_variation_policy(text, f) for f in findings]
        findings = [self._enrich(text, ctx, f, roles) for f in findings]
        findings.sort(key=lambda f: (f.span.start, f.span.end))

        elapsed = (time.perf_counter() - t0) * 1000.0
        return CheckResult(
            text=text,
            context=ctx,
            # 揺れも捨てずに載せる。CheckResult.reportable が verdict で
            # 絞り込むので、既定の出力には現れない。UI が「揺れ」タブとして
            # 別に見せられるよう、データとしては残しておく。
            findings=findings,
            elapsed_ms=elapsed,
            engine=f"deference-{self.engine}",
            meta={
                "n_roles": len(roles),
                "n_reportable": sum(1 for f in findings if f.is_reportable),
                "n_variations": sum(
                    1 for f in findings if f.verdict is Verdict.VARIATION
                ),
                "audience": ctx.audience.value,
            },
        )

    def check_file(
        self, path: "str | Path", context: Optional[MailContext] = None
    ) -> CheckResult:
        """ファイルを読んで検査する。

        実証する主張: 「速度」。CLI の `deference check mail.txt` がそのまま通る経路。
        """
        p = Path(path)
        return self.check(p.read_text(encoding="utf-8"), context)

    def check_batch(
        self,
        texts: Sequence[str],
        contexts: Optional[Sequence[MailContext]] = None,
    ) -> List[CheckResult]:
        """複数通をまとめて検査する。

        実証する主張: 「速度」。評価で条件間の応答時間を同じ手順で測るための入口。
        """
        ctxs = list(contexts) if contexts else [MailContext()] * len(texts)
        return [self.check(t, c) for t, c in zip(texts, ctxs)]

    # ------------------------------------------------------------------
    def _apply_variation_policy(self, text: str, f: Finding) -> Finding:
        """揺れの切り分けを、検出器によらずパイプライン側で一律に適用する。

        NormDetector は自前で :class:`VariationSet` を通すが、ニューラル側は
        通さない。揺れを指摘しないというのは**製品の方針**であって特定の
        検出器の性質ではないので、ここで全ての Finding に同じ規則を当てる。
        これを検出器任せにすると、学習済みモデルを有効にした途端に
        「お伺いいたします」のような定着形を指摘し始める。

        実証する主張: 「過剰指摘の少なさ」。方針を一箇所に集約することで、
        エンジンを差し替えても揺れの扱いが変わらないことを保証する。
        """
        if f.verdict is Verdict.VARIATION:
            return f
        var = self.variations.suppresses(text, f.span, f.error_type)
        if var is None:
            return f
        meta = dict(f.meta)
        meta.setdefault("variation_reason", var.reason)
        meta.setdefault("variation_acceptability", var.acceptability)
        return Finding(
            span=f.span,
            error_type=f.error_type,
            verdict=Verdict.VARIATION,
            confidence=f.confidence,
            suggestions=f.suggestions,
            citation=f.citation,
            message=f.message,
            detector=f.detector,
            meta=meta,
        )

    def _merge(self, findings: Sequence[Finding]) -> List[Finding]:
        """同じ位置に複数の検出器が当たったら、確信度の高い方を残す。

        ただし**揺れの判定だけは規範エンジンの結論を優先する**。
        揺れかどうかは文脈に依存する方針判断（たとえば身内敬語は、宛先が社内なら
        指針【26】により揺れ）であって、モデルはそれを学んでいない。
        確信度だけで勝たせると、社内宛でも身内敬語を指摘してしまい、
        「同じ本文でも宛先で結果が変わる」という中心的な振る舞いが壊れる。

        実証する主張: 「過剰指摘の少なさ」と「向きの誤り検出」。
        """
        out: List[Finding] = []
        for f in sorted(findings, key=lambda x: -x.confidence):
            if any(f.span.overlaps(o.span) for o in out):
                continue
            # 同じ位置に揺れ判定があれば、それを引き継ぐ
            if any(
                o.verdict is Verdict.VARIATION and f.span.overlaps(o.span)
                for o in findings
            ):
                f = Finding(
                    span=f.span,
                    error_type=f.error_type,
                    verdict=Verdict.VARIATION,
                    confidence=f.confidence,
                    suggestions=f.suggestions,
                    citation=f.citation,
                    message=f.message,
                    detector=f.detector,
                    meta={
                        **f.meta,
                        **{
                            k: v
                            for o in findings
                            if o.verdict is Verdict.VARIATION
                            and f.span.overlaps(o.span)
                            for k, v in o.meta.items()
                            if k.startswith("variation_")
                        },
                    },
                )
            out.append(f)
        return out

    def _enrich(
        self,
        text: str,
        ctx: MailContext,
        f: Finding,
        roles: Sequence[RoleAssignment],
    ) -> Finding:
        """修正候補と根拠を埋める。

        実証する主張: 「根拠提示」と「修正候補の妥当性」。候補は必ず
        :class:`Corrector` 経由（＝規則で作れる集合）で、根拠は必ず
        :class:`RuleCitation` 経由（＝原典の該当箇所）で入る。
        """
        role = next((r for r in roles if r.predicate_span.overlaps(f.span)), None)

        if f.verdict is Verdict.VARIATION:
            var = self.variations.suppresses(text, f.span, f.error_type)
            reason = f.meta.get("variation_reason") or (var.reason if var else "")
            citation = var.citation if var else None
            message = self.citer.variation_note(reason, citation)
            return Finding(
                span=f.span,
                error_type=f.error_type,
                verdict=f.verdict,
                confidence=f.confidence,
                suggestions=(),
                citation=citation,
                message=message,
                detector=f.detector,
                meta=f.meta,
            )

        suggestions = tuple(
            self.corrector.suggest(text, f.span, f.error_type, ctx, role)
        )
        key = f.meta.get("citation_key")
        if key:
            from . import norms

            try:
                citation = norms.cite(key)
            except KeyError:
                citation = self.citer.for_error(f.error_type, context=ctx, role=role)
        else:
            citation = self.citer.for_error(f.error_type, context=ctx, role=role)
        message = self.citer.explain(
            f.error_type,
            context=ctx,
            role=role,
            suggestions=suggestions,
            surface=f.span.text,
        )
        return Finding(
            span=f.span,
            error_type=f.error_type,
            verdict=f.verdict,
            confidence=f.confidence,
            suggestions=suggestions,
            citation=citation,
            message=message,
            detector=f.detector,
            meta=f.meta,
        )

    # ------------------------------------------------------------------
    def teach(self, error_type: ErrorType) -> str:
        """誤り種別の仕組みの解説（CLI の ``explain`` と UI のタブで使う）。

        実証する主張: 「根拠提示」。指摘を直すだけでなく、次から自分で選べるようにすることが製品価値の中心である。
        """

    def forms(
        self,
        base_verb: str,
        *,
        audience: Audience = Audience.EXTERNAL,
    ) -> dict:
        """ある動詞について、立場ごとの正しい形を一覧する。

        実証する主張: 「根拠提示」。利用者が学べる出力にするための入口。
        """
        from . import norms
        from .generate import Generator
        from .types import Party

        # 辞書にも特定形にも無い語は、規則で形を作りようがない。
        # 空の表を返して「候補なし」に見せるより、引けないことを明示する。
        known = {v.plain for v in norms.VERBS} | set(norms.SPECIAL_BASES)
        if base_verb not in known:
            raise KeyError(base_verb)

        g = Generator()
        out = {}
        pairs = (
            ("自分 → 相手", Party.SELF, Party.ADDRESSEE),
            ("身内 → 相手", Party.SELF_GROUP, Party.ADDRESSEE),
            ("相手", Party.ADDRESSEE, Party.SELF),
            ("第三者", Party.THIRD_PARTY, Party.ADDRESSEE),
        )
        for label, actor, target in pairs:
            try:
                out[label] = g.correct_forms(
                    base_verb, actor, target, 1, audience
                )
            except KeyError:
                out[label] = []
        return out
