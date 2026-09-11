"""評価の比較条件 (A) textlint / (B) 表層規則群 / (C) 汎用LLM。

**公正な比較が命。** ベースラインを不当に弱く作れば、示せるのは「弱い相手には勝つ」
だけになる。特に (B) は、無料のオンライン校正ツールが実装している水準の規則を
**手加減せずに**書く。それでも敬意の向きは捕まらない、というのが実証したい中身である。

実証する主張との対応:
    - 「向きの誤り検出」: (A) と (B) は :class:`MailContext` を**受け取らない**か、
      受け取っても使わない。表層規則とはそういうものだという定義であり、
      だからこそ〈誰の行為か〉に依存する誤りに到達できない。
    - 「速度」: 全条件が ``elapsed_ms`` を埋めるので、応答時間を横並びで測れる。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import (
    CheckResult,
    ErrorType,
    Finding,
    MailContext,
    Span,
    Verdict,
    ERROR_TYPE_JA,
)

__all__ = [
    "Baseline",
    "TextlintBaseline",
    "RuleBaseline",
    "LLMBaseline",
    "SurfaceRule",
    "ALL_BASELINES",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TEXTLINT_DIR = _REPO_ROOT / ".baseline_textlint"


# ---------------------------------------------------------------------------
class Baseline:
    """比較条件の共通インターフェース。

    実証する主張: 「向きの誤り検出」。比較条件を同じ口で扱い、同じ本文集合・同じ手順で測る。公正な比較の土台。
    """

    name = "baseline"
    label = "baseline"

    def available(self) -> bool:
        """この条件が実行可能か。

        実証する主張: 「向きの誤り検出」。利用できない条件を成績ゼロと取り違えないための入口。
        """
        raise NotImplementedError

    def unavailable_reason(self) -> str:
        """実行できない理由。

        実証する主張: 「根拠提示」。測定不可の理由を結果に残し、空欄を成績と誤読させない。
        """
        return ""

    def check(self, text: str, context: MailContext) -> CheckResult:
        """本文1通を検査する。

        実証する主張: 「速度」。全条件が同じ口で elapsed_ms を返すので、応答時間を横並びで測れる。
        """
        raise NotImplementedError

    def check_batch(
        self, texts: Sequence[str], contexts: Sequence[MailContext]
    ) -> List[CheckResult]:
        """複数通をまとめて検査する。

        実証する主張: 「速度」。条件ごとの起動コストを1件あたりに割り戻して公平に測る。
        """
        return [self.check(t, c) for t, c in zip(texts, contexts)]


# ---------------------------------------------------------------------------
# (A) textlint
# ---------------------------------------------------------------------------

#: textlint の ruleId → Deference の誤り種別。対応が付くものだけを写す。
#: 対応の付かない指摘は ErrorType.NONE のままにして「種別を出さない」ことを表す。
_TEXTLINT_TYPE_MAP: Dict[str, ErrorType] = {
    "ja-technical-writing/no-mix-dearu-desumasu": ErrorType.STYLE_MIXING,
    "japanese/no-mix-dearu-desumasu": ErrorType.STYLE_MIXING,
    "jtf-style/1.1.1.本文": ErrorType.STYLE_MIXING,
    "jtf-style/1.1.2.本文": ErrorType.STYLE_MIXING,
    "jtf-style/1.1.3.本文": ErrorType.STYLE_MIXING,
}


class TextlintBaseline(Baseline):
    """条件 (A)。textlint（日本語プリセット）。

    実証する主張: 「向きの誤り検出」。textlint の日本語プリセットは技術文書の
    作法（文長・二重否定・ら抜き・敬体常体の統一など）を対象としており、
    npm の生態系を調べても**敬語専用のルールは存在しない**。したがって
    敬意の向きの誤りは設計上そもそも扱われていない。この条件は、
    「既存ツールで代替できるか」に対する率直な答えを出すために置く。
    """

    name = "textlint"
    label = "(A) textlint"

    def __init__(
        self,
        workdir: "str | Path" = _DEFAULT_TEXTLINT_DIR,
        presets: Sequence[str] = (
            "preset-ja-technical-writing",
            "preset-jtf-style",
            "preset-ja-spacing",
            "preset-japanese",
        ),
        timeout: float = 180.0,
    ) -> None:
        """textlint の作業ディレクトリとプリセットを設定する。

        実証する主張: 「向きの誤り検出」。日本語プリセットを可能な限り多く有効にし、ベースラインを弱く作らない。
        """
        self.workdir = Path(workdir)
        self.presets = tuple(presets)
        self.timeout = timeout
        self._reason = ""
        self._ensure_config()

    def _ensure_config(self) -> None:
        if not (self.workdir / "node_modules").exists():
            self._reason = (
                f"textlint が見つかりません（{self.workdir}/node_modules）。"
                "`cd .baseline_textlint && npm install` を実行してください。"
            )
            return
        cfg = self.workdir / ".textlintrc.json"
        want = {"rules": {p: True for p in self.presets}}
        try:
            if not cfg.exists() or json.loads(cfg.read_text()) != want:
                cfg.write_text(json.dumps(want, ensure_ascii=False, indent=2))
        except OSError as exc:  # pragma: no cover
            self._reason = f"textlint の設定を書けません: {exc}"

    def available(self) -> bool:
        """textlint と npx が使えるか。

        実証する主張: 「根拠提示」。使えないときは理由を添えて測定不可と記録する。
        """
        return not self._reason and shutil.which("npx") is not None

    def unavailable_reason(self) -> str:
        """textlint が使えない理由。

        実証する主張: 「根拠提示」。測定不可の理由を結果 JSON に残し、空欄を成績と誤読させない。
        """
        if self._reason:
            return self._reason
        if shutil.which("npx") is None:
            return "npx が見つかりません（Node.js が必要です）"
        return ""

    # ------------------------------------------------------------------
    def check(self, text: str, context: MailContext) -> CheckResult:
        """本文1通を textlint に掛ける。

        実証する主張: 「速度」。内部ではバッチ実行に委ね、起動コストを割り戻す。
        """
        return self.check_batch([text], [context])[0]

    def check_batch(
        self, texts: Sequence[str], contexts: Sequence[MailContext]
    ) -> List[CheckResult]:
        """1プロセスでまとめて実行する。

        1文ごとに npx を起動すると1件あたり1秒近くかかり、速度比較が
        textlint に対して不当に不利になる。**まとめて実行し、1件あたりに
        割り戻す**のが公正な測り方である。

        実証する主張: 「速度」。1文ごとに npx を起動すると textlint に不当に不利になるので、まとめて実行して割り戻す。
        """
        if not self.available():
            return [
                CheckResult(
                    text=t,
                    context=c,
                    findings=[],
                    elapsed_ms=0.0,
                    engine=self.name,
                    meta={"unavailable_reason": self.unavailable_reason()},
                )
                for t, c in zip(texts, contexts)
            ]

        tmpdir = Path(tempfile.mkdtemp(prefix="deference-textlint-"))
        try:
            paths: List[Path] = []
            for i, t in enumerate(texts):
                p = tmpdir / f"m{i:05d}.txt"
                p.write_text(t, encoding="utf-8")
                paths.append(p)

            t0 = time.perf_counter()
            proc = subprocess.run(
                ["npx", "textlint", "--format", "json", *[str(p) for p in paths]],
                cwd=str(self.workdir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = (time.perf_counter() - t0) * 1000.0
            per_item = elapsed / max(1, len(texts))

            try:
                payload = json.loads(proc.stdout or "[]")
            except json.JSONDecodeError:
                payload = []

            by_path: Dict[str, List[dict]] = {}
            for entry in payload:
                by_path[str(entry.get("filePath", ""))] = entry.get("messages", [])

            out: List[CheckResult] = []
            for p, t, c in zip(paths, texts, contexts):
                msgs = by_path.get(str(p), []) or by_path.get(str(p.resolve()), [])
                out.append(
                    CheckResult(
                        text=t,
                        context=c,
                        findings=self._to_findings(t, msgs),
                        elapsed_ms=per_item,
                        engine=self.name,
                        meta={"batch_ms": elapsed, "batch_size": len(texts)},
                    )
                )
            return out
        except (subprocess.TimeoutExpired, OSError) as exc:
            return [
                CheckResult(
                    text=t, context=c, findings=[], elapsed_ms=0.0,
                    engine=self.name, meta={"unavailable_reason": str(exc)},
                )
                for t, c in zip(texts, contexts)
            ]
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _to_findings(self, text: str, messages: Sequence[dict]) -> List[Finding]:
        lines = text.split("\n")
        starts: List[int] = []
        acc = 0
        for ln in lines:
            starts.append(acc)
            acc += len(ln) + 1

        out: List[Finding] = []
        for m in messages:
            idx = m.get("index")
            if idx is None:
                line = int(m.get("line", 1)) - 1
                col = int(m.get("column", 1)) - 1
                idx = starts[line] + col if 0 <= line < len(starts) else 0
            start = max(0, min(int(idx), len(text)))
            end = min(len(text), start + max(1, len(str(m.get("message", "")[:1]))))
            rng = m.get("range")
            if isinstance(rng, list) and len(rng) == 2:
                start, end = max(0, int(rng[0])), min(len(text), int(rng[1]))
            if end <= start:
                end = min(len(text), start + 4)
            rule = str(m.get("ruleId") or "")
            out.append(
                Finding(
                    span=Span(start, end, text[start:end]),
                    error_type=_TEXTLINT_TYPE_MAP.get(rule, ErrorType.NONE),
                    verdict=Verdict.NORM_DIVERGENCE,
                    confidence=1.0,
                    message=str(m.get("message", "")).splitlines()[0],
                    detector="textlint",
                    meta={"ruleId": rule, "severity": m.get("severity")},
                )
            )
        return out


# ---------------------------------------------------------------------------
# (B) 表層規則群
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceRule:
    """表層パターン1件。

    実証する主張: 「向きの誤り検出」。表層規則を一件ずつデータとして持ち、何を拾い何を拾わないかを検証できるようにする。
    """

    id: str
    pattern: re.Pattern
    error_type: ErrorType
    message: str


def _r(rid: str, pat: str, et: ErrorType, msg: str) -> SurfaceRule:
    return SurfaceRule(rid, re.compile(pat), et, msg)


#: 無料のオンライン校正ツールが実装している水準の規則群。
#: **手加減しない。** 二重敬語・さ入れ・ら抜き・ご〜される・お〜できる など、
#: 表層で書けるものは可能な限り書く。それでも向きは捕まらないことを示すのが目的。
_SURFACE_RULES: Tuple[SurfaceRule, ...] = (
    # --- 二重敬語 ---------------------------------------------------------
    _r("double.oni-narareru", r"[おご].{1,8}?になられ", ErrorType.DOUBLE_KEIGO, "二重敬語の可能性があります"),
    _r("double.ossharareru", r"おっしゃられ", ErrorType.DOUBLE_KEIGO, "「おっしゃる」に「られる」が重なっています"),
    _r("double.meshiagarareru", r"召し上がられ", ErrorType.DOUBLE_KEIGO, "二重敬語の可能性があります"),
    _r("double.goran-narareru", r"ご覧になられ", ErrorType.DOUBLE_KEIGO, "二重敬語の可能性があります"),
    _r("double.irassharareru", r"いらっしゃられ", ErrorType.DOUBLE_KEIGO, "二重敬語の可能性があります"),
    _r("double.nasarareru", r"なさられ", ErrorType.DOUBLE_KEIGO, "二重敬語の可能性があります"),
    _r("double.okaerininarare", r"お帰りになられ", ErrorType.DOUBLE_KEIGO, "二重敬語の可能性があります"),
    _r("double.haiken-sasete", r"拝見させていただ", ErrorType.DOUBLE_KEIGO, "「拝見する」に「させていただく」が重なっています"),
    _r("double.oukagai-sasete", r"お伺いさせていただ", ErrorType.DOUBLE_KEIGO, "敬語が重なっている可能性があります"),
    _r("double.ukagawasete", r"伺わせていただ", ErrorType.DOUBLE_KEIGO, "敬語が重なっている可能性があります"),
    _r("double.gosetsumei-sare", r"ご.{1,6}?されます", ErrorType.GO_SARERU, "「ご〜される」は適切な尊敬語の形ではないとされています"),
    # --- ご〜される / お〜できる -------------------------------------------
    _r("go-sareru", r"ご([一-龥ァ-ヶー]{1,6})され(ます|ました|る|た|て)", ErrorType.GO_SARERU, "「ご〜される」の形です"),
    _r("ogo-dekiru", r"[おご]([一-龥ァ-ヶー]{1,6})でき(ます|ました|ません|る|ない)", ErrorType.OGO_DEKIRU, "「お（ご）〜できる」は謙譲語Ⅰの可能形です"),
    # --- さ入れ言葉 --------------------------------------------------------
    _r("sa.yoma", r"読まさせて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.yasuma", r"休まさせて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.ika", r"行かさせて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.kaera", r"帰らさせて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.ukagawa", r"伺わさせて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.kaka", r"書かさせて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.okura", r"送らさせて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.mata", r"待たさせて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.tsuka", r"使わさせて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.tanoma", r"頼まさせて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.hanasa", r"話さ+させて", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    _r("sa.generic", r"[かがさたなばまらわ]させていただ", ErrorType.SA_INSERTION, "さ入れ言葉の可能性があります"),
    # --- ら抜き言葉（指針の射程外だが校正ツールは拾う） ----------------------
    _r("ranuki", r"(見れ|食べれ|来れ|出れ|着れ|寝れ|考えれ)(ます|る|ない|た)", ErrorType.NONE, "ら抜き言葉の可能性があります"),
    # --- 接客表現・冗長 ----------------------------------------------------
    _r("manual.nohou", r"の方(で|は|が)?(よろしい|大丈夫|なります)", ErrorType.NONE, "いわゆる接客表現です"),
    _r("manual.ninarimasu", r"円になります", ErrorType.NONE, "いわゆる接客表現です"),
    _r("manual.yoroshikatta", r"よろしかったでしょうか", ErrorType.NONE, "いわゆる接客表現です"),
    _r("manual.narimasu", r"になります(ね|よ)", ErrorType.NONE, "いわゆる接客表現です"),
    _r("redundant.tondemo", r"とんでもございません", ErrorType.NONE, "「とんでもないことでございます」とする案内もあります"),
    _r("redundant.doublepolite", r"(ます|です)(です|ます)", ErrorType.NONE, "丁寧語が重なっています"),
)

_STYLE_PLAIN = re.compile(r"(だ|である|だった|であった|した|する|ない|いる|なる)$")
_STYLE_POLITE = re.compile(r"(ます|ました|ません|ませんでした|です|でした|ください)$")
_SASETE_RE = re.compile(r"させていただ")


class RuleBaseline(Baseline):
    """条件 (B)。無料のオンライン校正ツール相当の規則群（表層パターンのみ）。

    実証する主張: 「向きの誤り検出」。この条件は **:class:`MailContext` を
    一切参照しない**。参照した時点でそれは「表層規則」ではなくなり、比較が
    無意味になるからである。したがって、宛先や人物の立場が分からなければ
    決まらない誤り（身内敬語・向きの取り違え・敬意の逆転）は構造的に拾えない。
    その差こそがこの評価で見たいものなので、規則自体は手加減せずに書いてある。
    """

    name = "rule_baseline"
    label = "(B) 表層規則群"

    RULES = _SURFACE_RULES
    #: 「させていただく」を指摘する回数の下限（一般的な校正ツールに合わせて厳しめ）
    SASETE_THRESHOLD = 2

    def available(self) -> bool:
        """常に実行できる（外部依存が無い）。

        実証する主張: 「速度」。表層規則は外部プロセスを起こさないので、最速の比較対象になる。
        """
        return True

    def check(self, text: str, context: MailContext) -> CheckResult:
        # context は意図的に使わない（上の docstring を参照）
        """表層パターンだけで本文を検査する。

        実証する主張: 「向きの誤り検出」。MailContext を意図的に参照しない。参照した時点で「表層規則」ではなくなり、比較が無意味になる。
        """
        t0 = time.perf_counter()
        findings: List[Finding] = []

        for rule in self.RULES:
            for m in rule.pattern.finditer(text):
                findings.append(
                    Finding(
                        span=Span(m.start(), m.end(), m.group(0)),
                        error_type=rule.error_type,
                        verdict=Verdict.NORM_DIVERGENCE,
                        confidence=1.0,
                        message=rule.message,
                        detector=self.name,
                        meta={"rule_id": rule.id},
                    )
                )

        findings.extend(self._sasete(text))
        findings.extend(self._style(text))

        # 位置が重なったものは1件に畳む
        merged: List[Finding] = []
        for f in sorted(findings, key=lambda x: (x.span.start, -len(x.span))):
            if any(f.span.overlaps(o.span) for o in merged):
                continue
            merged.append(f)

        return CheckResult(
            text=text,
            context=context,
            findings=merged,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            engine=self.name,
            meta={"n_rules": len(self.RULES)},
        )

    def _sasete(self, text: str) -> List[Finding]:
        ms = list(_SASETE_RE.finditer(text))
        if len(ms) < self.SASETE_THRESHOLD:
            return []
        return [
            Finding(
                span=Span(m.start(), m.end(), m.group(0)),
                error_type=ErrorType.SASETE_ITADAKU_OVERUSE,
                verdict=Verdict.NORM_DIVERGENCE,
                confidence=1.0,
                message=f"「させていただく」が{len(ms)}回使われています",
                detector=self.name,
                meta={"rule_id": "sasete.density", "count": len(ms)},
            )
            for m in ms
        ]

    def _style(self, text: str) -> List[Finding]:
        parts = [p for p in re.split(r"(?<=[。！？\n])", text) if p.strip()]
        styles: List[Tuple[int, int, str]] = []
        pos = 0
        for p in parts:
            body = p.rstrip("。！？\n 　")
            if body:
                start = pos + p.index(body)
                if _STYLE_POLITE.search(body):
                    styles.append((start, start + len(body), "敬体"))
                elif _STYLE_PLAIN.search(body):
                    styles.append((start, start + len(body), "常体"))
            pos += len(p)
        if len(styles) < 3:
            return []
        n_polite = sum(1 for *_, s in styles if s == "敬体")
        n_plain = len(styles) - n_polite
        if n_polite == 0 or n_plain == 0:
            return []
        minority = "常体" if n_plain <= n_polite else "敬体"
        return [
            Finding(
                span=Span(max(s, e - 8), e, text[max(s, e - 8) : e]),
                error_type=ErrorType.STYLE_MIXING,
                verdict=Verdict.NORM_DIVERGENCE,
                confidence=1.0,
                message="敬体と常体が混在しています",
                detector=self.name,
                meta={"rule_id": "style.mix"},
            )
            for s, e, st in styles
            if st == minority
        ]


# ---------------------------------------------------------------------------
# (C) 汎用LLM（プロンプトのみ）
# ---------------------------------------------------------------------------

_LLM_SYSTEM = """あなたは日本語のビジネス文書の校正者です。
文化審議会答申「敬語の指針」を基準に、敬語の使い方について気づいた点を挙げてください。

重要な方針:
- 世代や場面で許容度が変わる「揺れ」は指摘しないでください。たとえば
  「お伺いする」「ご利用いただく」「ご持参ください」「お待ちしています」は
  指針が問題ないとしている、または習慣として定着していると認めている形です。
- 「させていただく」は指針が「許容度には個人差がある」としています。
  1〜2回程度の使用は指摘しないでください。
- 断定的に「間違い」と責めず、規範上の整理として述べてください。

出力は JSON のみ。次の形式に厳密に従ってください:
{"findings": [{"text": "誤りの箇所の原文そのまま", "type": "種別", "suggestion": "修正候補"}]}

「種別」は次のいずれかの英字キーを使ってください:
"""


class LLMBaseline(Baseline):
    """条件 (C)。汎用の大規模LLM（プロンプトのみ）。

    実証する主張: 「速度」と「過剰指摘の少なさ」。LLM は向きの誤りをある程度
    捕まえるが、CPU 即応にはならず、揺れへの過剰指摘も出やすい。
    公正を期すため、揺れを指摘しないよう**プロンプトでも同じ方針を伝える**。

    応答はディスクにキャッシュする。キャッシュがあれば API キー無しでも動く。
    """

    name = "llm"
    label = "(C) 汎用LLM"

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-5",
        api_key: Optional[str] = None,
        cache_dir: "str | Path | None" = None,
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        """プロバイダ・モデル・キャッシュ先を設定する。

        実証する主張: 「速度」。応答をキャッシュし、再実行のたびに API を呼ばない。
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.cache_dir = Path(cache_dir or (_REPO_ROOT / ".cache" / "llm"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Any = None

    # ------------------------------------------------------------------
    def available(self) -> bool:
        """API キーがあるか、キャッシュが1件でもあれば使える。

        実証する主張: 「速度」。API キーが無ければ測定不可として扱い、数字を捏造しない。
        """
        if self.api_key:
            try:
                import anthropic  # noqa: F401

                return True
            except ImportError:
                return any(self.cache_dir.glob("*.json"))
        return any(self.cache_dir.glob("*.json"))

    def unavailable_reason(self) -> str:
        """LLM 条件が使えない理由。

        実証する主張: 「根拠提示」。API キーが無いことを明記し、数字を捏造しない。
        """
        if not self.api_key:
            return (
                "ANTHROPIC_API_KEY が設定されておらず、キャッシュもありません。"
                "この条件は測定不可として扱います。"
            )
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return "anthropic SDK が入っていません（pip install anthropic）"
        return ""

    def _prompt(self) -> str:
        keys = "\n".join(
            f'  "{e.value}" = {ERROR_TYPE_JA[e]}'
            for e in ErrorType
            if e is not ErrorType.NONE
        )
        return _LLM_SYSTEM + keys

    def _cache_path(self, text: str, context: MailContext) -> Path:
        key = hashlib.sha256(
            f"{self.model}|{context.audience.value}|{context.writer_org}|"
            f"{context.recipient_org}|{text}".encode("utf-8")
        ).hexdigest()[:32]
        return self.cache_dir / f"{key}.json"

    # ------------------------------------------------------------------
    def check(self, text: str, context: MailContext) -> CheckResult:
        """本文1通を LLM に掛ける（応答はキャッシュする）。

        実証する主張: 「速度」。キャッシュ済みなら初回計測時の所要時間を再利用し、見かけ上速くしない。
        """
        cache = self._cache_path(text, context)
        t0 = time.perf_counter()
        cached = False
        if cache.exists():
            try:
                payload = json.loads(cache.read_text())
                cached = True
            except (OSError, json.JSONDecodeError):
                payload = None
        else:
            payload = None

        if payload is None:
            payload = self._call(text, context)
            if payload is None:
                return CheckResult(
                    text=text, context=context, findings=[], elapsed_ms=0.0,
                    engine=self.name,
                    meta={"unavailable_reason": self.unavailable_reason()},
                )
            try:
                cache.write_text(json.dumps(payload, ensure_ascii=False))
            except OSError:
                pass

        elapsed = (time.perf_counter() - t0) * 1000.0
        if cached:
            elapsed = float(payload.get("_elapsed_ms", elapsed))
        else:
            payload["_elapsed_ms"] = elapsed
            try:
                cache.write_text(json.dumps(payload, ensure_ascii=False))
            except OSError:
                pass

        return CheckResult(
            text=text,
            context=context,
            findings=self._to_findings(text, payload),
            elapsed_ms=elapsed,
            engine=self.name,
            meta={"model": self.model, "cached": cached},
        )

    def _call(self, text: str, context: MailContext) -> Optional[dict]:
        if not self.api_key:
            return None
        try:
            import anthropic
        except ImportError:
            return None
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.api_key, timeout=self.timeout
            )
        aud = {"external": "社外", "internal": "社内", "public": "不特定多数"}[
            context.audience.value
        ]
        user = (
            f"宛先: {aud}\n"
            f"自分側の組織: {context.writer_org}\n"
            f"相手側の組織: {context.recipient_org}\n"
            + (
                "本文に出る人物の立場:\n"
                + "\n".join(f"  {p.name}: {p.side.value}" for p in context.persons)
                + "\n"
                if context.persons
                else ""
            )
            + f"\n本文:\n{text}"
        )
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system=self._prompt(),
                    messages=[{"role": "user", "content": user}],
                )
                raw = "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                )
                return self._parse(raw)
            except Exception as exc:  # noqa: BLE001 — SDK の例外型に依存しない
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(delay)
                delay *= 2
        return None

    @staticmethod
    def _parse(raw: str) -> dict:
        raw = raw.strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {"findings": []}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"findings": []}

    def _to_findings(self, text: str, payload: dict) -> List[Finding]:
        out: List[Finding] = []
        for item in payload.get("findings", []) or []:
            if not isinstance(item, dict):
                continue
            frag = str(item.get("text", "")).strip()
            if not frag:
                continue
            idx = text.find(frag)
            if idx < 0:
                continue
            try:
                et = ErrorType(str(item.get("type", "none")))
            except ValueError:
                et = ErrorType.NONE
            from .types import Suggestion

            sug = str(item.get("suggestion", "")).strip()
            out.append(
                Finding(
                    span=Span(idx, idx + len(frag), frag),
                    error_type=et,
                    verdict=Verdict.NORM_DIVERGENCE,
                    confidence=1.0,
                    suggestions=(
                        (Suggestion(text=sug, generated_by="llm"),) if sug else ()
                    ),
                    message=str(item.get("reason", "")),
                    detector=self.name,
                    meta={"model": self.model},
                )
            )
        return out


ALL_BASELINES: Dict[str, type] = {
    "textlint": TextlintBaseline,
    "rule_baseline": RuleBaseline,
    "llm": LLMBaseline,
}
