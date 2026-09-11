"""誤りスパンと誤り種別を予測する系列ラベリングモデル（BIO × ErrorType）。

このモジュールは Deference の「ニューラル側」である。規則側（norms.py を引く
検出器）が取りこぼす言い回しを拾うために、文脈プレフィックス付きの本文を
トークン分類で読む。**規範そのもの（語形・分類・引用）はここには一切持たない。**
語形の判定と根拠付与は norms.py と後段の pipeline が担う。

実証する主張との対応:
    - 「向きの誤り検出」: :meth:`ErrorSpanClassifier.encode` は本文の前に
      〈宛先が社内か社外か〉〈誰が自分側で誰が相手側か〉を書いたプレフィックスを
      連結する。表層文字列だけを見る校正器はこの情報を持たないため、
      「田中がおっしゃいました」の適否を原理的に決められない。文脈を入力に
      含めることが、向き系の誤り（DIRECTION_SWAP / UCHI_SONKEIGO /
      SELF_SONKEIGO / DEFERENCE_INVERSION）を学習可能にする条件である。
    - 「過剰指摘の少なさ」: :meth:`ErrorSpanClassifier.encode` は誤りが
      1件も無い入力に対して全トークン ``"O"`` のラベル列を返す。揺れ
      （:class:`~deference.types.Verdict` の ``VARIATION``）のサンプルを
      負例として与える経路がここで開く。推論側では ``threshold`` で
      低確信の指摘を落とせる。
    - 「速度」: torch / transformers を **遅延 import** する。
      ``import deference.model`` だけでは torch を読み込まないため、CLI の
      起動時間が規則側だけの経路で犠牲にならない。推論の実測値は
      :attr:`ErrorSpanClassifier.last_elapsed_ms` に残す。
    - 「根拠提示」: このモジュールは ``citation`` を埋めない。根拠は必ず
      norms.py 由来でなければならないため、ニューラル側が根拠を捏造できない
      ように構造で禁じている（:meth:`predict` の返す Finding は
      ``citation is None``）。

依存の方針:
    torch / transformers は関数内で import する。モデルの重みを持たない環境でも
    :meth:`ErrorSpanClassifier.tiny_for_test` はネットワーク無しで動く
    （文字単位の fast tokenizer をその場で組み立てるため）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from .types import (
    AUDIENCE_JA,
    ERROR_TYPE_JA,
    ErrorType,
    Finding,
    InjectedError,
    MailContext,
    Span,
    Verdict,
)

if TYPE_CHECKING:  # 型注釈のためだけの import（実行時には評価されない）
    import torch as _torch_types  # noqa: F401

__all__ = [
    "DETECTOR_NAME",
    "OUTSIDE_LABEL",
    "ErrorSpanClassifier",
    "label_list",
    "labeled_error_types",
]


#: :class:`~deference.types.Finding` の ``detector`` に入れる名前。
#: 規則側（"deference-rule"）と区別できるようにしておく。
DETECTOR_NAME = "deference-neural"

#: BIO の外側ラベル。
OUTSIDE_LABEL = "O"

#: 損失計算から除外するラベル id（transformers の慣習に合わせる）。
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# ラベル体系
# ---------------------------------------------------------------------------


def labeled_error_types() -> List[ErrorType]:
    """ラベルを持つ誤り種別（``ErrorType.NONE`` を除く全種別）を宣言順で返す。

    実証する主張: 「向きの誤り検出」。向き系の誤りを含む全種別が独立の
    ラベルを持つことをここで保証する。種別を潰して1つの「敬語エラー」に
    まとめてしまうと、向きの誤りだけを取り出した評価ができなくなる。
    """
    return [t for t in ErrorType if t is not ErrorType.NONE]


def label_list() -> List[str]:
    """BIO × ErrorType のラベル集合を返す（モジュール関数版）。

    実証する主張: 「向きの誤り検出」。``"O"`` + 各誤り種別の ``B-`` / ``I-`` で、
    どの文字範囲がどの種別かを同時に予測できる形にする。

    Returns:
        ``["O", "B-double_keigo", "I-double_keigo", ...]`` の順序固定リスト。
        順序は :class:`~deference.types.ErrorType` の宣言順に従うので、
        保存済みチェックポイントとの id 対応が壊れない。
    """
    labels = [OUTSIDE_LABEL]
    for error_type in labeled_error_types():
        labels.append(f"B-{error_type.value}")
        labels.append(f"I-{error_type.value}")
    return labels


# ---------------------------------------------------------------------------
# 遅延 import ヘルパ
# ---------------------------------------------------------------------------


def _torch():
    """torch を遅延 import する。

    実証する主張: 「速度」。``import deference.model`` の時点で torch を
    読み込まないことで、規則側だけを使う CLI 起動を数百 ms 単位で速く保つ。
    """
    import torch  # noqa: PLC0415

    return torch


def _transformers():
    """transformers を遅延 import する。

    実証する主張: 「速度」。理由は :func:`_torch` と同じ。
    """
    import transformers  # noqa: PLC0415

    return transformers


# ---------------------------------------------------------------------------
# ネットワーク不要の文字単位 tokenizer（テスト・オフライン用）
# ---------------------------------------------------------------------------

# 【norms.py に置かない理由】
# ここに並ぶのは「敬語の規範」ではなく、テスト用 tokenizer の語彙範囲という
# 純粋に実装都合のパラメータである。規範データベースに混ぜると norms.py の
# 「唯一の正しさの源」という位置づけが濁るため、本モジュールのローカル定数に置く。
_TINY_VOCAB_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x0020, 0x007E),  # ASCII 印字可能
    (0x3000, 0x303F),  # 和文句読点・括弧
    (0x3040, 0x309F),  # ひらがな
    (0x30A0, 0x30FF),  # カタカナ
    (0xFF01, 0xFF5E),  # 全角英数・記号
)

_TINY_SPECIALS: Tuple[str, ...] = ("<s>", "<pad>", "</s>", "<unk>")


def _build_char_tokenizer(max_length: int):
    """1文字＝1トークンの fast tokenizer を組み立てる（ネットワーク不要）。

    実証する主張: 「速度」と「向きの誤り検出」の検証可能性。
    offset mapping を持つ fast tokenizer をオフラインで用意できるので、
    ネットワークの無い CI でも「文字オフセットの往復」テストが回る。
    語彙に無い漢字は ``<unk>`` になるが、**オフセットは常に1文字単位で正しい**ため、
    スパン復元の検証にはこれで十分である。

    Args:
        max_length: 打ち切り長。tokenizer 側の既定値として設定する。

    Returns:
        ``transformers.PreTrainedTokenizerFast``。
    """
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors  # noqa: PLC0415

    tf = _transformers()

    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(_TINY_SPECIALS)}
    for lo, hi in _TINY_VOCAB_RANGES:
        for code in range(lo, hi + 1):
            vocab.setdefault(chr(code), len(vocab))

    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    # 「任意の1文字」を isolated で切ることで、改行や空白も含め文字単位に分割する。
    # Oniguruma は (?s) を解さないので [\s\S] で代用する。
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> </s> $B </s>",
        special_tokens=[("<s>", vocab["<s>"]), ("</s>", vocab["</s>"])],
    )
    return tf.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        model_max_length=max_length,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<s>",
        sep_token="</s>",
    )


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------


class ErrorSpanClassifier:
    """文脈込みで誤りスパンと誤り種別を予測するトークン分類器。

    既定の骨格は ``xlm-roberta-base``（約 0.28B パラメータ）。
    fugashi / MeCab を必要としない SentencePiece 系であること、fast tokenizer が
    あり offset mapping を返せることが選定条件である（BertJapaneseTokenizer 系は
    形態素解析器を要求するため使わない）。

    実証する主張:
        - 「向きの誤り検出」: 入力に :class:`~deference.types.MailContext` を
          プレフィックスとして畳み込むので、同じ表層文字列でも宛先や人物の
          立場によって別の予測を出せる。
        - 「過剰指摘の少なさ」: :meth:`predict` の ``threshold`` と、
          揺れサンプルを ``"O"`` として符号化できる :meth:`encode` の設計により、
          「揺れを誤りにしない」方針を学習と推論の両方で貫ける。
        - 「速度」: CPU 前提。torch は遅延 import。推論時間は
          :attr:`last_elapsed_ms` に記録する。
        - 「根拠提示」: 返す :class:`~deference.types.Finding` は
          ``citation=None`` / ``suggestions=()``。根拠と修正候補は必ず
          norms.py を引く後段（cite / correct）が埋める。
    """

    #: 既定の骨格モデル。SentencePiece ベースで fast tokenizer を持ち、
    #: 約 278M パラメータ（目標 0.1〜0.3B の範囲内）。
    DEFAULT_MODEL_ID = "xlm-roberta-base"

    #: ``model_id`` にこの値を渡すと、ネットワーク不要の極小モデルを組み立てる。
    TINY_MODEL_ID = "tiny"

    # -- 構築 ---------------------------------------------------------------

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str = "cpu",
        max_length: int = 512,
        num_labels: int | None = None,
    ) -> None:
        """骨格モデルまたは保存済みディレクトリから分類器を組み立てる。

        実証する主張: 「速度」。torch / transformers はこの中で初めて import
        される。未学習（トークン分類ヘッドがランダム初期化）の状態でも
        :meth:`predict` が例外なく動くことを要件とする。

        Args:
            model_id: Hugging Face のモデル ID かローカルディレクトリ。
                :data:`TINY_MODEL_ID` を渡すと :meth:`tiny_for_test` に委譲する。
            device: ``"cpu"`` / ``"cuda"`` / ``"mps"``。既定は CPU。
            max_length: 符号化の打ち切り長（サブワード数）。
            num_labels: ラベル数の明示指定。既存チェックポイントのラベル体系が
                :func:`label_list` と異なるときの逃げ道。既定 ``None`` は
                :func:`label_list` の長さを使う。
        """
        if model_id == self.TINY_MODEL_ID:
            tokenizer, model = self._build_tiny_backend(max_length=max_length)
        else:
            tokenizer, model = self._load_backend(model_id, num_labels=num_labels)
        self._setup(model_id, tokenizer, model, device=device, max_length=max_length)

    def _setup(
        self,
        model_id: str,
        tokenizer: Any,
        model: Any,
        *,
        device: str,
        max_length: int,
    ) -> None:
        """内部状態を確定させる（``__init__`` と ``tiny_for_test`` の共通処理）。"""
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(
                "offset mapping を返す fast tokenizer が必要です。"
                f"{model_id!r} の tokenizer は fast ではありません。"
                "SentencePiece ベースで fast 実装のあるモデル"
                "（既定: xlm-roberta-base）を指定してください。"
            )
        self.model_id = model_id
        self.max_length = int(max_length)
        self.device = device
        self._tokenizer = tokenizer
        self._model = model.to(device)
        self._model.eval()
        raw_id2label = getattr(self._model.config, "id2label", None) or {}
        self.id2label: Dict[int, str] = {int(k): str(v) for k, v in raw_id2label.items()}
        if not self.id2label:
            self.id2label = dict(enumerate(label_list()))
        self.label2id: Dict[str, int] = {v: k for k, v in self.id2label.items()}
        #: 直近の :meth:`predict_batch` に要した時間（ミリ秒）。「速度」の実測用。
        self.last_elapsed_ms: float = 0.0

    @classmethod
    def tokenizer_only(
        cls,
        directory: "str | Path",
        *,
        max_length: int = 192,
    ) -> "ErrorSpanClassifier":
        """重みを読まず、符号化と復号だけができるインスタンスを作る。

        ONNX / GGUF など外部ランタイムで推論するとき、PyTorch の重みまで
        読み込んでしまうと「軽くした」という主張と実測が食い違う。
        トークナイザとラベル表だけを載せた器を返し、:meth:`encode` と
        ``_decode_findings`` を共有できるようにする。

        実証する主張: 「速度」。量子化後の条件を、量子化前の重みを一切
        読み込まずに測るための入口である。
        """
        directory = Path(directory)
        tf = _transformers()
        tokenizer = tf.AutoTokenizer.from_pretrained(str(directory), use_fast=True)
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("offset mapping を返す fast tokenizer が必要です。")

        labels_path = directory / "labels.json"
        if labels_path.exists():
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
        else:
            labels = label_list()

        self = cls.__new__(cls)
        self.model_id = str(directory)
        self.max_length = int(max_length)
        self.device = "cpu"
        self._tokenizer = tokenizer
        self._model = None
        self.id2label = dict(enumerate(labels))
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.last_elapsed_ms = 0.0
        return self

    @classmethod
    def _load_backend(cls, model_id: str, *, num_labels: int | None) -> Tuple[Any, Any]:
        """tokenizer と token classification モデルを読み込む。"""
        tf = _transformers()
        labels = label_list()
        n_labels = len(labels) if num_labels is None else int(num_labels)
        tokenizer = tf.AutoTokenizer.from_pretrained(model_id, use_fast=True)
        config_kwargs: Dict[str, Any] = {"num_labels": n_labels}
        if n_labels == len(labels):
            config_kwargs["id2label"] = dict(enumerate(labels))
            config_kwargs["label2id"] = {lab: i for i, lab in enumerate(labels)}
        config = tf.AutoConfig.from_pretrained(model_id, **config_kwargs)
        model = tf.AutoModelForTokenClassification.from_pretrained(
            model_id, config=config, ignore_mismatched_sizes=True
        )
        return tokenizer, model

    @classmethod
    def _build_tiny_backend(
        cls,
        *,
        max_length: int,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        intermediate_size: int = 128,
    ) -> Tuple[Any, Any]:
        """ネットワーク不要の極小 backend を組み立てる。"""
        tf = _transformers()
        tokenizer = _build_char_tokenizer(max_length)
        labels = label_list()
        config = tf.XLMRobertaConfig(
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            # RoBERTa 系は position_ids を pad_token_id+1 から数えるので余裕を持たせる
            max_position_embeddings=max_length + 8,
            type_vocab_size=1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_labels=len(labels),
            id2label=dict(enumerate(labels)),
            label2id={lab: i for i, lab in enumerate(labels)},
        )
        model = tf.XLMRobertaForTokenClassification(config)
        return tokenizer, model

    @classmethod
    def tiny_for_test(
        cls,
        *,
        device: str = "cpu",
        max_length: int = 256,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
    ) -> "ErrorSpanClassifier":
        """ネットワーク不要の極小モデルを作る（pytest 用）。

        実証する主張: 「速度」。hidden 64 / layers 2 / heads 2 の構成と
        文字単位 tokenizer により、ネットワークもモデル重みも無い環境で
        符号化・推論の経路すべてを秒未満で検証できる。オフセット往復や
        ラベル整合のテストはこの経路で回す。

        Args:
            device: 配置先。既定 CPU。
            max_length: 打ち切り長。
            hidden_size: 隠れ次元。
            num_hidden_layers: 層数。
            num_attention_heads: ヘッド数。

        Returns:
            ランダム初期化された :class:`ErrorSpanClassifier`。
            予測内容に意味は無いが、例外なく :meth:`predict` が動く。
        """
        tokenizer, model = cls._build_tiny_backend(
            max_length=max_length,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
        )
        obj = cls.__new__(cls)
        obj._setup(cls.TINY_MODEL_ID, tokenizer, model, device=device, max_length=max_length)
        return obj

    @classmethod
    def from_pretrained(
        cls, path: str | Path, *, device: str = "cpu"
    ) -> "ErrorSpanClassifier":
        """:meth:`save` で書き出したディレクトリから復元する。

        実証する主張: 「速度」。``max_length`` などの推論条件を
        ``deference_model.json`` に残しておき、学習時と推論時で符号化条件が
        ずれないようにする（ずれるとスパンの位置がずれ、評価が壊れる）。

        Args:
            path: :meth:`save` の出力ディレクトリ。
            device: 配置先。

        Returns:
            復元された :class:`ErrorSpanClassifier`。
        """
        directory = Path(path)
        meta_path = directory / "deference_model.json"
        max_length = 512
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            max_length = int(meta.get("max_length", max_length))
        elif not directory.exists():
            # ローカルに無ければ Hugging Face Hub のリポジトリ ID とみなす。
            # max_length を既定値のままにすると学習時と符号化条件がずれ、
            # スパンの位置が合わなくなるので、メタ情報だけ先に取りに行く。
            try:
                from huggingface_hub import hf_hub_download

                meta_file = hf_hub_download(
                    repo_id=str(path), filename="deference_model.json"
                )
                meta = json.loads(Path(meta_file).read_text(encoding="utf-8"))
                max_length = int(meta.get("max_length", max_length))
            except Exception:  # noqa: BLE001 - Hub が使えない環境でも落とさない
                pass
        return cls(str(path), device=device, max_length=max_length)

    def save(self, path: str | Path) -> None:
        """モデル・tokenizer・符号化条件をディレクトリに保存する。

        実証する主張: 「根拠提示」。``deference_model.json`` にラベル体系と
        文脈プレフィックスの仕様を残すので、あとから「この予測がどの
        ラベル体系・どの文脈表現の下で出たか」を辿れる。

        Args:
            path: 出力ディレクトリ。無ければ作る。
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(directory)
        self._tokenizer.save_pretrained(directory)
        meta = {
            "model_id": self.model_id,
            "max_length": self.max_length,
            "labels": [self.id2label[i] for i in sorted(self.id2label)],
            "detector": DETECTOR_NAME,
            "context_prefix_example": self.context_prefix(MailContext()),
            "note": (
                "本文の文字オフセットは prefix_len を差し引いて復元する。"
                "citation と suggestions はこのモデルでは埋めない（norms.py 由来のみ）。"
            ),
        }
        (directory / "deference_model.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # -- ラベル -------------------------------------------------------------

    @staticmethod
    def label_list() -> List[str]:
        """BIO × ErrorType のラベル集合（``"O"`` + ``B-<type>`` + ``I-<type>``）。

        実証する主張: 「向きの誤り検出」。誤り種別をラベルに畳み込むので、
        スパン検出と同時に「どの向きの誤りか」を出力できる。
        種別を1つに潰さないことが、向き系だけを切り出した評価の前提である。

        Returns:
            長さ ``1 + 2 * (len(ErrorType) - 1)`` の順序固定リスト。
        """
        return label_list()

    # -- 文脈プレフィックス --------------------------------------------------

    @staticmethod
    def context_prefix(context: MailContext) -> str:
        """文脈を短いプレフィックス文字列に落とす。

        実証する主張: 「向きの誤り検出」。指針 第2章第1 の分類は
        〈自分側 / 相手側〉の区別を前提にしており、文字列だけでは決まらない。
        ここで宛先区分と人物の立場を本文の前に明示することで、
        「田中がおっしゃいました」が身内尊敬（UCHI_SONKEIGO）になるか
        妥当な尊敬語になるかをモデルが区別できるようになる。

        Args:
            context: メール1通ぶんのメタ情報。

        Returns:
            例: ``"[社外][書き手:自分側][相手:貴社][自分側:山田/田中][相手側:佐藤] "``。
            **末尾に半角空白を1つ含む**。この文字列長が ``prefix_len`` になる。
        """
        audience_ja = AUDIENCE_JA.get(context.audience, str(context.audience))
        recipient_side = context.recipient_org or "相手側"

        self_names: List[str] = []
        other_names: List[str] = []
        if context.writer_name:
            self_names.append(context.writer_name)
        if context.recipient_name:
            other_names.append(context.recipient_name)
        for person in context.persons:
            if not person.name:
                continue
            if person.side.is_self_side and person.name not in self_names:
                self_names.append(person.name)
            elif person.side.is_other_side and person.name not in other_names:
                other_names.append(person.name)

        parts = [f"[{audience_ja}]", "[書き手:自分側]", f"[相手:{recipient_side}]"]
        if self_names:
            parts.append(f"[自分側:{'/'.join(self_names)}]")
        if other_names:
            parts.append(f"[相手側:{'/'.join(other_names)}]")
        return "".join(parts) + " "

    # -- 符号化 -------------------------------------------------------------

    def encode(
        self,
        text: str,
        context: MailContext,
        *,
        errors: Sequence[InjectedError] = (),
    ) -> Dict[str, Any]:
        """文脈をプレフィックスとして与えつつ、本文の文字オフセットを保つ符号化。

        文脈は ``"[社外][書き手:自分側][相手:貴社] "`` のような短い文字列にして
        本文と連結する。offset_mapping は連結後の文字列基準なので、
        戻り値の ``prefix_len`` を差し引けば本文基準のオフセットに戻る。

        実証する主張:
            - 「向きの誤り検出」: 文脈を捨てずに入力へ畳み込む。
            - 「過剰指摘の少なさ」: ``errors`` が空でも ``labels`` を全て
              ``"O"`` として返す。揺れ（Verdict.VARIATION）のサンプルは
              誤りを1件も渡さずにここへ流せば、そのまま負例になる。
              ``error_type`` が ``ErrorType.NONE`` の項目も ``"O"`` に落ちる。

        Args:
            text: 本文（プレフィックスを含まない）。
            context: メールのメタ情報。
            errors: 注入済みの誤り台帳。スパンは**本文基準**であること。

        Returns:
            以下のキーを持つ辞書。

            - ``input_ids`` / ``attention_mask``: 通常の符号化結果
            - ``offset_mapping``: **連結後の文字列**基準の (start, end)
            - ``special_tokens_mask``: 特殊トークンの位置
            - ``prefix_len``: プレフィックスの文字数。本文基準に戻すのに使う
            - ``prefix``: プレフィックス文字列そのもの
            - ``labels``: BIO ラベル id 列。特殊トークンとプレフィックス部分は
              ``-100``（損失から除外）
        """
        prefix = self.context_prefix(context)
        encoded = self._tokenizer(
            prefix + text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        out: Dict[str, Any] = {
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
            "offset_mapping": [tuple(o) for o in encoded["offset_mapping"]],
            "special_tokens_mask": list(encoded["special_tokens_mask"]),
            "prefix_len": len(prefix),
            "prefix": prefix,
        }
        out["labels"] = self._align_labels(
            out["offset_mapping"], out["special_tokens_mask"], len(prefix), errors
        )
        return out

    def _align_labels(
        self,
        offsets: Sequence[Tuple[int, int]],
        special_tokens_mask: Sequence[int],
        prefix_len: int,
        errors: Sequence[InjectedError],
    ) -> List[int]:
        """文字スパンの誤り台帳をトークン単位の BIO ラベル列に変換する。

        実証する主張: 「過剰指摘の少なさ」。``ErrorType.NONE`` の項目と、
        長さ 0 のスパンは黙って ``"O"`` に落とす。揺れの台帳をそのまま
        流し込んでも偽の正例にならない。
        """
        outside = self.label2id.get(OUTSIDE_LABEL, 0)
        usable = [
            e
            for e in errors
            if e.error_type is not ErrorType.NONE and e.span.end > e.span.start
        ]
        started = [False] * len(usable)
        labels: List[int] = []
        for (start, end), is_special in zip(offsets, special_tokens_mask):
            if is_special:
                labels.append(IGNORE_INDEX)
                continue
            if end <= prefix_len:  # プレフィックス側のトークンは学習対象外
                labels.append(IGNORE_INDEX)
                continue
            body_start = max(0, start - prefix_len)
            body_end = end - prefix_len
            hit = -1
            for i, err in enumerate(usable):
                if body_start < err.span.end and err.span.start < body_end:
                    hit = i
                    break
            if hit < 0:
                labels.append(outside)
                continue
            tag = "I" if started[hit] else "B"
            started[hit] = True
            labels.append(
                self.label2id.get(f"{tag}-{usable[hit].error_type.value}", outside)
            )
        return labels

    # -- 推論 ---------------------------------------------------------------

    def predict(
        self, text: str, context: MailContext, *, threshold: float = 0.5
    ) -> List[Finding]:
        """本文全体を受け取り :class:`~deference.types.Finding` のリストを返す。

        実証する主張:
            - 「向きの誤り検出」: ``context`` を入力に含めるので、同じ本文でも
              宛先や人物の立場が変われば別の予測になりうる。
            - 「過剰指摘の少なさ」: ``threshold`` 未満の確信度のトークンは
              ``"O"`` 扱いにして落とす。既定 0.5。
            - 「根拠提示」: 返す Finding は ``citation=None`` /
              ``suggestions=()``。根拠と修正候補は norms.py を引く後段が埋める。
              ニューラル側が根拠を作らないことを構造で保証する。
            - 「速度」: 所要時間を :attr:`last_elapsed_ms` に記録する。

        Args:
            text: 本文。
            context: メールのメタ情報。
            threshold: softmax 確率のしきい値。

        Returns:
            ``Finding`` のリスト。``span`` は**本文基準の文字オフセット**で、
            ``text[span.start:span.end] == span.text`` が成り立つ。
            ``detector`` は :data:`DETECTOR_NAME`。
        """
        return self.predict_batch([text], [context], threshold=threshold)[0]

    def predict_batch(
        self,
        texts: Sequence[str],
        contexts: Sequence[MailContext],
        *,
        threshold: float = 0.5,
    ) -> List[List[Finding]]:
        """複数本文をまとめて推論する。

        実証する主張: 「速度」。CPU での実測を安定させるため、パディングを
        1回にまとめて行列演算の回数を減らす。所要時間は
        :attr:`last_elapsed_ms` に入る。

        Args:
            texts: 本文の並び。
            contexts: 各本文に対応する文脈。長さは ``texts`` と一致すること。
            threshold: softmax 確率のしきい値。

        Returns:
            本文ごとの ``Finding`` リスト（入力順）。
        """
        texts = list(texts)
        contexts = list(contexts)
        if len(texts) != len(contexts):
            raise ValueError(
                f"texts と contexts の長さが違います: {len(texts)} != {len(contexts)}"
            )
        if not texts:
            self.last_elapsed_ms = 0.0
            return []

        torch = _torch()
        started = time.perf_counter()
        encodings = [self.encode(t, c) for t, c in zip(texts, contexts)]
        features = [
            {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}
            for e in encodings
        ]
        batch = self._tokenizer.pad(features, padding=True, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        self._model.eval()
        with torch.no_grad():
            logits = self._model(**batch).logits
        probs = torch.softmax(logits.float(), dim=-1)
        confidences, predictions = probs.max(dim=-1)

        results: List[List[Finding]] = []
        for i, encoding in enumerate(encodings):
            length = len(encoding["input_ids"])
            results.append(
                self._decode_findings(
                    texts[i],
                    encoding,
                    predictions[i][:length].tolist(),
                    confidences[i][:length].tolist(),
                    threshold,
                )
            )
        self.last_elapsed_ms = (time.perf_counter() - started) * 1000.0
        return results

    def _decode_findings(
        self,
        text: str,
        encoding: Dict[str, Any],
        predictions: Sequence[int],
        confidences: Sequence[float],
        threshold: float,
    ) -> List[Finding]:
        """トークン単位の予測を本文基準の Finding に畳み戻す。

        実証する主張: 「向きの誤り検出」の評価可能性。``prefix_len`` を
        差し引いて本文基準に戻すことで、注入時のスパンと厳密に突き合わせられる。
        """
        prefix_len = int(encoding["prefix_len"])
        offsets = encoding["offset_mapping"]
        specials = encoding["special_tokens_mask"]
        findings: List[Finding] = []
        current: Optional[Dict[str, Any]] = None

        def flush() -> None:
            """現在のトークン列を Finding に確定する。

        実証する主張: 「向きの誤り検出」の評価可能性。連続する同種ラベルを1つのスパンに畳み、注入位置と突き合わせられる形にする。
        """
            nonlocal current
            if current is None:
                return
            span = self._trim_span(text, current["start"], current["end"])
            if span is not None:
                findings.append(self._make_finding(span, current["type"], current["confs"]))
            current = None

        for index, (offset, is_special) in enumerate(zip(offsets, specials)):
            if is_special or index >= len(predictions):
                flush()
                continue
            label = self.id2label.get(int(predictions[index]), OUTSIDE_LABEL)
            score = float(confidences[index])
            if label == OUTSIDE_LABEL or score < threshold:
                flush()
                continue
            tag, _, type_value = label.partition("-")
            try:
                error_type = ErrorType(type_value)
            except ValueError:  # 未知のラベル体系は黙って無視する
                flush()
                continue
            body_end = offset[1] - prefix_len
            if body_end <= 0:  # プレフィックス内のトークン
                flush()
                continue
            body_start = max(0, offset[0] - prefix_len)
            if current is not None and tag == "I" and current["type"] is error_type:
                current["end"] = body_end
                current["confs"].append(score)
            else:
                flush()
                current = {
                    "type": error_type,
                    "start": body_start,
                    "end": body_end,
                    "confs": [score],
                }
        flush()
        return findings

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> Optional[Span]:
        """スパンを本文の範囲に収め、前後の空白を落とす。

        SentencePiece 系の offset には直前の空白が含まれることがあるため、
        提示スパンが不自然にならないよう整える。
        """
        start = max(0, min(start, len(text)))
        end = max(0, min(end, len(text)))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            return None
        return Span(start, end, text[start:end])

    def _make_finding(
        self, span: Span, error_type: ErrorType, confidences: Sequence[float]
    ) -> Finding:
        """1件の Finding を組み立てる。

        実証する主張: 「根拠提示」と「利用者の日本語を否定しない」方針。
        ``message`` は「規範上はこう案内されている」という情報提示に留め、
        断定的な否定語を使わない。根拠そのもの（citation）はここでは空にし、
        norms.py を引く後段に委ねる。
        """
        score = sum(confidences) / len(confidences) if confidences else 0.0
        name = ERROR_TYPE_JA.get(error_type, error_type.value)
        message = (
            f"「{span.text}」は{name}に当たりうる箇所として検出されました。"
            "規範上の扱いと根拠は、指針の該当箇所と照らして確認できます。"
        )
        return Finding(
            span=span,
            error_type=error_type,
            verdict=Verdict.NORM_DIVERGENCE,
            confidence=round(float(score), 4),
            suggestions=(),
            citation=None,  # 根拠は norms.py 由来のみ。ここでは埋めない。
            message=message,
            detector=DETECTOR_NAME,
            meta={"model_id": self.model_id, "threshold_applied": True},
        )

    # -- 補助 ---------------------------------------------------------------

    def parameter_count(self) -> int:
        """パラメータ総数を返す。

        実証する主張: 「速度」。目標サイズ 0.1〜0.3B に収まっていることを
        テストと図表で数値として示すために使う。
        """
        return sum(p.numel() for p in self._model.parameters())

    @property
    def model(self) -> Any:
        """内部の transformers モデル（学習スクリプトが直接触るための口）。

        実証する主張: 「速度」。重みへのアクセスをここに集約し、遅延読み込みを崩さない。
        """
        return self._model

    @property
    def tokenizer(self) -> Any:
        """内部の tokenizer（データコレータ構築のための口）。

        実証する主張: 「速度」。トークナイザだけを使う経路（ONNX 推論など）で重みを読まずに済む。
        """
        return self._tokenizer

    def __repr__(self) -> str:  # pragma: no cover - 表示用
        """表示用。

        実証する主張: 「速度」。読み込んだ backbone とラベル数を一目で確かめられるようにする。
        """
        return (
            f"ErrorSpanClassifier(model_id={self.model_id!r}, "
            f"device={self.device!r}, max_length={self.max_length}, "
            f"num_labels={len(self.id2label)})"
        )
