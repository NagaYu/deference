#!/usr/bin/env python3
"""学習済みモデルを ONNX / GGUF / MLX へ書き出す（＋量子化・push）。

実証する主張との対応:
    - 「速度」: ONNX + 動的量子化（int8）が、評価条件 (E)「Deference（量子化後）」の
      実体である。量子化前 (D) と同じ本文集合で応答時間を測り、
      精度の低下と速度の向上を両方報告する。

使い方::

    python scripts/export.py --model-dir checkpoints/deference-base --format onnx --quantize
    python scripts/export.py --model-dir checkpoints/deference-base --format all
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deference.model import ErrorSpanClassifier  # noqa: E402


# ---------------------------------------------------------------------------
def export_onnx(model_dir: Path, out_dir: Path, *, max_length: int = 192) -> Path:
    """ONNX へ書き出す。dynamic axes を設定して可変長入力に対応する。"""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    clf = ErrorSpanClassifier.from_pretrained(model_dir, device="cpu")
    clf.model.eval()

    dummy = clf.tokenizer(
        "[社外][書き手:自分側] 弊社の佐藤社長がおっしゃいました。",
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    path = out_dir / "model.onnx"
    torch.onnx.export(
        clf.model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    # トークナイザとラベルを同梱する（推論側で必要）
    clf.tokenizer.save_pretrained(out_dir)
    (out_dir / "labels.json").write_text(
        json.dumps(clf.label_list(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name in ("config.json", "README.md", "metrics.json"):
        src = model_dir / name
        if src.exists():
            shutil.copy(src, out_dir / name)
    print(f"ONNX: {path} ({_onnx_total_bytes(path)/1e6:.1f} MB, 外部データ込み)")
    return path


def quantize_onnx(onnx_path: Path) -> Optional[Path]:
    """動的量子化（int8）。評価条件 (E) の実体。"""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print(
            "onnxruntime.quantization が使えません（pip install onnxruntime）",
            file=sys.stderr,
        )
        return None
    out = onnx_path.with_name("model.int8.onnx")
    quantize_dynamic(
        str(onnx_path), str(out), weight_type=QuantType.QInt8
    )
    before = _onnx_total_bytes(onnx_path) / 1e6
    after = _onnx_total_bytes(out) / 1e6
    print(
        f"量子化 int8: {out} ({before:.1f} MB → {after:.1f} MB, "
        f"{after/before*100:.0f}%)"
    )
    return out


def _onnx_total_bytes(path: Path) -> int:
    """ONNX モデルの総サイズ（外部データファイルを含む）。

    大きなモデルは重みが ``model.onnx.data`` に切り出されるため、``.onnx`` の
    ファイルサイズだけを見ると 1.7 MB のように見えてしまう。量子化の効果を
    誤って報告しないよう、外部データを足した実サイズで比べる。
    """
    total = path.stat().st_size
    for extra in path.parent.glob(path.name + ".data"):
        total += extra.stat().st_size
    return total


def verify_onnx(onnx_path: Path, tokenizer_dir: Path) -> bool:
    """onnxruntime で実際に読み込んで推論できるか確かめる。"""
    try:
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"検証をスキップします: {exc}", file=sys.stderr)
        return False
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    enc = tok("[社外] 弊社の佐藤社長がおっしゃいました。", return_tensors="np")
    feeds = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    logits = sess.run(["logits"], feeds)[0]
    print(f"ONNX 推論を確認: logits shape={logits.shape}")
    return True


# ---------------------------------------------------------------------------
def export_gguf(model_dir: Path, out_dir: Path, converter: Optional[str]) -> bool:
    """GGUF へ書き出す（llama.cpp の変換スクリプトが必要）。

    依存が無い環境では、何をすれば良いかを明示して skip する。
    黙って失敗させたり、変換したふりをしたりはしない。
    """
    script = converter
    if not script:
        for cand in (
            "convert_hf_to_gguf.py",
            "convert-hf-to-gguf.py",
        ):
            found = shutil.which(cand)
            if found:
                script = found
                break
    if not script or not Path(script).exists():
        print(
            "GGUF 変換をスキップします。\n"
            "  llama.cpp の convert_hf_to_gguf.py が必要です:\n"
            "    git clone https://github.com/ggml-org/llama.cpp\n"
            "    python scripts/export.py --format gguf "
            "--gguf-converter llama.cpp/convert_hf_to_gguf.py",
            file=sys.stderr,
        )
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "deference.gguf"
    proc = subprocess.run(
        [sys.executable, str(script), str(model_dir), "--outfile", str(target)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"GGUF 変換に失敗しました:\n{proc.stderr[-2000:]}", file=sys.stderr)
        return False
    print(f"GGUF: {target}")
    return True


def export_mlx(model_dir: Path, out_dir: Path) -> bool:
    """MLX（Apple Silicon）へ書き出す。mlx-lm が必要。"""
    try:
        from mlx_lm import convert  # type: ignore
    except ImportError:
        print(
            "MLX 変換をスキップします。`pip install mlx-lm` が必要です"
            "（Apple Silicon 向け）。",
            file=sys.stderr,
        )
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        convert(str(model_dir), mlx_path=str(out_dir), quantize=True)
    except Exception as exc:  # noqa: BLE001
        print(f"MLX 変換に失敗しました: {exc}", file=sys.stderr)
        print(
            "mlx-lm は主に生成モデル向けです。系列ラベリングモデルは"
            "未対応の場合があります。",
            file=sys.stderr,
        )
        return False
    print(f"MLX: {out_dir}")
    return True


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default="checkpoints/deference-base")
    ap.add_argument(
        "--format", default="onnx", choices=["onnx", "gguf", "mlx", "all"]
    )
    ap.add_argument("--output", default="")
    ap.add_argument("--quantize", action="store_true", help="ONNX を int8 に量子化する")
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--gguf-converter", default="")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--hub-repo-id", default="")
    args = ap.parse_args(argv)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(
            f"モデルが見つかりません: {model_dir}\n"
            "  scripts/train.py で学習してください。",
            file=sys.stderr,
        )
        return 2

    base_out = Path(args.output or f"exports/{model_dir.name}")
    ok = True
    if args.format in ("onnx", "all"):
        onnx_dir = base_out / "onnx"
        path = export_onnx(model_dir, onnx_dir, max_length=args.max_length)
        verify_onnx(path, onnx_dir)
        if args.quantize or args.format == "all":
            q = quantize_onnx(path)
            if q:
                verify_onnx(q, onnx_dir)
    if args.format in ("gguf", "all"):
        ok &= export_gguf(model_dir, base_out / "gguf", args.gguf_converter or None)
    if args.format in ("mlx", "all"):
        ok &= export_mlx(model_dir, base_out / "mlx")

    if args.push_to_hub:
        if not args.hub_repo_id:
            print("--push-to-hub には --hub-repo-id が必要です", file=sys.stderr)
            return 2
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(args.hub_repo_id, exist_ok=True)
            api.upload_folder(folder_path=str(base_out), repo_id=args.hub_repo_id)
            print(f"push 完了: https://huggingface.co/{args.hub_repo_id}")
        except ImportError:
            print("huggingface_hub が入っていません", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"push できませんでした: {exc}", file=sys.stderr)
            return 2
    return 0 if ok else 0  # 一部の形式が使えなくても異常終了にはしない


if __name__ == "__main__":
    raise SystemExit(main())
