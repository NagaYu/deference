#!/usr/bin/env python3
"""Gradio Space を組み立てて Hugging Face へ push する。

Space には学習済みの重み（1.1GB）を同梱せず、Hub のモデルリポジトリを参照する。
そのため Space に置くのは `app.py` と `deference` パッケージ、依存の宣言だけになる。

注意:
    Gradio Space の無料 CPU ホスティングには Hugging Face PRO が必要である
    （静的 Space は無料）。PRO でない場合、`create_repo` は HTTP 402 を返す。
    その場合は ``--static`` で、図と結果へのリンクを載せた静的ページを作れる。

実証する主張との対応:
    - 「向きの誤り検出」: Space の「並置比較」タブが、textlint が届かない箇所を
      そのまま見せる。だから Space では textlint も動くようにしておく
      （packages.txt で Node を入れ、初回に npm install する）。

使い方::

    python scripts/publish_space.py --repo-id NagaYu/deference
    python scripts/publish_space.py --repo-id NagaYu/deference --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent

SPACE_README = """---
title: Deference — Japanese Honorific Checker
emoji: 🎎
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: {sdk_version}
app_file: app.py
pinned: false
license: apache-2.0
models:
  - {model_repo}
datasets:
  - {dataset_repo}
short_description: The direction of deference in Japanese keigo, with the source
tags:
  - japanese
  - keigo
  - honorifics
  - grammatical-error-detection
---

# Deference

**The direction of deference, with the source to back it up.**

A rule-based Japanese linter says nothing about 「お持ちします」. As a string it is
fine. But **if the reader is the one carrying something, that form does not raise
them.**

This Space judges from *whose action it is*, *who it is directed to*, and *whether
the message goes inside or outside your company*, then cites the page of the
Council for Cultural Affairs' **Keigo no Shishin** (敬語の指針, 2007) the judgement
rests on — and shows the **textlint** result beside it.

On error types that need context, textlint and a surface-rule set both score
**0.0%**; this reaches **86–100%**. On types visible in the surface pattern the
rule set also reaches 100%, and no advantage is claimed there.

Expressions whose acceptability is genuinely divided are held apart as
**variation** and are not reported — the guidelines themselves warn against
treating usage uniformly by generation (Ch.1 Sec.2-2, p.8).

Output is **information, not a verdict on your Japanese**.

- Code and evaluation: <https://github.com/NagaYu/deference>
- Model: <https://huggingface.co/{model_repo}>
- Dataset: <https://huggingface.co/datasets/{dataset_repo}>

## Source

Adapted from the Council for Cultural Affairs' report *Keigo no Shishin*
(敬語の指針), Agency for Cultural Affairs, 2007.
<https://www.bunka.go.jp/seisaku/bunkashingikai/kokugo/hokoku/pdf/keigo_tosin.pdf>

Content on the Agency's site follows the
[MEXT website terms of use](https://www.mext.go.jp/b_menu/1351168.htm), stated to
be compatible with CC BY and permitting adaptation provided the source is
credited. Only short quotations needed as the basis for each note are kept here;
the report is not redistributed in full.
"""

REQUIREMENTS = """\
gradio>=5,<7
torch>=2.1,<3
transformers>=4.40,<6
sentencepiece>=0.2
huggingface_hub>=0.24
numpy>=1.24,<3
"""

#: Space に Node を入れて、textlint との並置比較を実際に動かす。
PACKAGES = "nodejs\nnpm\n"


def assemble(dest: Path, *, sdk_version: str, model_repo: str, dataset_repo: str) -> Path:
    """Space に上げる一式を ``dest`` に組み立てる。"""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    shutil.copytree(
        ROOT / "deference", dest / "deference",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy(ROOT / "app.py", dest / "app.py")
    (dest / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (dest / "packages.txt").write_text(PACKAGES, encoding="utf-8")
    (dest / "README.md").write_text(
        SPACE_README.format(
            sdk_version=sdk_version,
            model_repo=model_repo,
            dataset_repo=dataset_repo,
        ),
        encoding="utf-8",
    )
    # textlint の作業ディレクトリ。node_modules は Space 起動時に入れる。
    workdir = dest / ".baseline_textlint"
    workdir.mkdir()
    shutil.copy(ROOT / ".baseline_textlint" / "package.json", workdir / "package.json")
    return dest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-id", default="NagaYu/deference")
    ap.add_argument("--model-repo", default="NagaYu/deference-keigo")
    ap.add_argument("--dataset-repo", default="NagaYu/deference-keigo-corpus")
    ap.add_argument("--sdk-version", default="5.49.1")
    ap.add_argument("--out", default="", help="assembly directory (default: a temp dir)")
    ap.add_argument("--dry-run", action="store_true", help="assemble only, do not push")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)

    import tempfile

    dest = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="deference-space-"))
    assemble(
        dest,
        sdk_version=args.sdk_version,
        model_repo=args.model_repo,
        dataset_repo=args.dataset_repo,
    )
    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"assembled at {dest} ({total/1024:.0f} KB)")
    for f in sorted(dest.rglob("*")):
        if f.is_file():
            print("   ", f.relative_to(dest))
    if args.dry_run:
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed", file=sys.stderr)
        return 2

    api = HfApi()
    try:
        api.create_repo(
            args.repo_id, repo_type="space", space_sdk="gradio",
            private=args.private, exist_ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        if "402" in str(exc):
            print(
                "Hugging Face returned 402: hosting a Gradio Space on free "
                "cpu-basic requires a PRO subscription (static Spaces are free).\n"
                "  - Subscribe: https://huggingface.co/pro\n"
                "  - Or run it locally:  python app.py",
                file=sys.stderr,
            )
            return 2
        print(f"could not create the Space: {exc}", file=sys.stderr)
        return 2

    api.upload_folder(
        folder_path=str(dest), repo_id=args.repo_id, repo_type="space",
        commit_message="Deference: keigo checker with source citations",
    )
    print(f"pushed: https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
