"""CLI の動作。`deference check mail.txt --audience external` がそのまま動くこと。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

MAIL = (
    "田中様\n\n"
    "いつもお世話になっております。\n"
    "弊社の佐藤社長が、そのようにおっしゃっておりました。\n"
    "当日は私が資料をお持ちになります。\n"
    "明日は都合により休まさせていただきます。\n"
    "よろしくお願いいたします。\n"
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "deference", *args],
        cwd=str(ROOT), capture_output=True, text=True,
    )


@pytest.fixture()
def mail_file(tmp_path) -> Path:
    p = tmp_path / "mail.txt"
    p.write_text(MAIL, encoding="utf-8")
    return p


def test_check_text_output_english_by_default(mail_file):
    """既定は英語。GitHub / Hugging Face 向けの公開言語に合わせる。"""
    proc = _run("check", str(mail_file), "--audience", "external", "--no-color")
    assert proc.returncode == 0, proc.stderr
    assert "Notes from the guidelines" in proc.stdout
    assert "Keigo no Shishin" in proc.stdout
    assert "bunka.go.jp" in proc.stdout
    # 原典からの引用は日本語のまま残す（英訳に置き換えない）
    assert "「" in proc.stdout


def test_check_text_output_japanese(mail_file):
    """--lang ja で日本語表示に切り替わること。"""
    proc = _run(
        "check", str(mail_file), "--audience", "external", "--no-color", "--lang", "ja"
    )
    assert proc.returncode == 0, proc.stderr
    assert "規範上の案内" in proc.stdout
    assert "敬語の指針" in proc.stdout


def test_check_json_output(mail_file):
    proc = _run("check", str(mail_file), "--audience", "external", "--format", "json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["text"] == MAIL
    assert payload["context"]["audience"] == "external"
    assert payload["reportable_count"] >= 1
    for f in payload["findings"]:
        assert f["span"]["text"] == MAIL[f["span"]["start"] : f["span"]["end"]]
        if f["verdict"] == "norm_divergence":
            assert f["citation"] is not None, "根拠のない指摘があります"
            assert f["message"]


def test_stdin(mail_file):
    proc = subprocess.run(
        [sys.executable, "-m", "deference", "check", "-", "--format", "json"],
        cwd=str(ROOT), input=MAIL, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["reportable_count"] >= 1


def test_person_flag_changes_result(mail_file):
    plain = _run("check", str(mail_file), "--format", "json")
    tagged = _run(
        "check", str(mail_file), "--format", "json",
        "--person", "佐藤:self_group", "--person", "田中:addressee",
    )
    assert plain.returncode == tagged.returncode == 0
    assert json.loads(tagged.stdout)["reportable_count"] >= 1


def test_explain_subcommand():
    proc = _run("explain", "uchi_sonkeigo", "--no-color")
    assert proc.returncode == 0, proc.stderr
    assert "uchi" in proc.stdout
    assert "Keigo no Shishin" in proc.stdout

    ja = _run("explain", "uchi_sonkeigo", "--no-color", "--lang", "ja")
    assert ja.returncode == 0, ja.stderr
    assert "身内に尊敬語" in ja.stdout


def test_forms_subcommand():
    proc = _run("forms", "言う", "--no-color")
    assert proc.returncode == 0, proc.stderr
    assert "おっしゃ" in proc.stdout
    assert "申し上げ" in proc.stdout


def test_unknown_verb_is_reported_kindly():
    proc = _run("forms", "ぬるぽする", "--no-color")
    assert proc.returncode == 2
    assert "Not in the verb lexicon" in proc.stderr

    ja = _run("forms", "ぬるぽする", "--no-color", "--lang", "ja")
    assert ja.returncode == 2
    assert "辞書にない動詞" in ja.stderr


def test_demo_subcommand():
    proc = _run("demo", "--no-color")
    assert proc.returncode == 0, proc.stderr
    assert "external" in proc.stdout and "internal" in proc.stdout
