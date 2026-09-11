"""pytest の共通フィクスチャ。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 候補生成の例外を握りつぶさない。import 漏れのような本物のバグを
# テストで表面化させるため（deference/correct.py の suggest を参照）。
os.environ.setdefault("DEFERENCE_DEBUG", "1")

from deference.correct import Corrector  # noqa: E402
from deference.detect import NormDetector  # noqa: E402
from deference.generate import Generator, default_context  # noqa: E402
from deference.inject import ErrorInjector  # noqa: E402
from deference.pipeline import Deference  # noqa: E402
from deference.types import Audience  # noqa: E402
from deference.variation import VariationSet  # noqa: E402

#: テストの規模。CI では小さく、手元では -k で広げられるようにする。
CORPUS_N = int(os.environ.get("DEFERENCE_TEST_CORPUS", "400"))


@pytest.fixture(scope="session")
def generator() -> Generator:
    return Generator(seed=0)


@pytest.fixture(scope="session")
def injector(generator: Generator) -> ErrorInjector:
    return ErrorInjector(seed=0, generator=generator)


@pytest.fixture(scope="session")
def corpus(generator: Generator):
    return generator.generate_corpus(CORPUS_N)


@pytest.fixture(scope="session")
def variation_set() -> VariationSet:
    return VariationSet()


@pytest.fixture(scope="session")
def detector() -> NormDetector:
    return NormDetector()


@pytest.fixture(scope="session")
def corrector() -> Corrector:
    return Corrector()


@pytest.fixture(scope="session")
def deference() -> Deference:
    return Deference(engine="norm")


@pytest.fixture()
def sample_context():
    return default_context(Audience.EXTERNAL)
