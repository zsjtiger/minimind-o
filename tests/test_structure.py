import importlib
import os
import pathlib

import pytest

from baize_v import paths

ROOT = pathlib.Path(__file__).resolve().parent.parent

MODULES = [
    "baize_v.config", "baize_v.paths",
    "baize_v.models", "baize_v.models.minimind", "baize_v.models.omni", "baize_v.models.talker",
    "baize_v.models.projectors", "baize_v.models.encoders",
    "baize_v.data", "baize_v.data.dataset", "baize_v.data.chat", "baize_v.data.augment",
    "baize_v.data.labels", "baize_v.data.collate",
    "baize_v.train.dist", "baize_v.train.schedule", "baize_v.train.init", "baize_v.train.checkpoint",
    "baize_v.train.losses", "baize_v.train.loop",
    "baize_v.inference.realtime", "baize_v.inference.loader",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("path", [paths.TOKENIZER_DIR, paths.VAD_PATH, paths.SPEAKER_DIR, paths.EVAL_AUDIO_DIR])
def test_bundled_assets_exist(path):
    assert os.path.exists(path), path


def test_paths_are_repo_relative():
    assert paths.ROOT_DIR == str(ROOT)
    assert paths.OUT_DIR == os.path.join(str(ROOT), "out")


def test_no_legacy_imports_left():
    legacy = ("from model.", "import model.", "from trainer.", "from dataset.", "webui/", "../model/", "./model/")
    hits = []
    for py in ROOT.rglob("*.py"):
        if any(part in {".git", "tests"} for part in py.parts):
            continue
        text = py.read_text(encoding="utf-8")
        hits += [f"{py.relative_to(ROOT)}: {pat}" for pat in legacy if pat in text]
    assert not hits, hits
