"""Sanity check that CLI imports the fewshot wiring correctly."""

import importlib


def test_app_imports_fewshot_modules():
    app = importlib.import_module("lingxi.app")
    # The top-level imports should have brought these into the module namespace
    assert hasattr(app, "parse_annotation_command")


def test_feishu_cli_imports():
    # If feishu_cli isn't importable, the entire workflow is broken
    importlib.import_module("lingxi.channels.feishu_cli")
