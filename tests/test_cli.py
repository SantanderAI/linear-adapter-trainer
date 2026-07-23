# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import pytest

from linear_adapter_trainer.cli import _output, build_parser, cmd_run, cmd_train


def test_output_defaults_to_safetensors():
    assert _output({})["adapter_path"] == "artifacts/adapter.safetensors"


@pytest.mark.parametrize("command", ["generate", "train", "evaluate", "run"])
def test_parser_accepts_config_for_each_command(command):
    args = build_parser().parse_args([command, "config.toml"])
    assert args.command == command
    assert args.config == "config.toml"


@pytest.mark.parametrize("command", [cmd_train, cmd_run])
def test_training_commands_reject_legacy_output_before_work(command, monkeypatch):
    started = False

    def fail_if_started(*args, **kwargs):
        nonlocal started
        started = True
        raise AssertionError("training work started before output validation")

    monkeypatch.setattr("linear_adapter_trainer.cli.build_knowledge_base", fail_if_started)
    with pytest.raises(ValueError, match=r"Change output\.adapter_path"):
        command({"output": {"adapter_path": "artifacts/adapter.pt"}})
    assert not started
