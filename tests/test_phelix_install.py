"""Smoke tests for the Sherlock Phelix environment installation.

These tests are intentionally light: they validate imports, editable install
paths, tool availability, and optional GPU visibility without running training.
Set PHELIX_INSTALL_TEST_STRICT=1 after installing the Sherlock environment to
turn missing optional dependencies into failures instead of skips.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


STRICT = os.environ.get("PHELIX_INSTALL_TEST_STRICT") == "1"
REQUIRE_GPU = os.environ.get("PHELIX_REQUIRE_GPU") == "1"
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()


def _skip_or_fail(message: str) -> None:
    if STRICT:
        pytest.fail(message)
    pytest.skip(message)


def _import_or_skip(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        _skip_or_fail(f"{module_name} import failed: {exc}")


def _require_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        _skip_or_fail(f"required command not found on PATH: {name}")
    return path


def test_python_runtime_is_312_or_newer():
    if sys.version_info < (3, 12):
        _skip_or_fail(f"Python >=3.12 required, got {sys.version.split()[0]}")


@pytest.mark.parametrize("command", ["uv", "gcc", "g++", "jackhmmer"])
def test_required_commands_are_available(command: str):
    _require_command(command)


def test_jackhmmer_seq_limit_patch_is_active():
    jackhmmer = _require_command("jackhmmer")
    result = subprocess.run(
        [jackhmmer, "-h", "--seq_limit", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    help_result = subprocess.run(
        [jackhmmer, "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert "--seq_limit" in f"{help_result.stdout}\n{help_result.stderr}"


@pytest.mark.parametrize(
    "module_name",
    ["alphafold3", "atomworks", "allatom_design", "jax", "torch", "rdkit"],
)
def test_core_packages_import(module_name: str):
    _import_or_skip(module_name)


@pytest.mark.parametrize(
    "module_name",
    ["alphafold3", "atomworks", "allatom_design"],
)
def test_editable_packages_resolve_to_checkout(module_name: str):
    module = _import_or_skip(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        _skip_or_fail(f"{module_name} has no __file__; cannot verify editable install")

    resolved = Path(module_file).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        _skip_or_fail(
            f"{module_name} is not imported from checkout {PROJECT_ROOT}: {resolved}"
        )


def test_jax_and_torch_runtime_visibility():
    jax = _import_or_skip("jax")
    torch = _import_or_skip("torch")

    try:
        jax_devices = jax.devices()
    except Exception as exc:
        _skip_or_fail(f"jax.devices() failed: {exc}")
    assert jax_devices, "jax.devices() returned no devices"

    if REQUIRE_GPU:
        assert torch.cuda.is_available(), "torch CUDA is not available"
        assert any(device.platform == "gpu" for device in jax_devices), (
            f"JAX GPU device not visible; devices={jax_devices}"
        )
