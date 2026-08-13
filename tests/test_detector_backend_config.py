"""Detector-generated backend configs must be both valid and runnable."""

import pytest

from mcpify.backend import create_adapter
from mcpify.detect.base import ProjectInfo
from mcpify.detect.camel import CamelDetector
from mcpify.detect.openai import OpenaiDetector
from mcpify.validate import MCPConfigValidator


@pytest.mark.parametrize("detector_cls", [OpenaiDetector, CamelDetector])
def test_library_backend_config_is_valid_and_runnable(detector_cls: type) -> None:
    """A library project's generated backend must validate and build an adapter.

    Regression: the library-project default was ``{"type": "python"}``, which
    ``VALID_BACKEND_TYPES`` does not include and ``create_adapter`` cannot
    build, so ``mcpify serve`` on a detected library project raised
    ``ValueError: Unsupported backend type: python``.
    """
    info = ProjectInfo(
        name="lib",
        description="library project",
        main_files=["main.py"],
        readme_content="",
        project_type="library",
        dependencies=[],
    )
    # `_generate_backend_config_from_content` does not use `self`, so it can be
    # called unbound — no API key or network needed.
    backend = detector_cls._generate_backend_config_from_content(None, info)

    assert backend["type"] in MCPConfigValidator.VALID_BACKEND_TYPES
    create_adapter(backend)  # must not raise
