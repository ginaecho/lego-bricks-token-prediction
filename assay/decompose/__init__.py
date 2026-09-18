"""Encoding a plain-English request into a brick vector."""

from __future__ import annotations

from assay.decompose.keyword import KeywordEncoder, RULES_SHA256
from assay.decompose.parser import Decomposition, parse
from assay.decompose.prompt import build_prompt

__all__ = ["Decomposition", "parse", "build_prompt", "KeywordEncoder", "RULES_SHA256"]
