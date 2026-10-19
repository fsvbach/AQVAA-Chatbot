"""Unit tests for role-based LLM profile resolution and factory caching.

These neither hit the network nor require a running model: client constructors
only build SDK objects (AsyncOpenAI / ollama.AsyncClient) lazily. Run with:
    pytest tests/test_llm_profiles.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Settings, LLMProvider, LLMRole, ModelProfile
from llm import factory


@pytest.fixture(autouse=True)
def _isolate_llm_env(monkeypatch):
    """Strip LLM_* vars so Settings resolves purely from kwargs. Other test
    modules (e.g. test_vector_store) pull in deps that load the real .env into
    os.environ, which would otherwise override _env_file=None."""
    for key in list(os.environ):
        if key.startswith("LLM_"):
            monkeypatch.delenv(key, raising=False)

_GEN = dict(
    llm_provider=LLMProvider.VLLM,
    llm_base_url="http://gen:8000/v1",
    llm_model="gen-model",
)
_AUX = dict(
    aux_base_url="http://aux:8001/v1",
    aux_model="aux-model",
)


def _settings(**overrides) -> Settings:
    """Build Settings isolated from the real .env so resolution is deterministic."""
    return Settings(_env_file=None, **overrides)


def test_non_aux_roles_always_use_generation():
    s = _settings(**_GEN, **_AUX)
    for role in (LLMRole.GENERATION, LLMRole.REWRITING, LLMRole.CLASSIFICATION):
        p = s.profile(role)
        assert p.model == "gen-model"
        assert p.base_url == "http://gen:8000/v1"


def test_aux_roles_use_aux_when_set():
    s = _settings(**_GEN, **_AUX)
    for role in (LLMRole.GUARDRAILS, LLMRole.QUERY_ROUTING):
        p = s.profile(role)
        assert p.model == "aux-model"
        assert p.base_url == "http://aux:8001/v1"


def test_aux_roles_fall_back_to_generation_when_unset():
    s = _settings(**_GEN)  # no aux_* set
    for role in (LLMRole.GUARDRAILS, LLMRole.QUERY_ROUTING):
        p = s.profile(role)
        assert p.model == "gen-model"
        assert p.base_url == "http://gen:8000/v1"


def test_aux_provider_inherits_generation_when_unset():
    s = _settings(**_GEN, **_AUX)  # aux_provider left unset
    assert s.profile(LLMRole.GUARDRAILS).provider == LLMProvider.VLLM


def test_aux_fields_fall_back_independently():
    # aux model set, but aux base_url unset -> base_url inherits generation.
    s = _settings(**_GEN, aux_model="aux-model")
    p = s.profile(LLMRole.GUARDRAILS)
    assert p.model == "aux-model"
    assert p.base_url == "http://gen:8000/v1"


def test_build_client_dedupes_equal_profiles():
    p1 = ModelProfile(LLMProvider.VLLM, "http://x:8000/v1", "k", "m")
    p2 = ModelProfile(LLMProvider.VLLM, "http://x:8000/v1", "k", "m")
    assert factory._build_client(p1) is factory._build_client(p2)


def test_build_client_distinct_across_profiles():
    p1 = ModelProfile(LLMProvider.VLLM, "http://x:8000/v1", "k", "m1")
    p2 = ModelProfile(LLMProvider.VLLM, "http://y:8001/v1", "k", "m2")
    c1, c2 = factory._build_client(p1), factory._build_client(p2)
    assert c1 is not c2
    assert c1.model == "m1"
    assert c2.model == "m2"
