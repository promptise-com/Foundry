"""Tests for ``promptise.models`` — bring-your-own-model resolution — and the
``promptise models`` CLI.

No test here talks to a provider. Construction is verified with placeholder
credentials (chat-model classes validate config without a network call), and
every user-facing error is asserted to say what to do.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from promptise import Model
from promptise import models as m
from promptise.cli import app
from promptise.models import (
    PROVIDERS,
    ModelSetupError,
    check_model,
    env_template,
    find_provider,
    parse_model,
    resolve_model,
)

runner = CliRunner(env={"COLUMNS": "200"})  # Rich wraps at COLUMNS; keep phrases on one line

_AZURE = {
    "AZURE_OPENAI_ENDPOINT": "https://demo.openai.azure.com/",
    "AZURE_OPENAI_API_KEY": "k",
    "OPENAI_API_VERSION": "2024-10-21",
}
_ALL_VARS = sorted(
    {v.name for p in PROVIDERS for v in p.env}
    | {a for p in PROVIDERS for v in p.env for a in v.aliases}
    | {"OPENAI_API_KEY", "OPENAI_BASE_URL"}
)


@pytest.fixture()
def clean_env(monkeypatch):
    for name in _ALL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")  # never read the developer's own .env
    return monkeypatch


def _base(llm) -> str | None:
    return (
        getattr(llm, "openai_api_base", None)
        or getattr(llm, "azure_endpoint", None)
        or getattr(llm, "anthropic_api_url", None)
    )


# ---------------------------------------------------------------------------
# Registry invariants — the docs and the CLI are generated from it
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_prefixes_are_unique_and_lowercase(self):
        seen: dict[str, str] = {}
        for p in PROVIDERS:
            for prefix in p.prefixes:
                assert prefix == prefix.lower(), prefix
                assert prefix not in seen, f"{prefix} claimed by {seen[prefix]} and {p.key}"
                seen[prefix] = p.key

    def test_nothing_needs_an_extra_by_default(self):
        """The whole point: pip install promptise reaches every provider."""
        tomllib = pytest.importorskip("tomllib")  # 3.11+; the rest of the module runs on 3.10
        for p in PROVIDERS:
            if p.base_url is None:  # native route must be core
                assert p.native_package is None, p.key
                assert p.native_installed, p.key
        extras = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]
        assert not any(
            k in extras for k in ("google", "bedrock", "ollama", "models-all", "anthropic")
        )
        core = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"][
            "dependencies"
        ]
        assert any(d.startswith("langchain-openai") for d in core)
        assert any(d.startswith("langchain-anthropic") for d in core)

    def test_native_keys_are_langchain_providers(self):
        from langchain.chat_models.base import _BUILTIN_PROVIDERS

        for p in PROVIDERS:
            assert p.native in _BUILTIN_PROVIDERS, p.key
            if p.native_package:
                assert _BUILTIN_PROVIDERS[p.native][0].split(".")[0] == p.native_package, p.key

    def test_every_env_var_feeds_a_word(self):
        for p in PROVIDERS:
            for v in p.env:
                assert v.word in m.WORDS, (p.key, v.name, v.word)
                assert v.word in p.words(), (p.key, v.word)

    def test_examples_resolve_to_their_own_provider(self):
        for p in PROVIDERS:
            provider, canonical, _ = parse_model(p.example)
            assert provider is p, p.example
            assert canonical.startswith(p.key + ":")

    def test_compat_templates_only_use_known_words(self):
        import re

        for p in PROVIDERS:
            if p.base_url:
                for placeholder in re.findall(r"{(\w+)}", p.base_url):
                    assert placeholder in m.WORDS, (p.key, placeholder)


# ---------------------------------------------------------------------------
# Parsing and aliases
# ---------------------------------------------------------------------------


class TestParse:
    @pytest.mark.parametrize(
        ("spec", "key", "model"),
        [
            ("azure:chat-prod", "azure_openai", "chat-prod"),
            ("AZURE:chat-prod", "azure_openai", "chat-prod"),
            ("azure_openai:chat-prod", "azure_openai", "chat-prod"),
            ("aoai:chat-prod", "azure_openai", "chat-prod"),
            ("foundry:Llama-3.3-70B-Instruct", "azure_ai", "Llama-3.3-70B-Instruct"),
            ("azure-ai:phi-4", "azure_ai", "phi-4"),
            ("gemini:gemini-2.5-pro", "google_genai", "gemini-2.5-pro"),
            ("google:gemini-2.5-pro", "google_genai", "gemini-2.5-pro"),
            ("vertex:gemini-2.5-pro", "google_vertexai", "gemini-2.5-pro"),
            (
                "bedrock:anthropic.claude-sonnet-4-20250514-v1:0",
                "bedrock",
                "anthropic.claude-sonnet-4-20250514-v1:0",
            ),
            ("aws:x", "bedrock", "x"),
            ("mistral:mistral-large-latest", "mistralai", "mistral-large-latest"),
            ("grok:grok-4", "xai", "grok-4"),
            (
                "hf:meta-llama/Llama-3.3-70B-Instruct",
                "huggingface",
                "meta-llama/Llama-3.3-70B-Instruct",
            ),
            ("claude:claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"),
            ("openai:gpt-5-mini", "openai", "gpt-5-mini"),
            ("local:llama3.1", "ollama", "llama3.1"),
        ],
    )
    def test_aliases(self, spec, key, model):
        provider, canonical, parsed = parse_model(spec)
        assert provider is not None and provider.key == key
        assert canonical == f"{key}:{model}" and parsed == model

    def test_unknown_prefix_is_left_to_langchain(self):
        provider, canonical, model = parse_model("gpt-5-mini")
        assert provider is None and canonical == "gpt-5-mini" == model
        assert parse_model("nope:thing")[0] is None

    def test_find_provider(self):
        assert find_provider("azure").key == "azure_openai"
        assert find_provider("Azure-OpenAI").key == "azure_openai"
        assert find_provider("azure_ai").key == "azure_ai"
        assert find_provider("nope") is None


# ---------------------------------------------------------------------------
# Every provider through the core install
# ---------------------------------------------------------------------------


class TestEveryProviderWithCoreOnly:
    @pytest.mark.parametrize(
        ("model", "cls", "base"),
        [
            (
                Model("llama-3.3-70b-versatile", provider="groq", api_key="k"),
                "ChatOpenAI",
                "https://api.groq.com/openai/v1",
            ),
            (
                Model("gemini-2.5-pro", provider="gemini", api_key="k"),
                "ChatOpenAI",
                "https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            (
                Model("mistral-large-latest", provider="mistral", api_key="k"),
                "ChatOpenAI",
                "https://api.mistral.ai/v1",
            ),
            (
                Model("deepseek-chat", provider="deepseek", api_key="k"),
                "ChatOpenAI",
                "https://api.deepseek.com/v1",
            ),
            (Model("grok-4", provider="grok", api_key="k"), "ChatOpenAI", "https://api.x.ai/v1"),
            (
                Model("m", provider="together", api_key="k"),
                "ChatOpenAI",
                "https://api.together.xyz/v1",
            ),
            (
                Model("m", provider="fireworks", api_key="k"),
                "ChatOpenAI",
                "https://api.fireworks.ai/inference/v1",
            ),
            (
                Model("command-a-03-2025", provider="cohere", api_key="k"),
                "ChatOpenAI",
                "https://api.cohere.ai/compatibility/v1",
            ),
            (
                Model("m", provider="openrouter", api_key="k"),
                "ChatOpenAI",
                "https://openrouter.ai/api/v1",
            ),
            (
                Model("sonar-pro", provider="perplexity", api_key="k"),
                "ChatOpenAI",
                "https://api.perplexity.ai",
            ),
            (
                Model("meta/llama-3.3-70b-instruct", provider="nvidia", api_key="k"),
                "ChatOpenAI",
                "https://integrate.api.nvidia.com/v1",
            ),
            (
                Model("org/model", provider="hf", api_key="k"),
                "ChatOpenAI",
                "https://router.huggingface.co/v1",
            ),
            (Model("llama3.1", provider="ollama"), "ChatOpenAI", "http://localhost:11434/v1"),
            (
                Model("claude-sonnet-4-5", provider="anthropic", api_key="k"),
                "ChatAnthropic",
                "https://api.anthropic.com",
            ),
            (Model("gpt-5-mini", provider="openai", api_key="k"), "ChatOpenAI", None),
        ],
    )
    def test_public_api_providers(self, clean_env, model, cls, base):
        llm = model.resolve()
        assert type(llm).__name__ == cls
        assert _base(llm) == base

    def test_bedrock_with_an_api_key_and_region(self, clean_env):
        llm = Model(
            "anthropic.claude-sonnet-4-20250514-v1:0",
            provider="bedrock",
            region="eu-central-1",
            api_key="ABSK",
        ).resolve()
        assert _base(llm) == "https://bedrock-runtime.eu-central-1.amazonaws.com/openai/v1"

    def test_vertex_with_a_token(self, clean_env):
        llm = Model(
            "gemini-2.5-pro",
            provider="vertex",
            project="my-proj",
            region="europe-west1",
            api_key="ya29",
        ).resolve()
        assert _base(llm) == (
            "https://europe-west1-aiplatform.googleapis.com/v1/projects/my-proj/locations/europe-west1/endpoints/openapi"
        )
        assert llm.model_name == "google/gemini-2.5-pro"  # publisher prefix added

    def test_vertex_without_a_token_explains(self, clean_env, monkeypatch):
        monkeypatch.setattr(m, "_vertex_token", lambda: None)
        with pytest.raises(ModelSetupError, match="gcloud auth print-access-token"):
            Model("gemini-2.5-pro", provider="vertex", project="p").resolve()

    def test_vertex_google_auth_failure_names_the_reason(self, clean_env, monkeypatch):
        """google-auth is installed but cannot mint a token: the error says why
        (chained), instead of advising to install a package that is present."""
        import sys
        import types

        class NoADC(Exception):
            pass

        fake_auth = types.ModuleType("google.auth")

        def default(scopes):
            raise NoADC("Your default credentials were not found")

        fake_auth.default = default  # type: ignore[attr-defined]
        fake_transport = types.ModuleType("google.auth.transport")
        fake_requests = types.ModuleType("google.auth.transport.requests")
        fake_requests.Request = object  # type: ignore[attr-defined]
        fake_transport.requests = fake_requests  # type: ignore[attr-defined]
        fake_auth.transport = fake_transport  # type: ignore[attr-defined]
        fake_google = types.ModuleType("google")
        fake_google.auth = fake_auth  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "google", fake_google)
        monkeypatch.setitem(sys.modules, "google.auth", fake_auth)
        monkeypatch.setitem(sys.modules, "google.auth.transport", fake_transport)
        monkeypatch.setitem(sys.modules, "google.auth.transport.requests", fake_requests)
        monkeypatch.setattr(m.importlib.util, "find_spec", lambda name: object())

        with pytest.raises(ModelSetupError) as info:
            Model("gemini-2.5-pro", provider="vertex", project="p").resolve()
        message = str(info.value)
        assert "NoADC: Your default credentials were not found" in message
        assert "gcloud auth application-default login" in message
        assert isinstance(info.value.__cause__, NoADC)

    def test_vertex_token_none_when_google_auth_missing(self, clean_env, monkeypatch):
        monkeypatch.setattr(m.importlib.util, "find_spec", lambda name: None)
        assert m._vertex_token() is None

    @staticmethod
    def _install_fake_google_auth(monkeypatch, credentials):
        """A ``google.auth`` whose ``default()`` hands out *credentials*."""
        import sys
        import types

        fake_auth = types.ModuleType("google.auth")
        fake_auth.default = lambda scopes: (credentials, "proj")  # type: ignore[attr-defined]
        fake_transport = types.ModuleType("google.auth.transport")
        fake_requests = types.ModuleType("google.auth.transport.requests")
        fake_requests.Request = lambda: "request"  # type: ignore[attr-defined]
        fake_transport.requests = fake_requests  # type: ignore[attr-defined]
        fake_auth.transport = fake_transport  # type: ignore[attr-defined]
        fake_google = types.ModuleType("google")
        fake_google.auth = fake_auth  # type: ignore[attr-defined]
        for name, module in (
            ("google", fake_google),
            ("google.auth", fake_auth),
            ("google.auth.transport", fake_transport),
            ("google.auth.transport.requests", fake_requests),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(m.importlib.util, "find_spec", lambda name: object())

    class _FakeCredentials:
        """google-auth credentials: ``valid`` flips, ``refresh`` mints the next token."""

        def __init__(self):
            self.valid = False
            self.token = None
            self.refreshes = 0
            self.fail_next_refresh = False

        def refresh(self, request):
            assert request == "request"
            if self.fail_next_refresh:
                raise RuntimeError("token endpoint unreachable")
            self.refreshes += 1
            self.token = f"ya29.MINTED-{self.refreshes}"
            self.valid = True

    def test_vertex_token_provider_refreshes_an_expired_token(self, clean_env, monkeypatch):
        """The auto-minted token is a provider, not a pinned string: it is
        refreshed once eagerly (fail fast), served while valid, and minted
        again when google-auth reports it expired (~1 h later)."""
        creds = self._FakeCredentials()
        self._install_fake_google_auth(monkeypatch, creds)
        token = m._vertex_token()
        assert callable(token) and creds.refreshes == 1
        assert token() == "ya29.MINTED-1" and creds.refreshes == 1  # still valid: no refresh
        creds.valid = False  # expired
        assert token() == "ya29.MINTED-2" and creds.refreshes == 2
        creds.valid = False
        creds.fail_next_refresh = True
        with pytest.raises(ModelSetupError, match="could not refresh.*token endpoint unreachable"):
            token()

    def test_vertex_auto_minted_token_reaches_the_client_as_a_provider(
        self, clean_env, monkeypatch
    ):
        creds = self._FakeCredentials()
        self._install_fake_google_auth(monkeypatch, creds)
        captured = {}
        monkeypatch.setattr(
            "langchain.chat_models.init_chat_model",
            lambda model, **kw: captured.update(model=model, **kw),
        )
        Model("gemini-2.5-pro", provider="vertex", project="p").resolve()
        assert captured["model"] == "openai:google/gemini-2.5-pro"
        assert callable(captured["api_key"]) and captured["api_key"]() == "ya29.MINTED-1"
        creds.valid = False
        assert captured["api_key"]() == "ya29.MINTED-2"  # what the next request sends

    def test_vertex_token_is_pinned_only_for_a_string_only_langchain_openai(
        self, clean_env, monkeypatch
    ):
        """Compatibility branch, said out loud: a langchain-openai that takes
        only a string api_key gets the minted token pinned, with a warning
        naming the 1 h lifetime and the upgrade that fixes it."""
        creds = self._FakeCredentials()
        self._install_fake_google_auth(monkeypatch, creds)
        monkeypatch.setattr(m, "_openai_api_key_accepts_callable", lambda: False)
        captured = {}
        monkeypatch.setattr(
            "langchain.chat_models.init_chat_model", lambda model, **kw: captured.update(kw)
        )
        with pytest.warns(UserWarning, match="pinned.*1 h.*1.0.1"):
            Model("gemini-2.5-pro", provider="vertex", project="p").resolve()
        assert captured["api_key"] == "ya29.MINTED-1"

    def test_installed_langchain_openai_accepts_a_callable_api_key(self):
        """The floor in pyproject (0.3.30) takes a string only; 1.0.1+ takes a
        callable. Whatever is installed, the detection must agree with it."""
        import re
        from importlib.metadata import version

        from langchain_openai import ChatOpenAI

        annotation = str(ChatOpenAI.model_fields["openai_api_key"].annotation)
        assert m._openai_api_key_accepts_callable() == ("Callable" in annotation)
        installed = tuple(int(p) for p in re.findall(r"\d+", version("langchain-openai"))[:3])
        assert m._openai_api_key_accepts_callable() == (installed >= (1, 0, 1))

    def test_vertex_token_given_by_hand_is_used_as_is(self, clean_env, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            "langchain.chat_models.init_chat_model", lambda model, **kw: captured.update(kw)
        )
        monkeypatch.setattr(m, "_vertex_token", lambda: pytest.fail("must not mint"))
        Model("gemini-2.5-pro", provider="vertex", project="p", api_key="ya29.by-hand").resolve()
        assert captured["api_key"] == "ya29.by-hand"

    def test_foundry_catalog(self, clean_env):
        llm = Model(
            "Llama-3.3-70B-Instruct",
            provider="foundry",
            endpoint="https://r.services.ai.azure.com/models",
            api_key="k",
        ).resolve()
        assert _base(llm) == "https://r.services.ai.azure.com/models"
        assert llm.default_query == {"api-version": "2024-05-01-preview"}
        custom = Model(
            "m",
            provider="foundry",
            endpoint="https://r/models",
            api_key="k",
            api_version="2025-01-01",
        ).resolve()
        assert custom.default_query == {"api-version": "2025-01-01"}

    def test_azure_openai_native(self, clean_env):
        llm = Model(
            "gpt-4o",
            provider="azure",
            deployment="chat-prod",
            endpoint="https://r.openai.azure.com/",
            api_key="k",
            api_version="2024-10-21",
            temperature=0,
        ).resolve()
        assert type(llm).__name__ == "AzureChatOpenAI"
        assert llm.model_name == "gpt-4o" and llm.deployment_name == "chat-prod"
        assert llm.azure_endpoint == "https://r.openai.azure.com/" and llm.temperature == 0
        # deployment defaults to the model name
        assert (
            Model(
                "chat-prod",
                provider="azure",
                endpoint="https://r.openai.azure.com/",
                api_key="k",
                api_version="2024-10-21",
            )
            .resolve()
            .deployment_name
            == "chat-prod"
        )

    def test_self_hosted_endpoints(self, clean_env):
        vllm = Model(
            "qwen2.5", provider="openai", endpoint="http://gpu:8000/v1", api_key="none"
        ).resolve()
        assert str(vllm.client._client.base_url) == "http://gpu:8000/v1/"
        remote = Model("llama3.1", provider="ollama", endpoint="gpu-box:11434").resolve()
        assert _base(remote) == "http://gpu-box:11434/v1"
        nim = Model(
            "meta/llama-3.3-70b-instruct",
            provider="nvidia",
            endpoint="http://nim:8000/v1",
            api_key="k",
        ).resolve()
        assert _base(nim) == "http://nim:8000/v1"
        hf = Model(
            "org/model",
            provider="hf",
            endpoint="https://xyz.endpoints.huggingface.cloud/v1",
            api_key="k",
        ).resolve()
        assert _base(hf) == "https://xyz.endpoints.huggingface.cloud/v1"

    def test_env_vars_feed_the_words(self, clean_env, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk")
        assert type(resolve_model("groq:llama-3.3-70b-versatile")).__name__ == "ChatOpenAI"
        monkeypatch.setenv("GEMINI_API_KEY", "AIza")  # alias of GOOGLE_API_KEY
        assert check_model("gemini:gemini-2.5-pro").ok
        monkeypatch.setenv("AWS_REGION", "us-west-2")  # alias of AWS_DEFAULT_REGION
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "ABSK")
        assert (
            _base(resolve_model("bedrock:x"))
            == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"
        )
        for k, v in _AZURE.items():
            monkeypatch.setenv(k, v)
        llm = resolve_model("azure:chat-prod")
        assert llm.deployment_name == "chat-prod" and llm.openai_api_version == "2024-10-21"

    def test_native_opt_in_needs_the_package_and_says_so(self, clean_env):
        with pytest.raises(ModelSetupError, match=r"pip install langchain-groq") as exc:
            Model("m", provider="groq", api_key="k", native=True).resolve()
        assert "drop native=True" in str(exc.value)

    def test_bare_name_is_inferred_by_langchain(self, clean_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        assert type(resolve_model("gpt-5-mini")).__name__ == "ChatOpenAI"
        assert type(Model("gpt-5-mini", api_key="k").resolve()).__name__ == "ChatOpenAI"

    def test_settings_and_extra_pass_through(self, clean_env):
        llm = Model(
            "m",
            provider="groq",
            api_key="k",
            temperature=0.2,
            max_tokens=64,
            timeout=9,
            extra={"seed": 7},
        ).resolve()
        assert llm.temperature == 0.2 and llm.max_tokens == 64 and llm.request_timeout == 9
        assert llm.seed == 7


# ---------------------------------------------------------------------------
# Diagnosis and errors
# ---------------------------------------------------------------------------


class TestHttpClientsArePerModel:
    """langchain-openai caches one httpx client per process; the second ``asyncio.run()``
    in a process then fails with "Event loop is closed". Every resolved model gets its own
    pair instead (the CLI runs curation and evaluation in separate loops)."""

    def test_two_resolutions_never_share_a_client(self, clean_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        a = Model("gpt-5-mini", provider="openai").resolve()
        b = Model("gpt-5-mini", provider="openai").resolve()
        assert a.root_async_client._client is not b.root_async_client._client
        assert a.root_client._client is not b.root_client._client
        compat = Model("llama-3.3-70b-versatile", provider="groq", api_key="k").resolve()
        again = Model("llama-3.3-70b-versatile", provider="groq", api_key="k").resolve()
        assert compat.root_async_client._client is not again.root_async_client._client

    def test_a_caller_supplied_client_is_kept(self, clean_env):
        import httpx

        mine = httpx.AsyncClient()
        llm = Model(
            "gpt-5-mini", provider="openai", api_key="k", extra={"http_async_client": mine}
        ).resolve()
        assert llm.root_async_client._client is mine

    def test_fresh_clients_mirror_the_sdk_defaults(self):
        from promptise.models import _fresh_http_clients

        pair = _fresh_http_clients({})
        assert set(pair) == {"http_client", "http_async_client"}
        assert pair["http_async_client"].timeout.read == 600.0
        assert pair["http_async_client"].timeout.connect == 5.0
        assert _fresh_http_clients({"http_client": object()}) == {}


class TestErrors:
    def test_azure_names_every_missing_variable_and_where_to_find_it(self, clean_env):
        with pytest.raises(ModelSetupError) as exc:
            resolve_model("azure:chat-prod")
        text = str(exc.value)
        for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "OPENAI_API_VERSION"):
            assert var in text
        assert "Foundry" in text and "DEPLOYMENT name" in text
        assert ".env file next to your script" in text
        assert "Model(..., endpoint=, api_key=, api_version=)" in text
        assert "promptise models env azure" in text

    def test_check_model_reports_without_raising(self, clean_env):
        result = check_model("azure:chat-prod")
        assert result.provider.key == "azure_openai" and not result.ok
        assert [v.name for v in result.missing_env] == [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "OPENAI_API_VERSION",
        ]
        assert check_model("gpt-5-mini").provider is None and check_model("gpt-5-mini").ok

    def test_words_in_code_satisfy_env(self, clean_env):
        assert check_model("openai:gpt-5-mini", provided={"api_key"}).ok
        assert check_model("azure:chat-prod", provided={"api_key", "endpoint", "api_version"}).ok
        assert not check_model("azure:chat-prod", provided={"api_key"}).ok

    @pytest.mark.parametrize(
        ("model", "needle"),
        [
            (
                {"model": "m", "provider": "groq", "deployment": "d"},
                "only Azure OpenAI addresses models by deployment",
            ),
            ({"model": "m", "provider": "ollama", "api_key": "k"}, "needs no key"),
            (
                {"model": "m", "provider": "openai", "api_version": "x"},
                "only Azure uses an API version",
            ),
            (
                {"model": "m", "provider": "groq", "region": "x"},
                "only Bedrock and Vertex AI are regional",
            ),
            (
                {"model": "m", "provider": "gemini", "project": "x"},
                "only Vertex AI needs a Google Cloud project",
            ),
        ],
    )
    def test_inapplicable_word_explains_what_to_do(self, clean_env, model, needle):
        with pytest.raises(ModelSetupError, match=needle):
            Model(**model).resolve()

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_api_key_does_not_count_as_given(self, clean_env, blank):
        """``Model(api_key="")`` — what ``${OPENAI_API_KEY}`` yields for a variable
        exported but empty — must not shadow the environment check."""
        with pytest.raises(ModelSetupError, match="OPENAI_API_KEY"):
            Model("gpt-5", provider="openai", api_key=blank).resolve()
        with pytest.raises(ModelSetupError, match="GROQ_API_KEY") as info:
            Model("llama-3.3-70b-versatile", provider="groq", api_key=blank).resolve()
        assert "OPENAI_API_KEY" not in str(info.value)
        assert "api_key" not in Model("gpt-5", provider="openai", api_key=blank).kwargs()

    def test_blank_api_key_falls_back_to_the_environment(self, clean_env, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-from-env")
        captured = {}
        monkeypatch.setattr(
            "langchain.chat_models.init_chat_model", lambda model, **kw: captured.update(kw)
        )
        Model("llama-3.3-70b-versatile", provider="groq", api_key="").resolve()
        assert captured["api_key"] == "gsk-from-env"
        captured.clear()
        resolve_model("groq:llama-3.3-70b-versatile", api_key="   ", endpoint="")
        assert captured["api_key"] == "gsk-from-env"
        assert captured["base_url"] == "https://api.groq.com/openai/v1"  # blank endpoint ignored

    def test_unknown_provider(self, clean_env):
        with pytest.raises(ModelSetupError, match="unknown provider 'nope'"):
            Model("x", provider="nope")
        with pytest.raises(ModelSetupError, match="promptise models list"):
            resolve_model("nope:thing")

    def test_env_template(self):
        text = env_template("azure")
        assert "export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/" in text
        assert "export OPENAI_API_VERSION=2024-10-21" in text
        assert "pip install" not in text
        assert "# optional" in env_template("vertex")
        with pytest.raises(ModelSetupError, match="unknown provider"):
            env_template("nope")


# ---------------------------------------------------------------------------
# Model — the string form and the YAML form agree with the keyword form
# ---------------------------------------------------------------------------


class TestModelForms:
    def test_string_form_is_a_shortcut(self):
        short = Model("azure:chat-prod")
        assert short.provider == "azure" and short.model == "chat-prod"
        assert short.spec == "azure_openai:chat-prod" == Model("chat-prod", provider="azure").spec
        assert Model("gpt-5-mini").provider is None and Model("gpt-5-mini").spec == "gpt-5-mini"

    @pytest.mark.asyncio
    async def test_build_agent_accepts_string_and_model(self, clean_env):
        from promptise import build_agent

        cfg = Model(
            "gpt-4o",
            provider="azure",
            deployment="chat-prod",
            endpoint="https://r.openai.azure.com/",
            api_key="k",
            api_version="2024-10-21",
        )
        assert await build_agent(servers={}, model=cfg) is not None
        with pytest.raises(ModelSetupError, match="AZURE_OPENAI_ENDPOINT"):
            await build_agent(servers={}, model="azure:chat-prod")

    def test_yaml_form_agrees_with_code(self, clean_env, tmp_path):
        from promptise.superagent import load_superagent_file
        from promptise.superagent_schema import DetailedModelConfig

        cfg = DetailedModelConfig(
            provider="azure", model="gpt-4o", deployment="chat-prod"
        )  # `model:` synonym
        assert cfg.name == "gpt-4o" and cfg.deployment == "chat-prod"
        path = tmp_path / "a.superagent"
        path.write_text(
            "agent:\n"
            "  model:\n"
            "    provider: azure\n"
            "    model: gpt-4o\n"
            "    deployment: chat-prod\n"
            "    endpoint: https://r.openai.azure.com/\n"
            "    api_key: k\n"
            "    api_version: '2024-10-21'\n"
            "    temperature: 0.1\n"
            "servers:\n"
            "  tools:\n"
            "    type: http\n"
            "    url: http://127.0.0.1:9/mcp\n",
            encoding="utf-8",
        )
        loader, _ = load_superagent_file(path)
        assert loader.to_model_string() == "azure:gpt-4o"
        expected = Model(
            "gpt-4o",
            provider="azure",
            deployment="chat-prod",
            endpoint="https://r.openai.azure.com/",
            api_key="k",
            api_version="2024-10-21",
            temperature=0.1,
        ).kwargs()
        assert loader.to_model_kwargs() == expected
        llm = resolve_model(loader.to_model_string(), **loader.to_model_kwargs())
        assert llm.deployment_name == "chat-prod" and llm.model_name == "gpt-4o"

    def test_repr_and_str_never_show_credentials(self):
        cfg = Model(
            "gpt-5",
            provider="openai",
            api_key="sk-SECRET-abc123",
            extra={"default_headers": {"Authorization": "Bearer hdr-SECRET"}},
        )
        for rendered in (repr(cfg), str(cfg), f"{cfg}", f"{cfg!r}"):
            assert "SECRET" not in rendered
            assert "api_key" not in rendered
            assert "default_headers" not in rendered
        # the identity is still there, and the values still reach the provider
        assert "gpt-5" in repr(cfg) and "openai" in repr(cfg)
        assert cfg.api_key == "sk-SECRET-abc123"
        assert cfg.kwargs()["api_key"] == "sk-SECRET-abc123"
        assert cfg.kwargs()["default_headers"]["Authorization"] == "Bearer hdr-SECRET"

    def test_yaml_base_url_is_an_alias_of_endpoint(self):
        from promptise.superagent_schema import DetailedModelConfig

        cfg = DetailedModelConfig(
            provider="openai", name="qwen", base_url="http://h/v1", api_key="x"
        )
        assert cfg.endpoint is None and cfg.base_url == "http://h/v1"


# ---------------------------------------------------------------------------
# .env — where secrets live
# ---------------------------------------------------------------------------


class TestDotenv:
    @pytest.fixture()
    def project(self, tmp_path, monkeypatch):
        for name in _ALL_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv("PROMPTISE_NO_DOTENV", raising=False)
        monkeypatch.setattr(m, "_dotenv_loaded", None)
        monkeypatch.setattr(m, "_dotenv_filled", {})
        monkeypatch.setattr(m, "_dotenv_skipped", set())
        (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")
        sub = tmp_path / "app" / "src"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)  # a script run from a subdirectory still finds it
        return tmp_path

    def test_dotenv_in_a_parent_directory_is_loaded(self, project):
        assert check_model("openai:gpt-5-mini").ok
        assert type(resolve_model("openai:gpt-5-mini")).__name__ == "ChatOpenAI"
        assert m.load_dotenv_if_present() == str(project / ".env")

    def test_origin_of_a_filled_variable_is_recorded(self, project, monkeypatch):
        assert m.dotenv_origin("OPENAI_API_KEY") is None  # nothing loaded yet
        assert m.load_dotenv_if_present() == str(project / ".env")
        assert m.dotenv_origin("OPENAI_API_KEY") == str(project / ".env")
        assert m.dotenv_origin("PATH") is None  # from the environment, not the file
        monkeypatch.setenv("OPENAI_API_KEY", "sk-changed-later")
        assert m.dotenv_origin("OPENAI_API_KEY") is None  # no longer the file's value
        assert os.environ["OPENAI_API_KEY"] == "sk-changed-later"

    def test_empty_value_in_the_file_sets_nothing(self, project):
        (project / ".env").write_text("OPENAI_API_KEY=\nGROQ_API_KEY=gsk-x\n", encoding="utf-8")
        m.load_dotenv_if_present()
        assert "OPENAI_API_KEY" not in os.environ and os.environ["GROQ_API_KEY"] == "gsk-x"
        assert m.dotenv_origin("OPENAI_API_KEY") is None

    def test_search_stops_at_the_project_root(self, project):
        """A .env above the directory holding pyproject.toml (or .git) belongs
        to somebody else — /tmp/.env on a shared host, another checkout."""
        (project / "app" / "pyproject.toml").write_text(
            "[project]\nname = 'app'\n", encoding="utf-8"
        )
        assert m.load_dotenv_if_present() is None
        assert not check_model("openai:gpt-5-mini").ok
        (project / "app" / "pyproject.toml").unlink()
        (project / "app" / ".git").write_text(
            "gitdir: /elsewhere\n", encoding="utf-8"
        )  # a worktree marker
        assert m.load_dotenv_if_present() is None

    def test_dotenv_in_the_project_root_itself_is_loaded(self, project):
        (project / "app" / "pyproject.toml").write_text(
            "[project]\nname = 'app'\n", encoding="utf-8"
        )
        (project / "app" / ".env").write_text("OPENAI_API_KEY=sk-from-root\n", encoding="utf-8")
        assert m.load_dotenv_if_present() == str(project / "app" / ".env")
        assert check_model("openai:gpt-5-mini").ok

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
    def test_world_writable_dotenv_is_skipped_with_a_warning(self, project):
        (project / ".env").chmod(0o666)
        with pytest.warns(UserWarning, match=r"ignoring .*\.env: world-writable .*chmod o-w"):
            assert m.load_dotenv_if_present() is None
        assert "OPENAI_API_KEY" not in os.environ
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # warned once per process, not once per lookup
            assert m.load_dotenv_if_present() is None
        (project / ".env").chmod(0o644)
        assert m.load_dotenv_if_present() == str(project / ".env")

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
    def test_world_writable_dotenv_is_skipped_but_a_safe_one_above_it_still_loads(self, project):
        (project / "app" / ".env").write_text("OPENAI_API_KEY=sk-from-app\n", encoding="utf-8")
        (project / "app" / ".env").chmod(0o666)
        with pytest.warns(UserWarning, match="world-writable"):
            assert m.load_dotenv_if_present() == str(project / ".env")
        assert m.dotenv_origin("OPENAI_API_KEY") == str(project / ".env")

    @pytest.mark.skipif(os.name != "posix", reason="POSIX ownership")
    def test_dotenv_owned_by_another_user_is_skipped(self, project, monkeypatch):
        real_uid = os.geteuid()
        monkeypatch.setattr(os, "geteuid", lambda: real_uid + 1)  # we are "somebody else"
        with pytest.warns(UserWarning, match=rf"owned by uid {real_uid}, not by you"):
            assert m.load_dotenv_if_present() is None

    def test_something_that_is_not_a_regular_file_is_skipped(self, project):
        (project / ".env").unlink()
        (project / ".env").mkdir()
        with pytest.warns(UserWarning, match="not a regular file"):
            assert m.load_dotenv_if_present() is None

    @pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="root reads anything")
    def test_unreadable_dotenv_is_a_model_setup_error(self, project):
        (project / ".env").chmod(0o000)
        try:
            with pytest.raises(ModelSetupError) as info:
                m.load_dotenv_if_present()
            message = str(info.value)
            assert f"cannot read {project / '.env'}" in message
            assert "PROMPTISE_NO_DOTENV=1" in message
            assert isinstance(info.value.__cause__, PermissionError)
            # Every entry point sees the same error, never a raw PermissionError —
            # a key given in code does not skip the file that is in the way.
            with pytest.raises(ModelSetupError, match="cannot read"):
                resolve_model("openai:gpt-5-mini")
            with pytest.raises(ModelSetupError, match="cannot read"):
                Model("gpt-5-mini", provider="openai", api_key="sk-in-code").resolve()
        finally:
            (project / ".env").chmod(0o644)

    def test_non_utf8_dotenv_is_a_model_setup_error(self, project):
        (project / ".env").write_bytes(b"# caf\xe9\nOPENAI_API_KEY=sk-latin1\n")
        with pytest.raises(ModelSetupError) as info:
            check_model("openai:gpt-5-mini")
        assert f"cannot read {project / '.env'}" in str(info.value)
        assert "encoding" in str(info.value) and "PROMPTISE_NO_DOTENV=1" in str(info.value)
        assert isinstance(info.value.__cause__, UnicodeDecodeError)

    def test_cli_version_and_help_work_next_to_an_unreadable_dotenv(self, project):
        """--version/--help never touch .env; every other command reports the
        file as one clean error (exit 2) instead of a traceback."""
        if os.name != "posix" or os.geteuid() == 0:
            pytest.skip("chmod 000 needs POSIX and a non-root user")
        (project / ".env").chmod(0o000)
        try:
            result = runner.invoke(app, ["--version"])
            assert result.exit_code == 0 and "Traceback" not in _out(result)
            result = runner.invoke(app, ["--help"])
            assert result.exit_code == 0 and "Usage" in _out(result)
            result = runner.invoke(app, ["models", "list"])
            assert result.exit_code == 2, _out(result)
            assert "cannot read" in _out(result) and "PROMPTISE_NO_DOTENV=1" in _out(result)
            assert "Traceback" not in _out(result)
        finally:
            (project / ".env").chmod(0o644)

    def test_superagent_reference_to_an_exported_empty_variable_uses_the_file(
        self, project, monkeypatch
    ):
        """The CLI's own template says ``api_key: ${OPENAI_API_KEY}``; with the
        variable exported but empty that reference yields "", which must not
        shadow the value the .env file holds."""
        from promptise.superagent import load_superagent_file

        monkeypatch.setenv("OPENAI_API_KEY", "")
        path = project / "app" / "a.superagent"
        path.write_text(
            "agent:\n"
            "  model:\n"
            "    provider: openai\n"
            "    model: gpt-5-mini\n"
            "    api_key: ${OPENAI_API_KEY}\n"
            "    temperature: 0\n"
            "servers:\n"
            "  tools:\n"
            "    type: http\n"
            "    url: http://127.0.0.1:9/mcp\n",
            encoding="utf-8",
        )
        loader, _ = load_superagent_file(path)
        kwargs = loader.to_model_kwargs()
        assert "api_key" not in kwargs and kwargs["temperature"] == 0
        captured = {}
        monkeypatch.setattr(
            "langchain.chat_models.init_chat_model", lambda model, **kw: captured.update(kw)
        )
        resolve_model(loader.to_model_string(), **kwargs)
        assert captured["api_key"] == "sk-from-dotenv"

    def test_cli_names_the_file_it_loaded(self, project):
        result = runner.invoke(app, ["models", "check", "openai:gpt-5-mini"])
        assert result.exit_code == 0, _out(result)
        assert f".env loaded from {project / '.env'}" in _out(result).replace("\n", "")
        assert f"set (from {project / '.env'})" in _out(result).replace("\n", "")

    def test_environment_wins_over_the_file(self, project, monkeypatch):
        import os

        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
        resolve_model("openai:gpt-5-mini")
        assert os.environ["OPENAI_API_KEY"] == "sk-from-shell"

    def test_opt_out(self, project, monkeypatch):
        monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
        assert m.load_dotenv_if_present() is None
        assert not check_model("openai:gpt-5-mini").ok

    def test_exported_empty_variable_is_filled_from_the_file(self, project, monkeypatch):
        """``export OPENAI_API_KEY=`` in a shell profile must not hide the
        value in .env — python-dotenv alone skips any key present in os.environ."""
        import os

        monkeypatch.setenv("OPENAI_API_KEY", "")
        assert m.load_dotenv_if_present() == str(project / ".env")
        assert os.environ["OPENAI_API_KEY"] == "sk-from-dotenv"
        assert check_model("openai:gpt-5-mini").ok

    def test_exported_empty_variable_is_explained_when_the_file_has_no_value(
        self, project, monkeypatch
    ):
        import os

        (project / ".env").write_text(
            "GROQ_API_KEY=gsk-x\n", encoding="utf-8"
        )  # nothing for OpenAI
        monkeypatch.setenv("OPENAI_API_KEY", "")
        problems = check_model("openai:gpt-5-mini").problems
        assert os.environ["OPENAI_API_KEY"] == ""
        assert any("OPENAI_API_KEY is exported but empty" in p for p in problems)
        assert any("an empty one is filled from the file" in p for p in problems)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _out(result) -> str:
    out = result.output
    try:
        out += result.stderr
    except (ValueError, AttributeError):
        pass
    return out


class TestModelsCli:
    def test_list(self, clean_env, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")  # Rich wraps narrow tables; keep cells whole
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0, _out(result)
        text = _out(result)
        assert "foundry" in text and "bedrock" in text and "OpenAI-compatible" in text
        assert "pip install" not in text

    def test_check_missing_exits_1_and_explains(self, clean_env):
        result = runner.invoke(app, ["models", "check", "azure:chat-prod"])
        assert result.exit_code == 1
        text = _out(result)
        assert "AZURE_OPENAI_ENDPOINT" in text and "MISSING" in text and "Not usable" in text

    def test_check_explains_an_exported_but_empty_variable(self, clean_env, monkeypatch):
        """The docs promise `models check` says which happened: unset, or
        exported but empty. Every problem check_model() diagnoses is printed."""
        monkeypatch.setenv("COLUMNS", "300")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        result = runner.invoke(app, ["models", "check", "openai:gpt-5-mini"])
        assert result.exit_code == 1
        text = _out(result)
        assert "OPENAI_API_KEY: MISSING (OPENAI_API_KEY is exported but empty" in text
        assert "unset it or give it a value" in text
        assert "  - OPENAI_API_KEY is exported but empty, which counts as not set" in text
        assert "  - put OPENAI_API_KEY in a .env file" in text
        monkeypatch.setenv("GOOGLE_API_KEY", "")  # an alias exported empty is named too
        monkeypatch.setenv("GEMINI_API_KEY", "")
        result = runner.invoke(app, ["models", "check", "gemini:gemini-2.5-pro"])
        assert "GOOGLE_API_KEY: MISSING (GOOGLE_API_KEY is exported but empty" in _out(result)

    def test_check_shows_the_route(self, clean_env, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "k")
        result = runner.invoke(app, ["models", "check", "groq:llama-3.3-70b-versatile"])
        assert result.exit_code == 0, _out(result)
        assert "https://api.groq.com/openai/v1" in _out(result).replace("\n", "")
        assert "Usable." in _out(result)

    def test_check_unknown_prefix_explains_inference(self, clean_env):
        result = runner.invoke(app, ["models", "check", "gpt-5-mini"])
        assert result.exit_code == 0
        assert "handed to LangChain" in _out(result)

    def test_check_ping_uses_a_real_model_call(self, clean_env, monkeypatch):
        from unittest.mock import AsyncMock

        from promptise import models_cli

        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.setattr(models_cli, "_ping", AsyncMock(return_value="'ok'"))
        result = runner.invoke(app, ["models", "check", "openai:gpt-5-mini", "--ping"])
        assert result.exit_code == 0, _out(result)
        assert "replied 'ok'" in _out(result)

    def test_env(self):
        result = runner.invoke(app, ["models", "env", "azure"])
        assert result.exit_code == 0
        assert "export AZURE_OPENAI_ENDPOINT=" in result.output
        assert runner.invoke(app, ["models", "env", "nope"]).exit_code == 2


def test_docs_provider_table_matches_the_registry():
    """docs/.snippets/providers-table.md is included on the docs pages; it must be exact."""
    gen = Path("docs/.snippets/gen_providers_table.py")
    spec = importlib.util.spec_from_file_location("gen_providers_table", gen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = Path("docs/.snippets/providers-table.md").read_text(encoding="utf-8")
    assert committed == module.render(), (
        "provider table is stale — run: .venv/bin/python docs/.snippets/gen_providers_table.py"
    )
    for p in PROVIDERS:
        for v in p.env:
            assert f"`{v.name}`" in committed, v.name


class TestEverySurfaceUsesTheRegistry:
    """Aliases, .env and Model must work wherever a model string is accepted."""

    def test_fallback_chain_resolves_aliases_and_model_objects(self, clean_env, monkeypatch):
        from promptise import FallbackChain

        monkeypatch.setenv("GROQ_API_KEY", "k")
        chain = FallbackChain(
            [
                Model(
                    "gpt-4o",
                    provider="azure",
                    deployment="chat-prod",
                    endpoint="https://r.openai.azure.com/",
                    api_key="k",
                    api_version="2024-10-21",
                ),
                "groq:llama-3.3-70b-versatile",
            ]
        )
        chain._ensure_resolved()
        assert [type(m).__name__ for m in chain._resolved] == ["AzureChatOpenAI", "ChatOpenAI"]
        assert chain._resolved[1].openai_api_base == "https://api.groq.com/openai/v1"

    def test_fallback_chain_error_is_actionable(self, clean_env):
        from promptise import FallbackChain

        with pytest.raises(ModelSetupError, match="GROQ_API_KEY"):
            FallbackChain(["groq:llama-3.3-70b-versatile"])._ensure_resolved()

    @pytest.mark.asyncio
    async def test_per_node_model_override_resolves_aliases(self, clean_env, monkeypatch):
        from langchain_core.messages import HumanMessage

        from promptise.engine.nodes import PromptNode
        from promptise.engine.state import GraphState

        monkeypatch.setenv("GROQ_API_KEY", "k")
        captured = {}

        def fake_resolve(spec, **kw):
            captured["spec"] = spec
            raise RuntimeError("stop here")  # resolution reached the registry; no call is made

        monkeypatch.setattr("promptise.models.resolve_model", fake_resolve)
        node = PromptNode("n", instructions="x", model_override="groq:llama-3.3-70b-versatile")
        result = await node.execute(GraphState(messages=[HumanMessage(content="hi")]), {})
        assert captured["spec"] == "groq:llama-3.3-70b-versatile"
        assert "stop here" in (result.error or "")
