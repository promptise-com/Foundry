"""Bring your own model — one install, any provider, credentials in code or in ``.env``.

Every place Promptise takes a model (``build_agent(model=...)``, ``.superagent``
and ``.agent`` files, ``promptise mcpcast --model``, the CLI) accepts one of:

1. **A string** — ``"provider:model"``; credentials come from the environment
   or a ``.env`` file::

       agent = await build_agent(model="azure:chat-prod", servers=...)

2. **A** :class:`Model` — provider, model and credentials **in code**, the way
   LangChain's ``init_chat_model`` takes them, with the same words for every
   provider (``provider``, ``model``, ``deployment``, ``api_key``, ``endpoint``,
   ``api_version``, ``region``, ``project``) and ``extra`` for anything
   provider-specific::

       from promptise import Model, build_agent

       agent = await build_agent(
           model=Model(
               "gpt-4o",                        # what the model is
               provider="azure",
               deployment="chat-prod",          # what you named it in Azure AI Foundry
               endpoint="https://my-resource.openai.azure.com/",
               api_key="...",
               api_version="2024-10-21",
           ),
           servers=...,
       )

3. **A config file** — the same fields under ``model:`` in a ``.superagent`` file.

Any LangChain ``BaseChatModel`` instance is accepted too, for full control.

**No extra packages.**  Every provider is reached through its OpenAI-compatible
endpoint using ``langchain-openai``, which is part of the core install — Groq,
Gemini, Bedrock, Vertex AI, Mistral, DeepSeek, xAI, Together, Fireworks,
Cohere, OpenRouter, Perplexity, NVIDIA, Hugging Face, Ollama and the Azure AI
Foundry catalog all work with ``pip install promptise`` alone.  OpenAI, Azure
OpenAI and Anthropic use their native integrations, which are core as well.
A provider's native LangChain integration can be opted into with
``Model(..., native=True)`` when it is installed.

**Where secrets live.**  A ``.env`` file in the working directory (or a
parent, up to the project root — the directory holding ``pyproject.toml`` or
``.git``) is loaded before a provider's variables are checked — the same file
the ``promptise`` CLI reads — so ``python my_agent.py`` and ``promptise run``
see the same keys.  Only a regular file — on POSIX one that you own and
that nobody else can write — is loaded (anything else is skipped with a
warning naming it); variables already in the environment always win over
the file, and nothing is loaded when ``PROMPTISE_NO_DOTENV=1``.
:func:`dotenv_origin` tells which file supplied a variable.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin

__all__ = [
    "PROVIDERS",
    "EnvVar",
    "Model",
    "ModelCheck",
    "ModelSetupError",
    "Provider",
    "check_model",
    "dotenv_origin",
    "env_template",
    "find_provider",
    "load_dotenv_if_present",
    "parse_model",
    "resolve_model",
]

WORDS: tuple[str, ...] = ("deployment", "api_key", "endpoint", "api_version", "region", "project")
"""The configuration words shared by every provider (see :class:`Model`)."""

_SETTINGS: tuple[str, ...] = ("temperature", "max_tokens", "timeout")


class ModelSetupError(RuntimeError):
    """A model cannot be used yet — the message says exactly what to do."""


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------

_dotenv_loaded: str | None = None
_dotenv_filled: dict[str, tuple[str, str]] = {}
"""Variable name → ``(.env file, value)`` it was filled with (see :func:`dotenv_origin`)."""
_dotenv_skipped: set[str] = set()
"""Unsafe ``.env`` files already warned about — once per process, not once per lookup."""

_PROJECT_ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git")
"""A directory holding one of these is the project root: the search for ``.env`` ends there."""

_DOTENV_FIX = "fix its permissions/encoding, move it, or set PROMPTISE_NO_DOTENV=1"


def _stat_dotenv(candidate: Path) -> os.stat_result | None:
    """``stat`` of *candidate* (following a symlink), or ``None`` when nothing is there.

    Raises:
        ModelSetupError: For any other failure — a directory that cannot be
            traversed, a symlink loop — naming the path and the ways out.
    """
    try:
        return candidate.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        raise ModelSetupError(f"cannot read {candidate}: {exc} — {_DOTENV_FIX}") from exc


def _unsafe_dotenv(candidate: Path, st: os.stat_result) -> str | None:
    """Why the existing file *candidate* (with ``stat`` *st*) must not be loaded, or ``None``.

    Only a regular file is safe, and on POSIX only one owned by the current
    user that nobody else can write: a ``.env`` planted in a shared ancestor
    such as ``/tmp`` by another account, or one anybody can edit, could
    otherwise redirect an endpoint or fill a key silently.  A symbolic link
    counts as the file it points to *and* as the link itself.  Windows
    synthesises the mode bits (every writable file reads as ``0o666``) and
    has no uid, so only the regular-file check applies there.
    """
    if not stat.S_ISREG(st.st_mode):
        return "not a regular file"
    if os.name != "posix":
        return None
    if st.st_mode & stat.S_IWOTH:
        return f"world-writable (mode {stat.S_IMODE(st.st_mode):o}) — run: chmod o-w {candidate}"
    if hasattr(os, "geteuid"):
        me = os.geteuid()
        owners = {st.st_uid}
        if candidate.is_symlink():
            owners.add(candidate.lstat().st_uid)
        for uid in sorted(owners):
            if uid != me:
                return f"owned by uid {uid}, not by you (uid {me})"
    return None


def _read_dotenv(path: str) -> dict[str, str | None]:
    """The parsed contents of the ``.env`` file at *path*.

    Raises:
        ModelSetupError: When the file cannot be read (permissions, a
            symlink loop, …) or is not UTF-8 — naming the file and the ways
            out, instead of a raw ``PermissionError``/``UnicodeDecodeError``
            from deep inside :func:`resolve_model` or the CLI.
    """
    from dotenv import dotenv_values

    try:
        return dotenv_values(path, encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelSetupError(f"cannot read {path}: {exc} — {_DOTENV_FIX}") from exc


def load_dotenv_if_present(*, cwd: str | os.PathLike[str] | None = None) -> str | None:
    """Load the nearest safe ``.env`` file from *cwd* (default: the working directory) upward.

    The search walks from *cwd* through its parents and stops after the
    project root — the first directory holding ``pyproject.toml`` or
    ``.git`` — so a file above the project (``/tmp/.env`` on a shared
    host, another user's home) is never consulted.  A candidate is loaded
    only when it is a regular file and, on POSIX, owned by you and not
    world-writable; anything else is skipped with a :class:`UserWarning`
    naming the file (once per process) and the search continues upward.

    A variable that is already set to a non-empty value is never
    overwritten.  A variable exported as an **empty string** (``export
    OPENAI_API_KEY=`` left in a shell profile) counts as unset everywhere in
    this module, so the file's non-empty value fills it — python-dotenv
    alone would skip it and the key would look "not set" although it is in
    the file.  Empty values in the file set nothing.  :func:`dotenv_origin`
    reports which file filled a variable.

    Returns the path that was loaded, or ``None``.  Set
    ``PROMPTISE_NO_DOTENV=1`` to disable (tests and hermetic deployments).
    Loading happens once per process per file; call again after
    ``os.chdir`` to pick up another one.

    Raises:
        ModelSetupError: When the file that would be loaded cannot be read
            or is not UTF-8 — the message names it and says what to do.
    """
    global _dotenv_loaded
    if os.environ.get("PROMPTISE_NO_DOTENV"):
        return None

    start = Path(cwd).absolute() if cwd is not None else Path.cwd()
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        st = _stat_dotenv(candidate)
        if st is not None:
            problem = _unsafe_dotenv(candidate, st)
            if problem is not None:
                if str(candidate) not in _dotenv_skipped:
                    _dotenv_skipped.add(str(candidate))
                    warnings.warn(f"ignoring {candidate}: {problem}", stacklevel=2)
            else:
                path = str(candidate)
                if _dotenv_loaded != path:
                    for key, value in _read_dotenv(path).items():
                        if value and not os.environ.get(key):
                            os.environ[key] = value
                            _dotenv_filled[key] = (path, value)
                    _dotenv_loaded = path
                return path
        if any(os.path.exists(os.path.join(directory, m)) for m in _PROJECT_ROOT_MARKERS):
            break  # the project root: a .env above it is not this project's
    return None


def dotenv_origin(name: str) -> str | None:
    """The ``.env`` file that supplied the current value of variable *name*, or ``None``.

    ``None`` means the variable came from the environment itself (or is not
    set at all): :func:`load_dotenv_if_present` records a file only for the
    variables it filled.
    """
    entry = _dotenv_filled.get(name)
    if entry is not None and os.environ.get(name) == entry[1]:
        return entry[0]
    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvVar:
    """One environment variable a provider reads."""

    name: str
    where: str
    """Where to find the value (the provider's console page, a CLI command…)."""
    word: str
    """The :class:`Model` word this variable feeds (``api_key``, ``endpoint``…);
    a value given in code for that word makes the variable unnecessary."""
    required: bool = True
    example: str = ""
    aliases: tuple[str, ...] = ()
    """Other variable names accepted for the same value (``GEMINI_API_KEY``)."""

    def value(self) -> str | None:
        """The value the environment currently holds for this variable, or ``None``.

        The first of :attr:`name` and :attr:`aliases` set to a non-empty string
        wins; a variable exported but empty counts as not set (which is why
        :func:`load_dotenv_if_present` fills empty ones from a ``.env`` file).
        """
        for name in (self.name, *self.aliases):
            if os.environ.get(name):
                return os.environ[name]
        return None


@dataclass(frozen=True)
class Provider:
    """How to reach one model provider."""

    key: str
    """The canonical prefix (``azure_openai``)."""
    title: str
    aliases: tuple[str, ...]
    """Friendly prefixes users may type instead of *key* (``azure``)."""
    env: tuple[EnvVar, ...] = ()
    example: str = ""
    """A complete example model string."""
    model_hint: str = ""
    """What the model part means for this provider."""
    notes: str = ""
    docs: str = ""
    base_url: str | None = None
    """OpenAI-compatible endpoint, with ``{endpoint}``, ``{region}`` and
    ``{project}`` placeholders filled from the words.  ``None`` for the three
    providers reached through their native core integration."""
    query: dict[str, str] = field(default_factory=dict)
    """Query parameters every request carries (Azure AI Foundry's ``api-version``)."""
    key_optional: bool = False
    """The endpoint works without a key (a local Ollama)."""
    native: str | None = None
    """LangChain ``init_chat_model`` provider key of the native integration."""
    native_package: str | None = None
    """Importable module of that integration (``langchain_groq``); ``None`` when
    it is part of the core install."""
    native_kwargs: dict[str, str | None] = field(default_factory=dict)
    """How the words map onto the native integration's keyword arguments."""

    @property
    def prefixes(self) -> tuple[str, ...]:
        """Every prefix that selects this provider in a model string.

        :attr:`key` first, then :attr:`aliases` — the strings
        :func:`find_provider` matches a ``prefix:model`` spec against.
        """
        return (self.key, *self.aliases)

    @property
    def display(self) -> str:
        """The prefix shown in docs and messages (a friendly alias for the
        LangChain-style keys, else the key itself)."""
        return _DISPLAY.get(self.key, self.key)

    @property
    def route(self) -> str:
        """``"native"`` (core integration) or ``"openai-compatible"``."""
        return "native" if self.base_url is None else "openai-compatible"

    @property
    def native_installed(self) -> bool:
        """Whether the native LangChain integration can be imported right now.

        ``True`` when the integration ships with the core install
        (:attr:`native_package` is ``None``); otherwise whether that package
        (``langchain_groq``…) is installed — checked without importing it.
        """
        if self.native_package is None:
            return True
        return importlib.util.find_spec(self.native_package) is not None

    def words(self) -> tuple[str, ...]:
        """The words this provider understands."""
        allowed = {v.word for v in self.env}
        if self.key == "azure_openai":
            allowed.add("deployment")
        if self.base_url is not None or self.key in ("openai", "anthropic"):
            allowed.add("endpoint")  # any OpenAI-compatible route can be pointed elsewhere
        if self.key == "azure_ai":
            allowed.add("api_version")
        return tuple(w for w in WORDS if w in allowed)

    def missing(self, provided: set[str]) -> list[EnvVar]:
        """Required variables that are neither set nor covered by a provided word."""
        return [v for v in self.env if v.required and v.word not in provided and v.value() is None]


def _key(name: str, where: str, **kw: Any) -> EnvVar:
    return EnvVar(name, where, "api_key", **kw)


_OPENAI_COMPAT_NATIVE: dict[str, str | None] = {"api_key": "api_key", "endpoint": "base_url"}

PROVIDERS: tuple[Provider, ...] = (
    Provider(
        "openai",
        "OpenAI",
        ("gpt",),
        env=(
            _key("OPENAI_API_KEY", "platform.openai.com → API keys", example="sk-..."),
            EnvVar(
                "OPENAI_BASE_URL",
                "only for an OpenAI-compatible server (vLLM, LM Studio, a gateway): its /v1 URL",
                "endpoint",
                required=False,
                example="http://localhost:8000/v1",
            ),
        ),
        example="openai:gpt-5-mini",
        model_hint="the model name",
        notes=(
            "Also the provider for any self-hosted or third-party server that speaks the "
            "OpenAI chat API: set endpoint= (or OPENAI_BASE_URL) to its /v1 URL."
        ),
        docs="https://platform.openai.com/docs/models",
        native="openai",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
    Provider(
        "azure_openai",
        "Azure OpenAI (OpenAI models deployed in Azure AI Foundry)",
        ("azure", "azure-openai", "azureopenai", "aoai"),
        env=(
            EnvVar(
                "AZURE_OPENAI_ENDPOINT",
                "Azure AI Foundry portal → your resource → Overview → Endpoint "
                "(https://<resource>.openai.azure.com/, no path)",
                "endpoint",
                example="https://my-resource.openai.azure.com/",
            ),
            _key(
                "AZURE_OPENAI_API_KEY",
                "Azure AI Foundry portal → your resource → Keys and Endpoint → KEY 1 "
                "(or Entra ID: pass extra={'azure_ad_token_provider': ...})",
            ),
            EnvVar(
                "OPENAI_API_VERSION",
                "the REST API version your deployment supports, e.g. 2024-10-21 "
                "(Azure docs → 'API version lifecycle')",
                "api_version",
                example="2024-10-21",
            ),
        ),
        example="azure:chat-prod",
        model_hint=(
            "in the string form, your DEPLOYMENT name (Foundry → Deployments → Name); with "
            "Model(...), the model name (gpt-4o) — the deployment goes in deployment="
        ),
        notes=(
            "Azure routes requests by deployment name, not model name: "
            "Model('gpt-4o', provider='azure', deployment='chat-prod', ...)."
        ),
        docs="https://learn.microsoft.com/azure/ai-services/openai/",
        native="azure_openai",
        native_kwargs={
            "deployment": "azure_deployment",
            "api_key": "api_key",
            "endpoint": "azure_endpoint",
            "api_version": "api_version",
        },
    ),
    Provider(
        "azure_ai",
        "Azure AI Foundry model catalog (Llama, Mistral, DeepSeek, Phi, Cohere, …)",
        ("foundry", "azure-ai", "azureai", "ai-foundry"),
        env=(
            EnvVar(
                "AZURE_INFERENCE_ENDPOINT",
                "Azure AI Foundry portal → Models + endpoints → your deployment → Endpoint "
                "(https://<resource>.services.ai.azure.com/models)",
                "endpoint",
                example="https://my-resource.services.ai.azure.com/models",
            ),
            _key(
                "AZURE_INFERENCE_CREDENTIAL",
                "Azure AI Foundry portal → Models + endpoints → your deployment → Key",
            ),
        ),
        example="foundry:Llama-3.3-70B-Instruct",
        model_hint="the deployment name shown in Models + endpoints",
        notes=(
            "For catalog models served by Azure AI Foundry's inference endpoint. OpenAI "
            "models deployed in Foundry use provider='azure' instead. The Azure AI Model "
            "Inference API is OpenAI-compatible; requests carry api-version="
            "2024-05-01-preview unless api_version= says otherwise."
        ),
        docs="https://learn.microsoft.com/azure/ai-foundry/",
        base_url="{endpoint}",
        query={"api-version": "2024-05-01-preview"},
        native="azure_ai",
        native_package="langchain_azure_ai",
        native_kwargs={"api_key": "credential", "endpoint": "endpoint"},
    ),
    Provider(
        "anthropic",
        "Anthropic Claude",
        ("claude",),
        env=(_key("ANTHROPIC_API_KEY", "console.anthropic.com → API keys", example="sk-ant-..."),),
        example="anthropic:claude-sonnet-4-5",
        model_hint="the model name",
        docs="https://docs.anthropic.com/en/docs/about-claude/models",
        native="anthropic",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
    Provider(
        "google_genai",
        "Google Gemini (AI Studio API key)",
        ("gemini", "google", "google-genai", "genai"),
        env=(
            _key(
                "GOOGLE_API_KEY",
                "aistudio.google.com → Get API key",
                example="AIza...",
                aliases=("GEMINI_API_KEY",),
            ),
        ),
        example="gemini:gemini-2.5-pro",
        model_hint="the model name",
        notes="For Gemini through Google Cloud (no API key, ADC) use provider='vertex'.",
        docs="https://ai.google.dev/gemini-api/docs/openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        native="google_genai",
        native_package="langchain_google_genai",
        native_kwargs={"api_key": "google_api_key", "endpoint": None},
    ),
    Provider(
        "google_vertexai",
        "Google Vertex AI (Gemini and Model Garden via Google Cloud)",
        ("vertex", "vertexai", "google-vertex", "gcp"),
        env=(
            EnvVar(
                "GOOGLE_CLOUD_PROJECT",
                "your GCP project id (console.cloud.google.com, top bar)",
                "project",
                example="my-project-123",
            ),
            EnvVar(
                "GOOGLE_CLOUD_LOCATION",
                "the region of your Vertex AI endpoint",
                "region",
                required=False,
                example="us-central1",
            ),
            _key(
                "GOOGLE_OAUTH_ACCESS_TOKEN",
                "an OAuth access token: `gcloud auth print-access-token` (valid ~1 h, never "
                "refreshed — for one-off runs); not needed when google-auth is installed and "
                "Application Default Credentials are configured — the token is minted for "
                "you and refreshed before it expires",
                required=False,
            ),
        ),
        example="vertex:gemini-2.5-pro",
        model_hint="the model name (a bare Gemini name is prefixed google/ for you)",
        notes=(
            "Authenticates with a Google OAuth access token. With google-auth installed "
            "(`pip install google-auth`) and `gcloud auth application-default login` done, "
            "the token is minted automatically and refreshed before it expires, so a "
            "long-running agent keeps working; a token set by hand (GOOGLE_OAUTH_ACCESS_TOKEN "
            "or api_key=) is used as given and expires after about an hour."
        ),
        docs="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library",
        base_url=(
            "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/endpoints/openapi"
        ),
        native="google_vertexai",
        native_package="langchain_google_vertexai",
        native_kwargs={
            "api_key": None,
            "endpoint": None,
            "project": "project",
            "region": "location",
        },
    ),
    Provider(
        "bedrock",
        "Amazon Bedrock",
        ("aws", "amazon"),
        env=(
            EnvVar(
                "AWS_DEFAULT_REGION",
                "the region your Bedrock models are enabled in",
                "region",
                example="us-east-1",
                aliases=("AWS_REGION",),
            ),
            _key(
                "AWS_BEARER_TOKEN_BEDROCK",
                "AWS console → Amazon Bedrock → API keys → Generate (a Bedrock API key; "
                "long-term keys start with ABSK)",
            ),
        ),
        example="bedrock:anthropic.claude-sonnet-4-20250514-v1:0",
        model_hint="the Bedrock model id or an inference-profile id/ARN",
        notes=(
            "Uses Bedrock's OpenAI-compatible endpoint with a Bedrock API key — no AWS SDK "
            "needed. For IAM/SSO credentials install langchain-aws and pass native=True."
        ),
        docs="https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html",
        base_url="https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
        native="bedrock_converse",
        native_package="langchain_aws",
        native_kwargs={"api_key": None, "endpoint": "endpoint_url", "region": "region_name"},
    ),
    Provider(
        "ollama",
        "Ollama (local models)",
        ("local",),
        env=(
            EnvVar(
                "OLLAMA_HOST",
                "only if Ollama is not on the default http://localhost:11434",
                "endpoint",
                required=False,
                example="http://localhost:11434",
            ),
        ),
        example="ollama:llama3.1",
        model_hint="a model you have pulled (`ollama pull llama3.1`)",
        notes="No API key. The model must support tool calling to drive MCP tools.",
        docs="https://ollama.com/library",
        base_url="{endpoint}/v1",
        key_optional=True,
        native="ollama",
        native_package="langchain_ollama",
        native_kwargs={"api_key": None, "endpoint": "base_url"},
    ),
    Provider(
        "groq",
        "Groq",
        (),
        env=(_key("GROQ_API_KEY", "console.groq.com → API Keys"),),
        example="groq:llama-3.3-70b-versatile",
        model_hint="the model name",
        docs="https://console.groq.com/docs/models",
        base_url="https://api.groq.com/openai/v1",
        native="groq",
        native_package="langchain_groq",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
    Provider(
        "mistralai",
        "Mistral AI",
        ("mistral",),
        env=(_key("MISTRAL_API_KEY", "console.mistral.ai → API Keys"),),
        example="mistral:mistral-large-latest",
        model_hint="the model name",
        docs="https://docs.mistral.ai/getting-started/models/",
        base_url="https://api.mistral.ai/v1",
        native="mistralai",
        native_package="langchain_mistralai",
        native_kwargs={"api_key": "api_key", "endpoint": "endpoint"},
    ),
    Provider(
        "deepseek",
        "DeepSeek",
        (),
        env=(_key("DEEPSEEK_API_KEY", "platform.deepseek.com → API keys"),),
        example="deepseek:deepseek-chat",
        model_hint="the model name",
        docs="https://api-docs.deepseek.com/",
        base_url="https://api.deepseek.com/v1",
        native="deepseek",
        native_package="langchain_deepseek",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
    Provider(
        "xai",
        "xAI Grok",
        ("grok",),
        env=(_key("XAI_API_KEY", "console.x.ai → API Keys"),),
        example="grok:grok-4",
        model_hint="the model name",
        docs="https://docs.x.ai/docs/models",
        base_url="https://api.x.ai/v1",
        native="xai",
        native_package="langchain_xai",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
    Provider(
        "together",
        "Together AI",
        (),
        env=(_key("TOGETHER_API_KEY", "api.together.ai → Settings → API Keys"),),
        example="together:meta-llama/Llama-3.3-70B-Instruct-Turbo",
        model_hint="the model path as listed by Together",
        docs="https://docs.together.ai/docs/serverless-models",
        base_url="https://api.together.xyz/v1",
        native="together",
        native_package="langchain_together",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
    Provider(
        "fireworks",
        "Fireworks AI",
        (),
        env=(_key("FIREWORKS_API_KEY", "fireworks.ai → API Keys"),),
        example="fireworks:accounts/fireworks/models/llama-v3p3-70b-instruct",
        model_hint="the full model path",
        docs="https://fireworks.ai/models",
        base_url="https://api.fireworks.ai/inference/v1",
        native="fireworks",
        native_package="langchain_fireworks",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
    Provider(
        "cohere",
        "Cohere",
        (),
        env=(_key("COHERE_API_KEY", "dashboard.cohere.com → API keys"),),
        example="cohere:command-a-03-2025",
        model_hint="the model name",
        docs="https://docs.cohere.com/docs/compatibility-api",
        base_url="https://api.cohere.ai/compatibility/v1",
        native="cohere",
        native_package="langchain_cohere",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
    Provider(
        "huggingface",
        "Hugging Face Inference Providers",
        ("hf",),
        env=(
            _key(
                "HF_TOKEN",
                "huggingface.co → Settings → Access Tokens",
                aliases=("HUGGINGFACEHUB_API_TOKEN",),
            ),
        ),
        example="hf:meta-llama/Llama-3.3-70B-Instruct",
        model_hint="the repository id (add :provider to pin an inference provider)",
        notes="A dedicated Inference Endpoint: set endpoint= to its URL.",
        docs="https://huggingface.co/docs/inference-providers",
        base_url="https://router.huggingface.co/v1",
        native="huggingface",
        native_package="langchain_huggingface",
        native_kwargs={"api_key": "huggingfacehub_api_token", "endpoint": "endpoint_url"},
    ),
    Provider(
        "nvidia",
        "NVIDIA NIM",
        ("nim",),
        env=(_key("NVIDIA_API_KEY", "build.nvidia.com → API key"),),
        example="nvidia:meta/llama-3.3-70b-instruct",
        model_hint="the model path",
        notes="A NIM you run yourself: set endpoint= to its /v1 URL.",
        docs="https://build.nvidia.com/models",
        base_url="https://integrate.api.nvidia.com/v1",
        native="nvidia",
        native_package="langchain_nvidia_ai_endpoints",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
    Provider(
        "openrouter",
        "OpenRouter",
        (),
        env=(_key("OPENROUTER_API_KEY", "openrouter.ai → Keys"),),
        example="openrouter:anthropic/claude-sonnet-4.5",
        model_hint="the provider/model path as listed by OpenRouter",
        docs="https://openrouter.ai/models",
        base_url="https://openrouter.ai/api/v1",
        native="openrouter",
        native_package="langchain_openrouter",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
    Provider(
        "perplexity",
        "Perplexity",
        (),
        env=(_key("PPLX_API_KEY", "perplexity.ai → Settings → API"),),
        example="perplexity:sonar-pro",
        model_hint="the model name",
        docs="https://docs.perplexity.ai/guides/model-cards",
        base_url="https://api.perplexity.ai",
        native="perplexity",
        native_package="langchain_perplexity",
        native_kwargs=_OPENAI_COMPAT_NATIVE,
    ),
)

_DISPLAY: dict[str, str] = {
    "azure_openai": "azure",
    "azure_ai": "foundry",
    "google_genai": "gemini",
    "google_vertexai": "vertex",
    "mistralai": "mistral",
}

_BY_PREFIX: dict[str, Provider] = {}
for _provider in PROVIDERS:
    for _prefix in _provider.prefixes:
        _BY_PREFIX[_prefix.lower()] = _provider


def find_provider(prefix: str) -> Provider | None:
    """The provider for a prefix or alias (``"azure"`` → Azure OpenAI), or ``None``."""
    norm = prefix.strip().lower()
    return (
        _BY_PREFIX.get(norm)
        or _BY_PREFIX.get(norm.replace("_", "-"))
        or _BY_PREFIX.get(norm.replace("-", "_"))
    )


def parse_model(spec: str) -> tuple[Provider | None, str, str]:
    """Split ``"prefix:model"`` into ``(provider, canonical_spec, model)``.

    A spec without a known prefix is returned with ``provider=None`` and left
    for LangChain to infer (``"gpt-5-mini"`` still works).
    """
    spec = spec.strip()
    if ":" in spec:
        prefix, _, model = spec.partition(":")
        provider = find_provider(prefix)
        if provider is not None:
            return provider, f"{provider.key}:{model}", model
    return None, spec, spec


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCheck:
    """The result of :func:`check_model`."""

    spec: str
    canonical: str
    provider: Provider | None
    model: str
    missing_env: list[EnvVar] = field(default_factory=list)
    native: bool = False

    @property
    def ok(self) -> bool:
        """``True`` when nothing stands in the way of building this model.

        A spec with no known prefix is always usable (LangChain infers the
        provider).  Otherwise every required variable is set or covered by a
        word given in code (:attr:`missing_env` is empty) and, when the
        native route was requested, its integration is installed.  When this
        is ``False``, :attr:`problems` says what to fix.
        """
        if self.provider is None:
            return True
        if self.native and not self.provider.native_installed:
            return False
        return not self.missing_env

    @property
    def problems(self) -> list[str]:
        """Human-readable problems, each with its fix."""
        out: list[str] = []
        p = self.provider
        if p is None:
            return out
        if self.native and not p.native_installed:
            out.append(
                f"native=True needs the {p.native_package} package → "
                f"pip install {(p.native_package or '').replace('_', '-')} "
                "(or drop native=True: the OpenAI-compatible route needs nothing)"
            )
        for var in self.missing_env:
            hint = f" (e.g. {var.example})" if var.example else ""
            empty = [n for n in (var.name, *var.aliases) if os.environ.get(n) == ""]
            if empty:
                out.append(
                    f"{empty[0]} is exported but empty, which counts as not set — "
                    f"unset it or give it a value — {var.where}{hint}"
                )
            else:
                out.append(f"{var.name} is not set — {var.where}{hint}")
        if self.missing_env:
            names = ", ".join(v.name for v in self.missing_env)
            words = ", ".join(f"{v.word}=" for v in self.missing_env)
            out.append(
                f"put {names} in a .env file next to your script (loaded automatically; a "
                f"variable already exported with a non-empty value wins, an empty one is "
                f"filled from the file), export it, or pass it in code: "
                f"Model(..., {words}) — see promptise models env {p.display}"
            )
        return out


def check_model(spec: str, provided: set[str] | None = None, *, native: bool = False) -> ModelCheck:
    """Diagnose a model string without calling anything.

    *provided* names the words the caller supplies in code (``{"api_key"}``);
    they count as satisfying the corresponding variables.
    """
    load_dotenv_if_present()
    provider, canonical, model = parse_model(spec)
    if provider is None:
        return ModelCheck(spec=spec, canonical=canonical, provider=None, model=model)
    return ModelCheck(
        spec=spec,
        canonical=canonical,
        provider=provider,
        model=model,
        missing_env=provider.missing(provided or set()),
        native=native,
    )


def _explain(provider: Provider, model: str, problems: list[str]) -> str:
    lines = [f"Cannot use model {provider.display}:{model!r} ({provider.title}) yet:"]
    lines.extend(f"  - {p}" for p in problems)
    if provider.model_hint:
        lines.append(f"  Model part means: {provider.model_hint}.")
    lines.append(f"  Example: {provider.example}")
    lines.append(f"  Diagnose any model string with: promptise models check {provider.example}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model — provider, model and credentials in code
# ---------------------------------------------------------------------------


@dataclass
class Model:
    """Provider, model and credentials **in code** — the LangChain shape.

    Positional ``model`` is what the model *is* (``"gpt-4o"``); ``provider``
    is where it runs (``"azure"``, ``"openai"``, ``"foundry"``, ``"bedrock"``…
    — any prefix or alias from :data:`PROVIDERS`, inferred from the model
    name like LangChain does when omitted); ``deployment`` is what you named
    it there (Azure OpenAI).  The remaining words are the same for every
    provider.  Anything else goes in ``extra`` verbatim.  A value given here
    counts as provided, so the matching environment variable is not required;
    ``None`` or a blank string (``api_key=""`` — what ``${OPENAI_API_KEY}``
    yields for a variable exported but empty) counts as *not* given, and the
    environment or ``.env`` supplies it.

    ``Model("azure:chat-prod")`` — the string form inside ``Model`` — means
    ``Model("chat-prod", provider="azure")``.

    ``repr()`` and ``str()`` never include ``api_key`` or ``extra``, so a
    ``Model`` is safe to log and to name in error messages.

    Args:
        model: The model name (Azure OpenAI: the underlying model, e.g.
            ``"gpt-4o"``; the request goes to ``deployment``).
        provider: Provider prefix or alias.  Omit it to let LangChain infer
            the provider from the model name (``gpt-…`` → OpenAI).
        deployment: Azure OpenAI deployment name — what you called the model
            when you deployed it in Azure AI Foundry.  Defaults to *model*.
        api_key: The provider's key (Bedrock: a Bedrock API key; Vertex AI: an
            OAuth access token; not applicable to a local Ollama).
        endpoint: Where to send requests — an Azure OpenAI resource endpoint,
            an Azure AI Foundry inference endpoint, or the ``/v1`` URL of any
            OpenAI-compatible server (vLLM, LM Studio, a proxy, a NIM…).
            For providers with a public API this overrides the default URL.
        api_version: Azure OpenAI REST API version (``"2024-10-21"``), or the
            Azure AI Foundry inference ``api-version``.
        region: Bedrock region or Vertex AI location.
        project: Google Cloud project id (Vertex AI).
        temperature: Sampling temperature.
        max_tokens: Completion token limit.
        timeout: Request timeout in seconds.
        native: Use the provider's native LangChain integration instead of
            its OpenAI-compatible endpoint (``langchain-aws`` for IAM auth on
            Bedrock, ``langchain-google-genai`` for Gemini-only features…).
            The package must be installed; nothing is installed for you.
        extra: Provider-specific keyword arguments passed through untouched
            (``{"azure_ad_token_provider": ...}``, ``{"default_headers": {...}}``).

    Example::

        Model("gpt-5-mini", provider="openai", api_key="sk-...")
        Model("gpt-4o", provider="azure", deployment="chat-prod",
              endpoint="https://r.openai.azure.com/", api_key="...", api_version="2024-10-21")
        Model("Llama-3.3-70B-Instruct", provider="foundry",
              endpoint="https://r.services.ai.azure.com/models", api_key="...")
        Model("anthropic.claude-sonnet-4-20250514-v1:0", provider="bedrock",
              region="us-east-1", api_key="ABSK...")
        Model("qwen2.5", provider="openai", endpoint="http://localhost:8000/v1", api_key="none")
    """

    model: str
    provider: str | None = None
    deployment: str | None = None
    # Never in repr()/str(): a Model ends up in error messages, logs, graph
    # history and event payloads whenever resolution fails.
    api_key: str | None = field(default=None, repr=False)
    endpoint: str | None = None
    api_version: str | None = None
    region: str | None = None
    project: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    native: bool = False
    # Provider extras carry tokens and token providers too (``default_headers``,
    # ``azure_ad_token_provider``…) — kept out of repr() as well.
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.provider is None and ":" in self.model:
            prefix, _, name = self.model.partition(":")
            if find_provider(prefix) is not None:
                self.provider, self.model = prefix, name
        if self.provider is not None and find_provider(self.provider) is None:
            known = ", ".join(p.display for p in PROVIDERS)
            raise ModelSetupError(
                f"unknown provider {self.provider!r}; use one of {known} (or an alias — "
                "`promptise models list` shows them)"
            )

    @property
    def provider_info(self) -> Provider | None:
        """The registry entry for ``provider`` (``None`` when left to inference)."""
        return find_provider(self.provider) if self.provider else None

    @property
    def spec(self) -> str:
        """The canonical ``"provider:model"`` string (or the bare model name)."""
        info = self.provider_info
        return f"{info.key}:{self.model}" if info else self.model

    def kwargs(self) -> dict[str, Any]:
        """Everything but the spec, as keyword arguments for :func:`resolve_model`.

        Words left ``None`` or blank are omitted, so the environment (or
        ``.env``) supplies them.
        """
        out: dict[str, Any] = {}
        for word in (*WORDS, *_SETTINGS):
            value = getattr(self, word)
            if _given(value):
                out[word] = value
        if self.native:
            out["native"] = True
        out.update(self.extra)
        return out

    def resolve(self) -> Any:
        """The LangChain chat model for this configuration."""
        return resolve_model(self.spec, **self.kwargs())


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _given(value: Any) -> bool:
    """Whether a word was given in code: not ``None`` and not a blank string.

    ``Model(api_key="")`` — what a ``${VAR}`` reference yields for a variable
    that is exported but empty — must not count as a credential, or it
    would shadow the environment check and reach the provider as an empty
    bearer token.
    """
    return value is not None and not (isinstance(value, str) and not value.strip())


def _unsupported(provider: Provider, word: str) -> str:
    hints = {
        (
            "deployment",
            None,
        ): "only Azure OpenAI addresses models by deployment name — put the name in model=",
        ("api_key", "ollama"): "a local Ollama needs no key; for a protected proxy pass "
        "extra={'default_headers': {...}}",
        ("api_version", None): "only Azure uses an API version",
        ("region", None): "only Bedrock and Vertex AI are regional",
        ("project", None): "only Vertex AI needs a Google Cloud project",
    }
    specific = (
        hints.get((word, provider.key))
        or hints.get((word, None))
        or ("pass provider-specific options in extra={...}")
    )
    return f"{provider.title} has no {word!r} setting: {specific}."


_VERTEX_TOKEN_FIX = (
    "configure Application Default Credentials (`gcloud auth application-default login`), "
    "or run `gcloud auth print-access-token` and set GOOGLE_OAUTH_ACCESS_TOKEN (or api_key=)"
)


def _vertex_token() -> Callable[[], str] | None:
    """A token provider backed by Application Default Credentials.

    Returns ``None`` when google-auth is not installed (the caller then
    explains how to obtain a token by hand).  Otherwise the credentials
    are refreshed once here — so no ADC, a revoked credential or a failing
    refresh raises :class:`ModelSetupError` naming the reason, chained to
    the google-auth exception, before any model is built — and the returned
    zero-argument callable yields a currently valid access token: the
    OpenAI-compatible client calls it before every request, and whenever
    google-auth reports the token expired (about an hour after it was
    minted) it is refreshed under a lock, so a long-running agent keeps
    working.  A refresh that fails later raises :class:`ModelSetupError`
    from the request that needed it.
    """
    if importlib.util.find_spec("google.auth") is None:
        return None
    import google.auth
    import google.auth.transport.requests

    request = google.auth.transport.requests.Request()
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    try:
        credentials, _ = google.auth.default(scopes=scopes)
        credentials.refresh(request)
    except Exception as exc:
        raise ModelSetupError(
            f"google-auth could not mint a Vertex AI access token "
            f"({type(exc).__name__}: {exc}) — {_VERTEX_TOKEN_FIX}"
        ) from exc
    if not credentials.token:
        return None
    lock = threading.Lock()

    def token() -> str:
        with lock:
            if not credentials.valid:
                try:
                    credentials.refresh(request)
                except Exception as exc:
                    raise ModelSetupError(
                        f"google-auth could not refresh the Vertex AI access token "
                        f"({type(exc).__name__}: {exc}) — {_VERTEX_TOKEN_FIX}"
                    ) from exc
            value = credentials.token
        if not value:
            raise ModelSetupError(
                f"google-auth refreshed the Vertex AI credentials but returned no access "
                f"token — {_VERTEX_TOKEN_FIX}"
            )
        return str(value)

    return token


def _openai_api_key_accepts_callable() -> bool:
    """Whether the installed langchain-openai takes a callable ``api_key``.

    ``ChatOpenAI`` 1.0.1 and later declare ``openai_api_key`` as ``SecretStr
    | None | Callable[[], str] | …`` and evaluate a callable before every
    request; older releases (the declared floor) take a string only.
    """
    from langchain_openai import ChatOpenAI

    info = ChatOpenAI.model_fields.get("openai_api_key")
    if info is None:
        return False
    pending: list[Any] = [info.annotation]
    while pending:
        item = pending.pop()
        if item is Callable or get_origin(item) is Callable:
            return True
        pending.extend(get_args(item))
    return False


def _pinned_vertex_token(token: Callable[[], str]) -> str:
    """The token as a plain string, for a langchain-openai that cannot refresh it.

    Compatibility for releases whose ``ChatOpenAI`` takes only a string
    ``api_key``: the token minted now is pinned for the life of the model
    and expires after about an hour — said out loud with a warning, never
    silently.
    """
    warnings.warn(
        "the installed langchain-openai takes only a string api_key, so the Vertex AI "
        "access token minted now is pinned for the life of this model and expires after "
        "about 1 h — upgrade langchain-openai (1.0.1 and later accept a callable api_key) "
        "so it is refreshed automatically",
        stacklevel=3,
    )
    return token()


def _fresh_http_clients(extra: dict[str, Any]) -> dict[str, Any]:
    """One ``httpx`` client pair for this resolved model — never shared across event loops.

    langchain-openai hands every model the same process-wide, cached ``httpx``
    clients. Their pooled connections belong to the event loop that opened
    them, so the second ``asyncio.run()`` in a process (the CLI's curation and
    then evaluation, the guided setup's Textual loop and then the lab's agent
    run) fails with ``RuntimeError: Event loop is closed`` on the first request.
    A client pair per model keeps each loop's connections its own. Clients the
    caller passed in ``extra`` win; the settings mirror the OpenAI SDK's own
    defaults (10-minute request timeout, 5-second connect, its pool limits) —
    the per-request timeout the SDK sets still applies.
    """
    if "http_client" in extra or "http_async_client" in extra:
        return {}
    import httpx

    timeout = httpx.Timeout(600.0, connect=5.0)
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
    return {
        "http_client": httpx.Client(timeout=timeout, limits=limits),
        "http_async_client": httpx.AsyncClient(timeout=timeout, limits=limits),
    }


OLLAMA_DEFAULT_HOST = "http://localhost:11434"
"""Where a local Ollama listens when ``OLLAMA_HOST`` / ``endpoint=`` say nothing."""


def ollama_host(value: str | None) -> str:
    """The Ollama server's origin from an ``OLLAMA_HOST``-style value.

    Accepts ``host:port`` without a scheme and a URL ending in ``/v1``;
    ``None`` or empty means :data:`OLLAMA_DEFAULT_HOST`.
    """
    host = (value or OLLAMA_DEFAULT_HOST).strip().rstrip("/")
    if not host.startswith("http"):
        host = f"http://{host}"
    return host[: -len("/v1")] if host.endswith("/v1") else host


class _Placeholders(dict[str, str]):
    """``str.format_map`` mapping that leaves an unknown word as ``{word}``."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def route_url(provider: Provider, words: dict[str, Any] | None = None) -> str | None:
    """The base URL the OpenAI-compatible route calls, or ``None`` for a native route.

    *words* given in code win; the provider's environment variables fill the
    rest.  Ollama's host defaults to :data:`OLLAMA_DEFAULT_HOST` and Vertex
    AI's region to ``us-central1``, as :func:`resolve_model` does.  A word
    still unknown (a required variable that is not set) stays a ``{word}``
    placeholder, so the result is always printable.
    """
    if provider.base_url is None:
        return None
    filled: dict[str, Any] = {w: v for w, v in (words or {}).items() if _given(v)}
    for var in provider.env:
        if var.word not in filled and var.value() is not None:
            filled[var.word] = var.value()
    if provider.key == "ollama":
        filled["endpoint"] = ollama_host(filled.get("endpoint"))
    if provider.key == "google_vertexai":
        filled.setdefault("region", "us-central1")
    if "endpoint" in filled and "{endpoint}" not in provider.base_url:
        return str(filled["endpoint"])  # a self-hosted server or proxy for this provider
    return provider.base_url.format_map(_Placeholders({k: str(v) for k, v in filled.items()}))


def resolve_model(spec: str, **kwargs: Any) -> Any:
    """Turn a ``"provider:model"`` string (plus optional words) into a chat model.

    Keyword arguments are the :class:`Model` words (``api_key``, ``endpoint``,
    ``deployment``, ``api_version``, ``region``, ``project``), the settings
    (``temperature``, ``max_tokens``, ``timeout``), ``native=True``, and any
    provider-specific extras, which are passed through.  A word that is
    ``None`` or a blank string counts as not given, so the environment (or
    ``.env``) supplies it — ``api_key=""`` never shadows the credential check.

    Raises:
        ModelSetupError: Naming each missing variable and where to find it,
            the word that does not apply to the provider, or a ``.env`` file
            that cannot be read.
    """
    from langchain.chat_models import init_chat_model

    native = bool(kwargs.pop("native", False))
    words = {w: kwargs.pop(w) for w in WORDS if _given(kwargs.get(w))}
    for w in WORDS:
        kwargs.pop(w, None)
    settings = {s: kwargs.pop(s) for s in _SETTINGS if kwargs.get(s) is not None}
    for s in _SETTINGS:
        kwargs.pop(s, None)
    extra = kwargs

    result = check_model(spec, set(words), native=native)
    provider, model = result.provider, result.model

    if provider is None:  # no prefix we know: LangChain's own inference
        try:
            return init_chat_model(model, **{**_native_kwargs_generic(words), **settings, **extra})
        except (ImportError, ValueError) as exc:
            known = ", ".join(p.display for p in PROVIDERS)
            raise ModelSetupError(
                f"{exc}\nPromptise understands these providers (run `promptise models list` "
                f"for aliases): {known}"
            ) from exc

    for word in words:
        if word not in provider.words():
            raise ModelSetupError(_unsupported(provider, word))
    if not result.ok:
        raise ModelSetupError(_explain(provider, model, result.problems))

    # Fill the words from the environment.
    for var in provider.env:
        if var.word not in words and var.value() is not None:
            words[var.word] = var.value()

    if provider.base_url is None or native:
        # Native integration (core for openai/azure_openai/anthropic; opt-in otherwise).
        mapped: dict[str, Any] = {}
        for word, value in words.items():
            name = provider.native_kwargs.get(word, word)
            if name is None:
                raise ModelSetupError(_unsupported(provider, word))
            mapped[name] = value
        if provider.key == "azure_openai" and "azure_deployment" not in mapped:
            mapped["azure_deployment"] = model
        assert provider.native is not None
        clients = (
            _fresh_http_clients(extra) if provider.native in ("openai", "azure_openai") else {}
        )
        try:
            return init_chat_model(
                f"{provider.native}:{model}", **{**mapped, **clients, **settings, **extra}
            )
        except ImportError as exc:
            raise ModelSetupError(
                f"{exc}\nDrop native=True to use the OpenAI-compatible route, which needs no package."
            ) from exc

    # OpenAI-compatible route through langchain-openai (core).
    api_key: str | Callable[[], str] | None = words.get("api_key")
    if provider.key == "google_vertexai" and api_key is None:
        token = _vertex_token()
        if token is None:
            raise ModelSetupError(
                _explain(
                    provider,
                    model,
                    [
                        "no OAuth access token: run `gcloud auth print-access-token` and set "
                        "GOOGLE_OAUTH_ACCESS_TOKEN (or api_key=), or install google-auth "
                        "(`pip install google-auth`) with Application Default Credentials "
                        "configured so a token is minted for you"
                    ],
                )
            )
        # Minted for you: the client calls the provider before every request, so
        # the token is refreshed before it expires (see _vertex_token) — unless
        # the installed langchain-openai takes only a string.
        api_key = token if _openai_api_key_accepts_callable() else _pinned_vertex_token(token)
    if api_key is None:
        if not provider.key_optional:
            raise ModelSetupError(_explain(provider, model, ["no API key"]))
        api_key = "not-needed"
    if provider.key == "google_vertexai" and "/" not in model:
        model = f"google/{model}"
    base_url = route_url(provider, words)
    assert base_url is not None
    query = dict(provider.query)
    if "api_version" in words and provider.query:
        query["api-version"] = str(words["api_version"])
    client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if query:
        client_kwargs["default_query"] = query
    clients = _fresh_http_clients(extra)
    return init_chat_model(f"openai:{model}", **{**client_kwargs, **clients, **settings, **extra})


def _native_kwargs_generic(words: dict[str, Any]) -> dict[str, Any]:
    """Words for an inferred provider: only the universal ones make sense."""
    out: dict[str, Any] = {}
    if "api_key" in words:
        out["api_key"] = words["api_key"]
    if "endpoint" in words:
        out["base_url"] = words["endpoint"]
    return out


def env_template(prefix: str) -> str:
    """Shell ``export`` lines for a provider, with placeholders and hints."""
    provider = find_provider(prefix)
    if provider is None:
        raise ModelSetupError(
            f"unknown provider {prefix!r}; run `promptise models list` to see them"
        )
    lines = [f"# {provider.title} — model string example: {provider.example}"]
    for var in provider.env:
        marker = "" if var.required else "  # optional"
        value = var.example or "..."
        lines.append(f"export {var.name}={value!s}{marker}")
        lines.append(f"#   ↳ {var.where}")
    if not provider.env:
        lines.append("# no environment variables required")
    return "\n".join(lines)
