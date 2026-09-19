"""Pytest configuration and fixtures."""

# ── Test-suite stability hygiene ──────────────────────────────────────────
# Must run BEFORE any test imports torch / transformers / sentence-transformers.
#
# Those ML libraries (used by guardrails, embeddings, RAG) spin up thread and
# process parallelism. Combined with the fork()s that later tests perform
# (subprocess execution, multiprocessing in runtime tests), that parallelism can
# kill the multiprocessing ``resource_tracker`` ("process died unexpectedly") and
# leave subprocess spawning in a degraded state — which made a handful of
# subprocess-based tests (shell interpolation, shell hooks) flake intermittently
# only when run as part of the full suite. Pinning single-threaded, fork-safe
# defaults removes that contention and makes the suite deterministic. This is
# standard hygiene for ML-backed test suites and affects tests only.
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# CLI output under test must be plain text: on GitHub Actions Rich/Typer
# colourise output (FORCE_COLOR / GITHUB_ACTIONS), which interleaves escape
# sequences inside the phrases the CLI tests look for. NO_COLOR switches colour
# off for Rich and Click; typer forces a terminal under GITHUB_ACTIONS unless
# told not to. (Not TERM=dumb: Rich then pins the width to 80 and ignores the
# COLUMNS the CLI tests set on their CliRunner to keep phrases on one line.)
os.environ["NO_COLOR"] = "1"
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
os.environ.pop("FORCE_COLOR", None)

from dotenv import load_dotenv  # noqa: E402

# Load environment variables from .env file
load_dotenv()
