"""The docs' vercel.json turns the redirects plugin's map into real 301s."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "vercel_redirects", ROOT / "hooks" / "vercel_redirects.py"
)
vercel_redirects = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vercel_redirects)


def test_a_moved_post_gets_a_301_to_its_guide():
    target = "https://www.promptise.com/guides/view/langchain-alternatives"
    config = vercel_redirects.vercel_config({"blog/langchain-alternative.md": target})
    assert config["redirects"] == [
        {"source": "/blog/langchain-alternative/", "destination": target, "statusCode": 301}
    ]


def test_a_page_moved_inside_the_docs_uses_its_url():
    config = vercel_redirects.vercel_config({"core/agents/network-server.md": "runtime/index.md"})
    assert config["redirects"] == [
        {"source": "/core/agents/network-server/", "destination": "/runtime/", "statusCode": 301}
    ]


def test_every_url_has_one_form_and_vercel_copies_stay_out_of_search():
    config = vercel_redirects.vercel_config({})
    assert config["trailingSlash"] is True
    assert config["headers"][0]["headers"] == [{"key": "X-Robots-Tag", "value": "noindex"}]
