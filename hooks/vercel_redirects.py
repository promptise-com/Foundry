"""Real 301 redirects for the docs.

The redirects plugin writes a small page that forwards with a meta refresh,
which is all GitHub Pages can do, and search engines treat that as a weak hint.
This hook writes ``vercel.json`` into the built site with the same map, so a
moved page answers with a real 301. ``redirect_maps`` in ``mkdocs.yml`` stays the
only list of moved pages.

``trailingSlash`` keeps one URL per page, because the docs' relative links need
the slash. Copies on ``*.vercel.app`` are kept out of search results.
"""

import json
from pathlib import Path


def _url(path: str) -> str:
    """The URL of a docs source path: ``blog/post.md`` is ``/blog/post/``."""
    if "://" in path:
        return path
    stem = path.removesuffix(".md").removesuffix("index").rstrip("/")
    return f"/{stem}/" if stem else "/"


def vercel_config(redirect_maps: dict[str, str]) -> dict:
    return {
        "trailingSlash": True,
        "redirects": [
            {"source": _url(old), "destination": _url(new), "statusCode": 301}
            for old, new in sorted(redirect_maps.items())
        ],
        "headers": [
            {
                "source": "/(.*)",
                "has": [{"type": "host", "value": r".*\.vercel\.app"}],
                "headers": [{"key": "X-Robots-Tag", "value": "noindex"}],
            }
        ],
    }


def on_post_build(config):
    redirect_maps = config["plugins"]["redirects"].config["redirect_maps"]
    path = Path(config["site_dir"]) / "vercel.json"
    path.write_text(json.dumps(vercel_config(redirect_maps), indent=2) + "\n", encoding="utf-8")
