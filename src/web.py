"""FastAPI web UI to browse daily briefs."""

from __future__ import annotations

from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from src.config import OUTPUT_DIR, PROJECT_ROOT

app = FastAPI(title="Macro Intelligence", version="0.2.0")


def _list_non_legacy(prefix: str) -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    pattern = f"{prefix}_*.html"
    files = [p for p in OUTPUT_DIR.glob(pattern) if "_legacy" not in p.stem]
    return sorted(files, reverse=True)


def _latest_from_triplet(prefix: str) -> Path | None:
    candidates = _list_non_legacy(prefix)
    return candidates[0] if candidates else None


def _list_layers() -> list[Path]:
    return _list_non_legacy("layer1")


def _list_briefs() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    # Keep archive links compatible with /brief/{date} routing.
    date_brief = re.compile(r"^brief_\d{4}-\d{2}-\d{2}\.html$")
    files = [p for p in OUTPUT_DIR.glob("brief_*.html") if date_brief.match(p.name)]
    return sorted(files, reverse=True)


@app.get("/")
def index():
    # Prefer static layer shells; fall back to dated variants, then legacy outputs.
    candidates: list[str] = [
        "synthesis_lite.html",
        "layer3.html",
        "layer2.html",
        "layer1.html",
        "synthesis.html",
        "brief_global.html",
        "flow_brief.html",
        "brief.html",
    ]
    for name in candidates:
        if (OUTPUT_DIR / name).exists():
            return RedirectResponse(url=f"/file/{name}", status_code=302)
    # Dated fallbacks (layer3_YYYY-MM-DD, synthesis_YYYY-MM-DD, brief_YYYY-MM-DD).
    for prefix in ("layer3", "synthesis", "brief_global", "flow_brief"):
        latest = _latest_from_triplet(prefix)
        if latest:
            return RedirectResponse(url=f"/file/{latest.name}", status_code=302)
    briefs = _list_briefs()
    if briefs:
        return RedirectResponse(url=f"/brief/{briefs[0].stem.replace('brief_', '')}", status_code=302)
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Macro Intelligence</title>
<style>
body {{ font-family: system-ui; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }}
</style></head>
<body>
<h1>Macro Intelligence</h1>
<p>No briefs yet. Run <code>python run.py</code>.</p>
</body></html>"""
    )


_ALLOWED_STATIC = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
}


@app.get("/file/{filename}")
def get_file(filename: str) -> FileResponse:
    path = OUTPUT_DIR / filename
    suffix = path.suffix.lower()
    if not path.exists() or suffix not in _ALLOWED_STATIC:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type=_ALLOWED_STATIC[suffix])


@app.get("/archive", response_class=HTMLResponse)
def archive() -> str:
    briefs = _list_briefs()
    links = "\n".join(
        f'<li><a href="/brief/{p.stem.replace("brief_", "")}">{p.name}</a></li>'
        for p in briefs[:30]
    )
    if not links:
        links = "<li>No briefs yet. Run <code>python run.py</code>.</li>"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Macro Intelligence — Archive</title>
<style>
body {{ font-family: system-ui; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }}
a {{ color: #4a6cf7; }}
</style></head>
<body>
<h1>Macro Intelligence</h1>
<p>Daily briefs in <code>{OUTPUT_DIR.relative_to(PROJECT_ROOT)}</code></p>
<ul>{links}</ul>
<p><a href="/docs">API docs</a></p>
</body></html>"""


@app.get("/brief/{brief_date}")
def get_brief(brief_date: str) -> FileResponse:
    path = OUTPUT_DIR / f"brief_{brief_date}.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Brief not found")
    return FileResponse(path, media_type="text/html")


@app.get("/api/briefs")
def api_list_briefs() -> dict:
    return {"briefs": [p.name for p in _list_briefs()]}


def main() -> None:
    import uvicorn

    uvicorn.run("src.web:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()
