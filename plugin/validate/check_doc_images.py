#!/usr/bin/env python3
"""Doc-image render guard — keeps every chart/image in docs/ RENDERABLE on GitHub.

Two failure classes break GitHub rendering, and this guard catches BOTH:

1. DETERMINISTIC broken reference (always a blank/broken image):
   - a markdown image points at a file that does not exist,
   - the file is actually an SVG (GitHub's CSP blanks inline <img> SVGs served from the repo),
   - the file is a 0-byte stub or a git-LFS pointer (served as text, not an image),
   - the file is not a real raster (PNG/JPEG/GIF/WEBP magic bytes).

2. STALE-CACHE after a content change (the "I updated it but GitHub still shows the old one" bug):
   GitHub proxies repo images through camo.githubusercontent.com and caches them by URL. If a
   chart's *content* changes but its *filename* stays the same, camo keeps serving the old bytes.
   The fix is to bump the versioned filename (foo_v4.png -> foo_v5.png) so the URL changes and camo
   refetches. This guard records each image's content hash in docs/plots/.image-manifest.json and
   FAILS if a referenced image's bytes changed without its filename changing — forcing the bump.

Usage:
  python plugin/validate/check_doc_images.py            # verify (CI / pre-commit)
  python plugin/validate/check_doc_images.py --update    # accept current state into the manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DOCS = os.path.join(REPO, "docs")
MANIFEST = os.path.join(DOCS, "plots", ".image-manifest.json")

# markdown ![alt](path)  and  <img src="path">
_MD_IMG = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?\s*(?:\"[^\"]*\")?\)")
_HTML_IMG = re.compile(r"<img\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)

_RASTER_MAGIC = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",  # RIFF....WEBP
}


def _iter_md_files():
    for root, _dirs, files in os.walk(DOCS):
        for fn in files:
            if fn.endswith(".md"):
                yield os.path.join(root, fn)


def _refs(md_path: str):
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    for m in _MD_IMG.finditer(text):
        yield m.group(1)
    for m in _HTML_IMG.finditer(text):
        yield m.group(1)


def _is_external(ref: str) -> bool:
    return ref.startswith(("http://", "https://", "data:", "//"))


def _classify(path: str) -> str:
    """Return '' if a valid raster, else a human reason the image won't render."""
    if not os.path.exists(path):
        return "missing file"
    size = os.path.getsize(path)
    if size == 0:
        return "0-byte file"
    with open(path, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        return "git-LFS pointer (not the actual image bytes)"
    stripped = head.lstrip()
    if stripped.startswith(b"<?xml") or stripped[:4].lower() == b"<svg":
        return "SVG (GitHub blanks inline <img> SVGs — rasterize to PNG)"
    for magic in _RASTER_MAGIC:
        if head.startswith(magic):
            return ""
    return "not a PNG/JPEG/GIF/WEBP raster"


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="accept the current referenced images into the manifest (run after a legit version bump)")
    args = ap.parse_args()

    referenced: dict[str, str] = {}   # repo-relative image path -> absolute path
    broken: list[str] = []

    for md in _iter_md_files():
        for ref in _refs(md):
            if _is_external(ref):
                continue
            ref_clean = ref.split("#", 1)[0].split("?", 1)[0]
            abs_path = os.path.normpath(os.path.join(os.path.dirname(md), ref_clean))
            rel = os.path.relpath(abs_path, REPO)
            reason = _classify(abs_path)
            if reason:
                broken.append(f"  {os.path.relpath(md, REPO)} -> {ref}  [{reason}]")
            else:
                referenced[rel] = abs_path

    # Stale-cache check: content changed but filename (hence GitHub URL) did not.
    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)
    stale: list[str] = []
    current = {}
    for rel, abs_path in sorted(referenced.items()):
        digest = _sha(abs_path)
        current[rel] = digest
        old = manifest.get(rel)
        if old is not None and old != digest:
            stale.append(f"  {rel}  (content changed, filename unchanged)")

    if args.update:
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
        with open(MANIFEST, "w") as f:
            json.dump(current, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"[check_doc_images] manifest updated: {len(current)} referenced images")
        if broken:
            print("WARNING: broken references still present:\n" + "\n".join(broken))
            return 1
        return 0

    ok = True
    if broken:
        ok = False
        print("BROKEN image references (will not render on GitHub):")
        print("\n".join(broken))
    if stale:
        ok = False
        print("\nSTALE-CACHE risk — image bytes changed but the filename did not, so GitHub's camo")
        print("proxy will keep serving the OLD image. Bump the versioned filename (e.g. _v4 -> _v5)")
        print("and update the markdown reference, then run: python plugin/validate/check_doc_images.py --update")
        print("\n".join(stale))
    if ok:
        print(f"[check_doc_images] OK — {len(referenced)} referenced images, all valid rasters, no stale-cache risk")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
