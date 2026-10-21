#!/usr/bin/env python3
"""Bump a doc chart to a fresh versioned filename so GitHub's camo image proxy refetches it.

GitHub caches repo images by URL; a content change under the SAME filename keeps serving the OLD
image. The fix is to rename foo_vN.png -> foo_v(N+1).png and update every markdown reference, so the
URL changes and the image renders fresh. This does that atomically (rename + update all docs refs),
then refreshes the render-guard manifest.

Use it whenever a chart's CONTENT changed, or to force a stale chart to re-render:
  python plugin/validate/bump_chart.py docs/plots/e2e_decode_progression_v4.png
  python plugin/validate/bump_chart.py --all        # bump every chart referenced in docs/
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DOCS = os.path.join(REPO, "docs")
_VER = re.compile(r"^(?P<stem>.+)_v(?P<n>\d+)(?P<ext>\.[A-Za-z0-9]+)$")


def _next_name(fname: str) -> str:
    m = _VER.match(fname)
    if m:
        return f"{m.group('stem')}_v{int(m.group('n')) + 1}{m.group('ext')}"
    stem, ext = os.path.splitext(fname)
    return f"{stem}_v2{ext}"  # first bump of an unversioned file -> _v2


def _referenced_images() -> set[str]:
    pat = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?\s*(?:\"[^\"]*\")?\)|<img\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
    out: set[str] = set()
    for root, _d, files in os.walk(DOCS):
        for fn in files:
            if not fn.endswith(".md"):
                continue
            md = os.path.join(root, fn)
            for m in pat.finditer(open(md, encoding="utf-8").read()):
                ref = m.group(1) or m.group(2)
                if ref and not ref.startswith(("http://", "https://", "data:", "//")):
                    out.add(os.path.normpath(os.path.join(os.path.dirname(md), ref.split("#")[0].split("?")[0])))
    return {p for p in out if p.lower().endswith(".png")}


def _bump_one(abs_png: str) -> bool:
    d, fname = os.path.split(abs_png)
    new_fname = _next_name(fname)
    new_abs = os.path.join(d, new_fname)
    if not os.path.exists(abs_png):
        print(f"  SKIP (missing): {os.path.relpath(abs_png, REPO)}")
        return False
    # git mv to preserve history; fall back to plain rename if not tracked
    r = subprocess.run(["git", "-C", REPO, "mv", abs_path_rel(abs_png), abs_path_rel(new_abs)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        os.rename(abs_png, new_abs)
    # update every markdown reference old_fname -> new_fname
    n_refs = 0
    for root, _d, files in os.walk(DOCS):
        for fn in files:
            if not fn.endswith(".md"):
                continue
            md = os.path.join(root, fn)
            txt = open(md, encoding="utf-8").read()
            if fname in txt:
                open(md, "w", encoding="utf-8").write(txt.replace(fname, new_fname))
                n_refs += 1
    print(f"  {fname} -> {new_fname}  ({n_refs} md file(s) updated)")
    return True


def abs_path_rel(p: str) -> str:
    return os.path.relpath(p, REPO)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pngs", nargs="*", help="chart PNG path(s) to bump")
    ap.add_argument("--all", action="store_true", help="bump every PNG referenced in docs/")
    a = ap.parse_args()
    targets = sorted(_referenced_images()) if a.all else [os.path.abspath(p) for p in a.pngs]
    if not targets:
        print("nothing to bump (pass PNG paths or --all)")
        return 1
    print(f"Bumping {len(targets)} chart(s):")
    results = [_bump_one(p) for p in targets]
    if any(results):
        subprocess.run([sys.executable, os.path.join(HERE, "check_doc_images.py"), "--update"], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
