#!/usr/bin/env bash
# Regenerate docs/agentic-workflow-slide.png from docs/agentic-workflow-slide.html.
#
# Verified method on this GNR node (no browser needed): weasyprint renders the HTML
# to PDF honoring the HTML's own inline `@page{size:1300px <H>px}` rule, then
# ImageMagick rasterizes page 1 to PNG at 120 dpi.
#
# IMPORTANT: the inline `@page` HEIGHT in the HTML <style> must be tall enough to fit
# the whole slide on ONE page. If content grows and the render splits onto a 2nd page
# (or clips the footer), bump that height (e.g. 940 -> 1000) until it is a single page.
#
# Trigger rule: whenever the skills under skills/ change in a way that alters the slide,
# update docs/agentic-workflow-slide.html first, then run this script and commit both.
set -euo pipefail
cd "$(dirname "$0")/.."

# weasyprint lives in the sglang-cpu venv on this node.
source /scratch/bkaul/venvs/sglang-cpu/bin/activate 2>/dev/null || true

python -c "from weasyprint import HTML; HTML(filename='docs/agentic-workflow-slide.html').write_pdf('/tmp/slide.pdf')"
convert -density 120 -background white -alpha remove '/tmp/slide.pdf[0]' -quality 95 docs/agentic-workflow-slide.png
identify docs/agentic-workflow-slide.png
echo "Rendered docs/agentic-workflow-slide.png — review it, then: git add docs/agentic-workflow-slide.{html,png} && commit."
