import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class DocImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = self.root / "plugin/validate"
        self.tools.mkdir(parents=True)
        self.plots = self.root / "docs/plots"
        self.plots.mkdir(parents=True)
        for name in ("check_doc_images.py", "bump_chart.py"):
            shutil.copyfile(Path(__file__).with_name(name), self.tools / name)
        self.png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
        )
        self.names = ("dsv4_baseline_emr_gnr_v1.png", "dsv4_socket_walltime_v1.png")
        for name in self.names:
            (self.plots / name).write_bytes(self.png)
        self.readme = self.root / "README.md"
        self.page = self.root / "docs/e2e-wall-time.md"
        for path, prefix in ((self.readme, "docs/plots/"), (self.page, "plots/")):
            path.write_text(
                "<!-- dsv4-comparison:start -->\n"
                + "\n".join(f"![chart]({prefix}{name})" for name in self.names)
                + "\n<!-- dsv4-comparison:end -->\n"
            )
        self.manifest = self.plots / ".image-manifest.json"

    def run_tool(self, name="check_doc_images.py", *args):
        return subprocess.run(
            [sys.executable, str(self.tools / name), *args],
            cwd=self.root, capture_output=True, text=True,
        )

    def test_shared_charts_pass_and_readme_only_drift_fails(self):
        self.assertEqual(self.run_tool().returncode, 0)
        newer = self.names[0].replace("v1", "v2")
        (self.plots / newer).write_bytes(self.png)
        self.readme.write_text(self.readme.read_text().replace(self.names[0], newer))
        result = self.run_tool()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different comparison charts", result.stdout)

    def test_missing_duplicate_and_reversed_markers_fail(self):
        original = self.readme.read_text()
        start, end = "<!-- dsv4-comparison:start -->", "<!-- dsv4-comparison:end -->"
        for text in (original.replace(start, ""), start + original, end + original.replace(end, "")):
            with self.subTest(text=text):
                self.readme.write_text(text)
                self.assertNotEqual(self.run_tool().returncode, 0)

    def test_readme_image_validation_and_reject_without_manifest_write(self):
        self.assertEqual(self.run_tool("check_doc_images.py", "--update").returncode, 0)
        before = self.manifest.read_bytes()
        self.readme.write_text(self.readme.read_text() + "\n![extra](docs/plots/extra.svg)\n")
        (self.plots / "extra.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        result = self.run_tool("check_doc_images.py", "--update")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SVG", result.stdout)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_update_cannot_bypass_version_bump(self):
        self.assertEqual(self.run_tool("check_doc_images.py", "--update").returncode, 0)
        before = self.manifest.read_bytes()
        (self.plots / self.names[0]).write_bytes(self.png + b"changed")
        result = self.run_tool("check_doc_images.py", "--update")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STALE-CACHE", result.stdout)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_bump_updates_both_pages_and_propagates_failures(self):
        result = self.run_tool("bump_chart.py", str(self.plots / self.names[0]))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        newer = self.names[0].replace("v1", "v2")
        for path in (self.readme, self.page):
            self.assertIn(newer, path.read_text())
            self.assertNotIn(self.names[0], path.read_text())
        self.assertIn(f"docs/plots/{newer}", json.loads(self.manifest.read_text()))
        self.assertEqual(self.run_tool().returncode, 0)
        self.assertNotEqual(self.run_tool("bump_chart.py", str(self.plots / "missing.png")).returncode, 0)
        self.readme.write_text(self.readme.read_text() + "\n![broken](missing.png)\n")
        result = self.run_tool("bump_chart.py", str(self.plots / self.names[1]))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing file", result.stdout)


if __name__ == "__main__":
    unittest.main()
