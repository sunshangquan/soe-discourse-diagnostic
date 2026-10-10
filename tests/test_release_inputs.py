import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ReleaseInputsTest(unittest.TestCase):
    def test_average_precision_groups_tied_scores(self):
        from pr_auc_eval import pr_auc

        expected = (1 / 2) * (1 / 2) + (2 / 3) * (1 / 2)
        self.assertAlmostEqual(expected, pr_auc([0.5, 0.5, 0.1], [1, 0, 1]))
        self.assertAlmostEqual(expected, pr_auc([0.5, 0.5, 0.1], [0, 1, 1]))

    def test_hedges_g_uses_sample_size_weighted_pooled_sd(self):
        from permutation_test import hedges_g
        from stats_addon import cosine_distance, doc_vectors, find_pairs, load_all

        segments, methods, _, _ = load_all()
        vectors = doc_vectors(methods["llm"], segments)
        lc_pairs, sl_pairs = find_pairs(segments)
        lc = [cosine_distance(vectors[a], vectors[b]) for a, b in lc_pairs]
        sl = [cosine_distance(vectors[a], vectors[b]) for a, b in sl_pairs]
        self.assertAlmostEqual(0.846464, hedges_g(lc, sl), places=5)

    def test_ngram_mining_input_is_packaged(self):
        path = ROOT / "phase0" / "shared_ngrams_top200.tsv"
        self.assertTrue(path.is_file())
        with path.open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
        self.assertEqual(53, len(rows))
        self.assertTrue(all(int(row["n"]) >= 5 and int(row["doc_freq"]) >= 8 for row in rows))

    def test_default_gold_is_three_annotator_majority(self):
        from gold_labels import load_majority_l1

        rows = load_majority_l1()
        self.assertEqual(86, len(rows))
        self.assertEqual(86, len({row["seg_id"] for row in rows}))
        binary = [row for row in rows if row["l1"] in ("slogan", "substantive")]
        self.assertEqual(83, len(binary))
        self.assertEqual(49, sum(row["l1"] == "substantive" for row in binary))

        from stats_addon import load_all
        from evaluate import load_gold_rows

        self.assertEqual(86, len(load_all()[3]))
        self.assertEqual(rows, load_gold_rows())

    def test_fresh_copy_reproduces_calibration_and_majority_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / "release"
            fresh.mkdir()
            for name in ("src", "data", "results", "phase0"):
                shutil.copytree(ROOT / name, fresh / name)
            for script in (
                "calibrate.py",
                "evaluate.py",
                "pr_auc_eval.py",
                "stats_addon.py",
                "permutation_test.py",
            ):
                subprocess.run(
                    [sys.executable, str(fresh / "src" / script)],
                    cwd=fresh,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            for name in (
                "slogan_density_ngram.csv",
                "scores_calibrated.csv",
                "eval_summary.md",
                "pr_auc.csv",
                "stats_addon.json",
                "permutation_test.csv",
            ):
                self.assertEqual(
                    (ROOT / "results" / name).read_bytes(),
                    (fresh / "results" / name).read_bytes(),
                    name,
                )
            summary = (fresh / "results" / "eval_summary.md").read_text(encoding="utf-8")
            self.assertIn("| llm | 0.796 |", summary)
            with (fresh / "results" / "pr_auc.csv").open(encoding="utf-8") as source:
                auc = {row["method"]: row for row in csv.DictReader(source)}
            self.assertEqual("0.412", auc["dict"]["ROC_AUC"])
            self.assertEqual("83", auc["dict"]["n"])
            self.assertEqual("0.748", auc["calibrated"]["PR_AUC"])


if __name__ == "__main__":
    unittest.main()
