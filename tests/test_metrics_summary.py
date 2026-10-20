"""Verify summary aggregation without loading models."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from assess.benchmark import save_metrics_summary


def scored_result(accuracy, forgetting=None, json_error=False):
    return {
        "json_error": json_error,
        "scores": {
            "accuracy": accuracy,
            "jaccard": accuracy,
            "hamming": accuracy,
            "accurate_retrieval": accuracy,
            "selective_forgetting": forgetting,
        },
    }


class MetricsSummaryTests(unittest.TestCase):
    def test_partial_periods_and_inapplicable_metrics(self):
        results = [
            [scored_result(1), scored_result(0, json_error=True)],
            [scored_result(1, 0.5), None],
            [None, None],
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "overall_metrics_summary_strict.json"
            save_metrics_summary(results, path)
            summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["accuracy"], {
            "mean": 2 / 3, "applicable_count": 3, "per_period_mean": [0.5, 1.0, None],
        })
        self.assertEqual(summary["selective_forgetting"], {
            "mean": 0.5, "applicable_count": 1, "per_period_mean": [None, 0.5, None],
        })
        self.assertEqual(summary["protocol"], {
            "question_count": 3, "parsed_count": 2, "format_error_count": 1,
            "successful_parse_rate": 2 / 3, "overall_accuracy": 2 / 3, "parsed_accuracy": 1.0,
        })

    def test_no_completed_results(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            save_metrics_summary([[None]], path)
            summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["accuracy"]["applicable_count"], 0)
        self.assertIsNone(summary["protocol"]["successful_parse_rate"])


if __name__ == "__main__":
    unittest.main()
