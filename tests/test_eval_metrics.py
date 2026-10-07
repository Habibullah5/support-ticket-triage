import json
from pathlib import Path

from app.models import CATEGORIES, PRIORITIES
from evals.run_eval import CASES_PATH, compute_metrics, load_cases, percentile, update_readme


def test_fixed_case_set_is_valid():
    cases = load_cases(CASES_PATH)
    assert len(cases) == 30
    assert len({c["id"] for c in cases}) == 30
    assert all(c["category"] in CATEGORIES and c["priority"] in PRIORITIES and c["ticket"].strip() for c in cases)
    assert {c["category"] for c in cases} == set(CATEGORIES)


def rec(cat_ok, pri_ok, schema_ok=True, cat="billing", tag="clear"):
    return {"id": "x", "tag": tag, "expected_category": cat, "expected_priority": "low",
            "pred_category": cat if cat_ok else "other", "pred_priority": "low" if pri_ok else "high",
            "schema_ok": schema_ok, "first_attempt_valid": schema_ok, "repair_attempted": False,
            "failure_reason": None, "latency_ms": 10.0, "input_tokens": 5, "output_tokens": 2,
            "cost_usd": None, "category_ok": cat_ok, "priority_ok": pri_ok, "both_ok": cat_ok and pri_ok}


def test_metrics_math_and_breakdowns():
    m = compute_metrics([rec(True, True), rec(True, False), rec(False, True, cat="refund"),
                         rec(False, False, schema_ok=False, cat="refund", tag="adversarial")])
    assert m["n"] == 4 and m["schema_pass"]["count"] == 3
    assert m["category_accuracy"]["count"] == 2 and m["priority_accuracy"]["count"] == 2
    assert m["both_correct"]["count"] == 1 and len(m["misses"]) == 3
    assert m["by_category"]["refund"]["both"]["count"] == 0 and m["by_case_type"]["adversarial"]["n"] == 1
    assert m["cost_usd"] is None


def test_percentile_nearest_rank():
    assert percentile(list(range(1, 101)), 95) == 95 and percentile([], 95) == 0.0 and percentile([7], 95) == 7


def test_update_readme_replaces_only_marked_block(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- EVAL_RESULTS_START -->\nold\n<!-- EVAL_RESULTS_END -->\nafter\n", encoding="utf-8")
    assert update_readme("NEW\n", readme)
    text = readme.read_text(encoding="utf-8")
    assert "NEW" in text and "old" not in text and text.startswith("before") and text.endswith("after\n")
    assert json.dumps(text)  # still plain text
