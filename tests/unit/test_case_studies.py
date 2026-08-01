from __future__ import annotations

from agente_5g.evaluation.case_studies import error_analysis_summary, select_case_studies

RECORDS = [
    # (identifier, attack_type, true_label, predicted_label, risk_score, explanation)
    ("s1", "SYNflood", True, True, 0.9, "high risk syn flood"),
    ("s2", "SYNflood", True, True, 0.6, "medium risk syn flood"),
    ("s3", "SYNflood", True, False, 0.3, "missed, risk 0.3"),
    ("s4", "SYNflood", True, False, 0.1, "missed badly, risk 0.1"),
    ("s5", "SYNflood", False, True, 0.7, "false alarm, risk 0.7"),
    ("s6", "SYNflood", False, False, 0.05, "correctly benign"),
    ("s7", "UDPflood", False, False, 0.02, "correctly benign udp"),
]


def test_select_case_studies_picks_highest_risk_tp():
    result = select_case_studies(RECORDS)
    tp = result["SYNflood"]["TP"]
    assert tp is not None
    assert tp.identifier == "s1"
    assert tp.risk_score == 0.9


def test_select_case_studies_picks_lowest_risk_fn():
    result = select_case_studies(RECORDS)
    fn = result["SYNflood"]["FN"]
    assert fn is not None
    assert fn.identifier == "s4"  # the most obviously missed case


def test_select_case_studies_picks_highest_risk_fp():
    result = select_case_studies(RECORDS)
    fp = result["SYNflood"]["FP"]
    assert fp is not None
    assert fp.identifier == "s5"


def test_select_case_studies_none_when_outcome_absent():
    result = select_case_studies(RECORDS)
    # UDPflood only has a TN record -- no TP/FN/FP exists
    assert result["UDPflood"]["TP"] is None
    assert result["UDPflood"]["FN"] is None
    assert result["UDPflood"]["FP"] is None


def test_select_case_studies_respects_requested_outcomes_only():
    result = select_case_studies(RECORDS, outcomes=("TP",))
    assert set(result["SYNflood"].keys()) == {"TP"}


def test_error_analysis_summary_counts_and_rates():
    rows = error_analysis_summary(RECORDS)
    syn = next(r for r in rows if r["attack_type"] == "SYNflood")

    assert syn["tp"] == 2
    assert syn["fp"] == 1
    assert syn["fn"] == 2
    assert syn["tn"] == 1
    assert syn["recall"] == 2 / 4
    assert syn["fpr"] == 1 / 2


def test_error_analysis_summary_handles_no_positive_class():
    rows = error_analysis_summary(RECORDS)
    udp = next(r for r in rows if r["attack_type"] == "UDPflood")

    assert udp["tp"] == 0
    assert udp["recall"] is None  # 0/0 undefined, not silently 0.0
