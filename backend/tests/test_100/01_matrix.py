"""
01_matrix.py — Build the 100-case test matrix and pre-calculate all expected benchmarks.

Design: 10 groups × 10 cases = 100 cases
  - Groups 1-5  → Month 1 (January  2025-01-15)
  - Groups 6-10 → Month 5 (May      2025-05-15)

Each group follows the same 10-case triangular pattern from the test spec.

Outputs:
  data/test_matrix.json        — 100 case definitions ready for insertion
  data/expected_benchmarks.json — pre-calculated PASS/FAIL targets
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

INTENT_COMPLAINT = 1   # Negative / Improvement Opportunity
INTENT_PRAISE    = 2   # Positive / Notice
RISK_ORDINARY    = 1
RISK_RED_FLAG    = 2
RISK_NEVER_EVENT = 3


def load_config():
    with open(os.path.join(DATA_DIR, 'config.json')) as f:
        return json.load(f)


def build_group(group_num, cfg, date_str):
    """
    Return the 10 cases for one group.

    10-case triangular pattern (from test spec):
    ─────────────────────────────────────────────────────────────────
    #  Patient  Intent     Sections       Doctors    Workers    Risk
    1    P1     Complaint  [S1]           [D1]       [W1]       NeverEvent
    2    P2     Praise     [S1,S2]        [D1,D2]    [W1,W2]    RedFlag
    3    P2     Complaint  [S1,S2,S3]     [D1-D3]    [W1-W3]    NeverEvent
    4    P3     Praise     [S1,S2,S3,S4]  [D1-D4]    [W1-W4]    Ordinary
    5    P3     Complaint  [S1..S5]       [D1-D5]    [W1-W5]    Ordinary
    6    P4     Praise     [S8]           [D5]       [W5]        Ordinary
    7    P4     Complaint  [S7,S8]        [D4,D5]    [W4,W5]    Ordinary
    8    P5     Praise     [S6,S7,S8]     [D3-D5]    [W3-W5]    NeverEvent
    9    P5     Complaint  [S5,S6,S7,S8]  [D2-D5]    [W2-W5]    RedFlag
    10   P5     Praise     [S4..S8]       [D1-D5]    [W1-W5]    RedFlag
    ─────────────────────────────────────────────────────────────────
    """
    D = cfg["doctor_ids"]   # [101,102,103,104,105]
    W = cfg["worker_ids"]   # [6,7,8,9,10]
    S = cfg["section_ids"]  # [43,95,93,60,72,98,309,83]  (S1..S8)
    P = [f"{cfg['patient_prefix']}{i}" for i in range(1, 6)]  # T100_P1..T100_P5

    template = [
        # (patient_idx, intent, section_slice, doctor_slice, worker_slice, risk)
        (0, INTENT_COMPLAINT, S[0:1],   D[0:1],   W[0:1],   RISK_NEVER_EVENT),
        (1, INTENT_PRAISE,    S[0:2],   D[0:2],   W[0:2],   RISK_RED_FLAG),
        (1, INTENT_COMPLAINT, S[0:3],   D[0:3],   W[0:3],   RISK_NEVER_EVENT),
        (2, INTENT_PRAISE,    S[0:4],   D[0:4],   W[0:4],   RISK_ORDINARY),
        (2, INTENT_COMPLAINT, S[0:5],   D[0:5],   W[0:5],   RISK_ORDINARY),
        (3, INTENT_PRAISE,    S[7:8],   D[4:5],   W[4:5],   RISK_ORDINARY),
        (3, INTENT_COMPLAINT, S[6:8],   D[3:5],   W[3:5],   RISK_ORDINARY),
        (4, INTENT_PRAISE,    S[5:8],   D[2:5],   W[2:5],   RISK_NEVER_EVENT),
        (4, INTENT_COMPLAINT, S[4:8],   D[1:5],   W[1:5],   RISK_RED_FLAG),
        (4, INTENT_PRAISE,    S[3:8],   D[0:5],   W[0:5],   RISK_RED_FLAG),
    ]

    cases = []
    for i, (pat_idx, intent, secs, docs, workers, risk) in enumerate(template):
        cases.append({
            "group":              group_num,
            "case_in_group":      i + 1,
            "patient_name":       P[pat_idx],
            "feedback_intent_type_id": intent,
            "target_section_ids": secs,
            "doctor_ids":         docs,
            "worker_ids":         workers,
            "clinical_risk_type_id": risk,
            "feedback_received_date": date_str,
            # Fixed lookup IDs (same for all cases)
            "issuing_department_id": cfg["issuing_dept_id"],
            "domain_id":            cfg["domain_id"],
            "category_id":          cfg["category_id"],
            "subcategory_id":       cfg["subcategory_id"],
            "classification_id":    cfg["classification_id"],
            "severity_id":          cfg["severity_id"],
            "stage_id":             cfg["stage_id"],
            "harm_id":              cfg["harm_id"],
            "source_id":            cfg["source_id"],
            "building_id":          cfg["building_id"],
        })
    return cases


def calculate_benchmarks(matrix, cfg):
    """Pre-calculate every expected metric for the verify script."""
    D = cfg["doctor_ids"]
    W = cfg["worker_ids"]
    S = cfg["section_ids"]
    month1 = cfg["month1_date"][:7]   # "2025-01"
    month5 = cfg["month5_date"][:7]   # "2025-05"

    # ── Patient profiles ──────────────────────────────────────────
    patient_counts = {}
    for case in matrix:
        p = case["patient_name"]
        patient_counts[p] = patient_counts.get(p, 0) + 1

    # ── Doctor monthly complaint/praise ───────────────────────────
    doctor_monthly = {did: {month1: {"C": 0, "P": 0}, month5: {"C": 0, "P": 0}} for did in D}
    for case in matrix:
        ym = case["feedback_received_date"][:7]
        intent = case["feedback_intent_type_id"]
        key = "C" if intent == INTENT_COMPLAINT else "P"
        for did in case["doctor_ids"]:
            doctor_monthly[did][ym][key] += 1

    doctor_totals = {}
    for did in D:
        doctor_totals[did] = {
            "C": doctor_monthly[did][month1]["C"] + doctor_monthly[did][month5]["C"],
            "P": doctor_monthly[did][month1]["P"] + doctor_monthly[did][month5]["P"],
        }

    # ── Worker monthly complaint/praise ───────────────────────────
    worker_monthly = {wid: {month1: {"C": 0, "P": 0}, month5: {"C": 0, "P": 0}} for wid in W}
    for case in matrix:
        ym = case["feedback_received_date"][:7]
        intent = case["feedback_intent_type_id"]
        key = "C" if intent == INTENT_COMPLAINT else "P"
        for wid in case["worker_ids"]:
            worker_monthly[wid][ym][key] += 1

    worker_totals = {}
    for wid in W:
        worker_totals[wid] = {
            "C": worker_monthly[wid][month1]["C"] + worker_monthly[wid][month5]["C"],
            "P": worker_monthly[wid][month1]["P"] + worker_monthly[wid][month5]["P"],
        }

    # ── Section subcases per month and total ─────────────────────
    section_month1 = {sid: 0 for sid in S}
    section_month5 = {sid: 0 for sid in S}
    section_total  = {sid: 0 for sid in S}
    total_subcases_m1 = 0
    total_subcases_m5 = 0

    for case in matrix:
        ym = case["feedback_received_date"][:7]
        for sid in case["target_section_ids"]:
            section_total[sid] = section_total.get(sid, 0) + 1
            if ym == month1:
                section_month1[sid] = section_month1.get(sid, 0) + 1
                total_subcases_m1 += 1
            else:
                section_month5[sid] = section_month5.get(sid, 0) + 1
                total_subcases_m5 += 1

    # ── Red flags & Never events ──────────────────────────────────
    rf_m1 = sum(1 for c in matrix if c["clinical_risk_type_id"] == RISK_RED_FLAG    and c["feedback_received_date"][:7] == month1)
    ne_m1 = sum(1 for c in matrix if c["clinical_risk_type_id"] == RISK_NEVER_EVENT and c["feedback_received_date"][:7] == month1)
    rf_m5 = sum(1 for c in matrix if c["clinical_risk_type_id"] == RISK_RED_FLAG    and c["feedback_received_date"][:7] == month5)
    ne_m5 = sum(1 for c in matrix if c["clinical_risk_type_id"] == RISK_NEVER_EVENT and c["feedback_received_date"][:7] == month5)

    return {
        "total_cases": len(matrix),

        "patient_counts": patient_counts,

        "doctor_month1":  {str(k): v for k, v in doctor_monthly.items()},
        "doctor_month5":  {str(k): {month5: doctor_monthly[k][month5]} for k in D},
        "doctor_totals":  {str(k): v for k, v in doctor_totals.items()},

        "worker_month1":  {str(k): v for k, v in worker_monthly.items()},
        "worker_month5":  {str(k): {month5: worker_monthly[k][month5]} for k in W},
        "worker_totals":  {str(k): v for k, v in worker_totals.items()},

        "section_subcases_month1": {str(k): v for k, v in section_month1.items()},
        "section_subcases_month5": {str(k): v for k, v in section_month5.items()},
        "section_subcases_total":  {str(k): v for k, v in section_total.items()},
        "total_subcases_month1":   total_subcases_m1,
        "total_subcases_month5":   total_subcases_m5,
        "total_subcases":          total_subcases_m1 + total_subcases_m5,

        "red_flags_month1":    rf_m1,
        "never_events_month1": ne_m1,
        "red_flags_month5":    rf_m5,
        "never_events_month5": ne_m5,
        "red_flags_total":     rf_m1 + rf_m5,
        "never_events_total":  ne_m1 + ne_m5,

        "month1": month1,
        "month5": month5,
    }


def main():
    print("=" * 60)
    print("01_MATRIX — Building 100-Case Test Matrix")
    print("=" * 60)

    cfg = load_config()
    if not cfg.get("all_checks_passed"):
        print("[ERROR] config.json shows failed prerequisites. Run 00_discover.py first.")
        sys.exit(1)

    matrix = []
    for g in range(1, 11):
        date_str = cfg["month1_date"] if g <= 5 else cfg["month5_date"]
        matrix.extend(build_group(g, cfg, date_str))

    print(f"\nGenerated {len(matrix)} cases across 10 groups")

    # Print group distribution summary
    m1_count = sum(1 for c in matrix if c["feedback_received_date"] == cfg["month1_date"])
    m5_count = sum(1 for c in matrix if c["feedback_received_date"] == cfg["month5_date"])
    print(f"  Month 1 ({cfg['month1_date']}): {m1_count} cases")
    print(f"  Month 5 ({cfg['month5_date']}): {m5_count} cases")

    # Calculate benchmarks
    bm = calculate_benchmarks(matrix, cfg)

    # Print benchmark summary
    print("\n[Expected Benchmarks]")
    print(f"  Total cases:              {bm['total_cases']}")
    print(f"  Total subcases:           {bm['total_subcases']} (M1={bm['total_subcases_month1']}, M5={bm['total_subcases_month5']})")
    print(f"  Red flags total:          {bm['red_flags_total']} (M1={bm['red_flags_month1']}, M5={bm['red_flags_month5']})")
    print(f"  Never events total:       {bm['never_events_total']} (M1={bm['never_events_month1']}, M5={bm['never_events_month5']})")

    print("\n  Patient profiles:")
    for p, cnt in sorted(bm['patient_counts'].items()):
        print(f"    {p}: {cnt} cases")

    print("\n  Doctor totals (C=Complaint, P=Praise):")
    for did, cp in bm['doctor_totals'].items():
        print(f"    Doctor {did}: {cp['C']}C + {cp['P']}P = {cp['C']+cp['P']} total")

    print("\n  Section subcases (total):")
    cfg_sections = cfg["section_ids"]
    for i, sid in enumerate(cfg_sections):
        total = bm['section_subcases_total'].get(str(sid), 0)
        print(f"    S{i+1} (ID={sid}): {total}")

    # Save outputs
    matrix_path = os.path.join(DATA_DIR, 'test_matrix.json')
    bm_path     = os.path.join(DATA_DIR, 'expected_benchmarks.json')

    with open(matrix_path, 'w', encoding='utf-8') as f:
        json.dump(matrix, f, indent=2)
    with open(bm_path, 'w', encoding='utf-8') as f:
        json.dump(bm, f, indent=2)

    print(f"\nSaved -> {matrix_path}")
    print(f"Saved -> {bm_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
