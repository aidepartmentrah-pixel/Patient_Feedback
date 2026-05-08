"""
02_insert.py — Insert all 100 test cases into the database.

Reads data/test_matrix.json, calls create_record() for each case,
tracks all inserted IDs.

Outputs: data/inserted_ids.json
"""

import sys, os, json, time

# Add both parent dirs: one for `backend.*` style, one for direct `api.*` imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, '..', '..')          # …/backend/
_REPO    = os.path.join(_BACKEND, '..')             # …/Patient_Feedback/
sys.path.insert(0, os.path.abspath(_BACKEND))
sys.path.insert(0, os.path.abspath(_REPO))

from backend.api.services.insert_service import create_record

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def build_payload(case_def):
    """Convert a case definition from the matrix into an insert_service payload."""
    doctors   = [{"doctor_id": did, "doctor_name": f"Dr. Test Doctor {i+1}"}
                 for i, did in enumerate(case_def["doctor_ids"])]
    employees = [{"employee_id": wid, "full_name": f"Worker Number {i+1}"}
                 for i, wid in enumerate(case_def["worker_ids"])]

    intent    = case_def["feedback_intent_type_id"]
    risk      = case_def["clinical_risk_type_id"]

    if intent == 1:   # Complaint
        complaint_text = (
            f"T100 Test Complaint | Group {case_def['group']} Case {case_def['case_in_group']} | "
            f"Patient {case_def['patient_name']} | Risk={'RedFlag' if risk==2 else 'NeverEvent' if risk==3 else 'Ordinary'}"
        )
    else:              # Praise
        complaint_text = (
            f"T100 Test Praise | Group {case_def['group']} Case {case_def['case_in_group']} | "
            f"Patient {case_def['patient_name']} | Risk={'RedFlag' if risk==2 else 'NeverEvent' if risk==3 else 'Ordinary'}"
        )

    return {
        "complaint_text":           complaint_text,
        "feedback_received_date":   case_def["feedback_received_date"],
        "issuing_department_id":    case_def["issuing_department_id"],
        "domain_id":                case_def["domain_id"],
        "category_id":              case_def["category_id"],
        "subcategory_id":           case_def["subcategory_id"],
        "classification_id":        case_def["classification_id"],
        "severity_id":              case_def["severity_id"],
        "stage_id":                 case_def["stage_id"],
        "harm_id":                  case_def["harm_id"],
        "clinical_risk_type_id":    risk,
        "feedback_intent_type_id":  intent,
        "requires_explanation":     False,
        "immediate_action":         "T100 test — immediate action recorded",
        "taken_action":             "T100 test — follow-up action recorded",
        "patient_name":             case_def["patient_name"],
        "is_inpatient":             True,
        "source_id":                case_def["source_id"],
        "building_id":              case_def["building_id"],
        "target_department_ids":    case_def["target_section_ids"],
        "doctors":                  doctors,
        "employees":                employees,
    }


def main():
    print("=" * 60)
    print("02_INSERT — Inserting 100 Test Cases")
    print("=" * 60)

    matrix = load_json('test_matrix.json')
    print(f"Loaded {len(matrix)} cases from test_matrix.json\n")

    inserted = []
    failed   = []
    t_start  = time.time()

    for idx, case_def in enumerate(matrix, 1):
        payload = build_payload(case_def)

        result = create_record(payload, save_mode='workflow')

        if result.get("success"):
            case_id = result["id"]
            inserted.append({
                "db_id":          case_id,
                "patient_name":   case_def["patient_name"],
                "group":          case_def["group"],
                "case_in_group":  case_def["case_in_group"],
                "date":           case_def["feedback_received_date"],
                "intent":         case_def["feedback_intent_type_id"],
                "risk":           case_def["clinical_risk_type_id"],
                "sections":       case_def["target_section_ids"],
                "doctors":        case_def["doctor_ids"],
                "workers":        case_def["worker_ids"],
            })
            status = "OK"
        else:
            failed.append({"index": idx, "error": result.get("message"), "case": case_def})
            status = f"FAIL: {result.get('message','?')}"

        elapsed = time.time() - t_start
        print(f"  [{idx:>3}/100] G{case_def['group']}:C{case_def['case_in_group']} "
              f"{'Complaint' if case_def['feedback_intent_type_id']==1 else 'Praise   '} "
              f"{'RF' if case_def['clinical_risk_type_id']==2 else 'NE' if case_def['clinical_risk_type_id']==3 else '  '} "
              f"-> {status}")

    total_time = time.time() - t_start

    # Save results
    ids_path = os.path.join(DATA_DIR, 'inserted_ids.json')
    with open(ids_path, 'w', encoding='utf-8') as f:
        json.dump({"inserted": inserted, "failed": failed}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE in {total_time:.1f}s")
    print(f"  Inserted: {len(inserted)}")
    print(f"  Failed:   {len(failed)}")
    if failed:
        print("\n  Failed cases:")
        for f_ in failed:
            print(f"    Case {f_['index']}: {f_['error']}")
    print(f"\nSaved -> {ids_path}")
    print("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
