import os
import re
from datetime import datetime

# ----------------------------
# CONFIG
# ----------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_EXCLUDE_DIRS = {
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    ".idea",
    ".vscode",
    "model_training",
    "models_directory",
    "tools",
    "GETTING_Schema",
    "data_exploration",
}

LABEL_MAP = {
    "backend/api/db_layer": "DB Layer",
    "backend/api/services": "Services",
    "backend/api/routers": "Routers",
    "backend/core": "Core",
}

# ----------------------------
# PARSER
# ----------------------------

def extract_functions(filepath):
    functions = []
    routes = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    for m in re.finditer(r"def\s+([a-zA-Z0-9_]+)\s*\(", content):
        functions.append(m.group(1))

    for m in re.finditer(r"@router\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]", content):
        routes.append((m.group(1).upper(), m.group(2)))

    return functions, routes

# ----------------------------
# CORE SCANNER ENGINE
# ----------------------------

def scan_tree(root_dir, include_only=None, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = set()

    results = []

    for root, dirs, files in os.walk(root_dir):
        # Exclude folders
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # Focus mode: only inside a subtree
        rel_root = os.path.relpath(root, root_dir)

        # Focus mode: only inside a subtree
        if include_only:
            # Normalize both to forward slashes for comparison
            rel_root_norm = rel_root.replace("\\", "/")
            include_norm = include_only.replace("\\", "/")

            if not rel_root_norm.startswith(include_norm):
                continue

        for file in files:
            if not file.endswith(".py"):
                continue

            path = os.path.join(root, file)
            rel = os.path.relpath(path, root_dir)

            functions, routes = extract_functions(path)
            results.append((rel, functions, routes))

    return results

# ----------------------------
# CLASSIFIER
# ----------------------------

def classify(rel_path):
    for key, label in LABEL_MAP.items():
        if rel_path.replace("\\", "/").startswith(key):
            return label
    return "Other"

# ----------------------------
# WRITER
# ----------------------------

def write_map(title, entries, output_file):
    sections = {}

    for rel, functions, routes in entries:
        bucket = classify(rel)
        sections.setdefault(bucket, []).append((rel, functions, routes))

    with open(output_file, "w", encoding="utf-8") as out:
        out.write(f"# 🗺️ {title}\n\n")
        out.write(f"_Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n")

        for section, files in sections.items():
            out.write(f"\n---\n## 📦 {section}\n\n")

            for rel, functions, routes in sorted(files):
                out.write(f"### 📄 {rel}\n")

                if routes:
                    out.write("**Routes:**\n")
                    for method, path in routes:
                        out.write(f"- `{method} {path}`\n")

                if functions:
                    out.write("**Functions:**\n")
                    for fn in functions:
                        out.write(f"- `{fn}()`\n")

                if not functions and not routes:
                    out.write("_No functions found_\n")

                out.write("\n")

    print("✅ Generated:", output_file)

# ----------------------------
# TOOL A: EXCLUDE MODE
# ----------------------------

def generate_map_excluding(output_name="_architecture_excluding.txt"):
    entries = scan_tree(
        BACKEND_ROOT,
        include_only=None,
        exclude_dirs=DEFAULT_EXCLUDE_DIRS
    )

    output_file = os.path.join(SCRIPT_DIR, output_name)
    write_map("Backend Architecture (Exclude Mode)", entries, output_file)

# ----------------------------
# TOOL B: FOCUS MODE
# ----------------------------

def generate_map_for_subtree(subfolder, label_name, output_name):
    entries = scan_tree(
        BACKEND_ROOT,
        include_only=subfolder,
        exclude_dirs=DEFAULT_EXCLUDE_DIRS
    )

    output_file = os.path.join(SCRIPT_DIR, output_name)
    write_map(f"{label_name} — Subtree Map", entries, output_file)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_mermaid_atlas(output_dir_name: str = "architecture_atlas") -> None:
    """
    Writes a set of Mermaid (.mmd) files that form an 'architecture atlas'.
    Output folder is created next to this script (tools/architecture_atlas/ by default).
    """
    atlas_dir = os.path.join(SCRIPT_DIR, output_dir_name)
    _ensure_dir(atlas_dir)

    # Naming convention: NN_topic_name.mmd
    diagrams = {
        "00_system_overview.mmd": r"""flowchart LR
    UI[Frontend React]
    API[FastAPI Backend]
    DB[(SQL Server)]
    ML[(ML SQLite)]

    UI --> API
    API --> DB
    API --> ML
""",
        "01_layered_architecture.mmd": r"""flowchart TB
    Routers --> Services
    Services --> DB_Layer
    Services --> ML_Layer
    Services --> Reporting
""",
        "02_api_surface.mmd": r"""flowchart LR
    InsertRouter
    TableViewRouter
    ReportsRouter
    FollowUpRouter
    SettingsRouter
    TrainingRouter
    TrendRouter
    DashboardRouter
    PatientsRouter
    DoctorsRouter
    ClassificationRouter
    NERRouter
    RedFlagsRouter
    NeverEventsRouter
    SeasonalReportsRouter

    InsertRouter --> Services
    TableViewRouter --> Services
    ReportsRouter --> Services
    FollowUpRouter --> Services
    SettingsRouter --> Services
    TrainingRouter --> Services
    TrendRouter --> Services
    DashboardRouter --> Services
    PatientsRouter --> Services
    DoctorsRouter --> Services
    ClassificationRouter --> Services
    NERRouter --> Services
    RedFlagsRouter --> Services
    NeverEventsRouter --> Services
    SeasonalReportsRouter --> Services
""",
        "10_insert_flow.mmd": r"""flowchart LR
    InsertRouter --> InsertService
    InsertService --> IncidentCaseDB
    InsertService --> IncidentCaseDoctorDB
    InsertService --> IncidentTargetDepartmentDB
    InsertService --> IncidentFeedbackDB
    InsertService --> ML_Insert
""",
        "11_table_view_flow.mmd": r"""flowchart LR
    TableViewRouter --> TableViewService
    TableViewService --> ReportsDB
    TableViewService --> IncidentCaseDB
    TableViewService --> ExcelExport
""",
        "12_followup_flow.mmd": r"""flowchart LR
    FollowUpRouter --> FollowUpService
    FollowUpService --> FollowUpDB
    FollowUpService --> ActionItemsDB
""",
        "13_seasonal_report_flow.mmd": r"""flowchart LR
    SeasonalReportsRouter --> SeasonalReportGenerator
    SeasonalReportGenerator --> SeasonalReportDB
    SeasonalReportGenerator --> SeasonalAggregationDB
    SeasonalReportGenerator --> OrgPolicyDB
""",
        "14_training_flow.mmd": r"""flowchart LR
    TrainingRouter --> TrainingService
    TrainingService --> TrainingDB
    TrainingService --> ML_Trainer
    ML_Trainer --> ML_DB
""",
        "20_domain_incident.mmd": r"""flowchart LR
    IncidentCaseDB --> IncidentDoctorDB
    IncidentCaseDB --> IncidentTargetDepartmentDB
    IncidentCaseDB --> IncidentFeedbackDB
    IncidentCaseDB --> FollowUpDB
    IncidentCaseDB --> ActionItemsDB
""",
        "21_domain_patients.mmd": r"""flowchart LR
    PatientsRouter --> PatientsService --> PatientsDB
    PatientsService --> IncidentCaseDB
""",
        "22_domain_doctors.mmd": r"""flowchart LR
    DoctorsRouter --> DoctorsService --> DoctorsDB
    DoctorsService --> IncidentCaseDB
""",
        "23_domain_reports.mmd": r"""flowchart LR
    ReportsRouter --> ReportsService --> ReportsDB
    ReportsService --> SeasonalAggregationDB
""",
        "24_domain_settings.mmd": r"""flowchart LR
    SettingsRouter --> SettingsService --> SettingsDB
    SettingsService --> OrgPolicyDB
""",
        "30_ml_pipeline.mmd": r"""flowchart LR
    InsertService --> PredictionService
    PredictionService --> EmbeddingModel
    PredictionService --> Classifiers
    PredictionService --> ML_DB
""",
        # Optional "index" file for easy navigation
        "README.txt": """Architecture Atlas (Mermaid)

How to use:
1) Open https://mermaid.live/
2) Paste any .mmd content
3) Export PNG/SVG and save it

File naming:
00-09  Overview
10-19  Core flows
20-29  Domain maps
30-39  ML / analytics
"""
    }

    for filename, content in diagrams.items():
        out_path = os.path.join(atlas_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")

    print("✅ Mermaid architecture atlas generated in:")
    print("   ", atlas_dir)
    print("   Files:", len(diagrams))


def build_index():
    """
    Builds an index of all python files in the backend:
    {
        "backend/api/routers/xxx.py": {"functions": [...], "routes": [...]},
        ...
    }
    """
    entries = scan_tree(
        BACKEND_ROOT,
        include_only=None,
        exclude_dirs=DEFAULT_EXCLUDE_DIRS
    )

    index = {}
    for rel, functions, routes in entries:
        index[rel.replace("\\", "/")] = {
            "functions": functions,
            "routes": routes
        }

    return index

def generate_flow_txt(flow_name: str, keywords: list[str], output_filename: str):
    """
    Example:
        flow_name = "Table View Flow"
        keywords = ["table_view", "reports"]
    """

    index = build_index()

    routers = {}
    services = {}
    dbs = {}

    for path, info in index.items():
        lower = path.lower()

        if not any(k in lower for k in keywords):
            continue

        if "/routers/" in lower:
            routers[path] = info
        elif "/services/" in lower:
            services[path] = info
        elif "/db_layer/" in lower:
            dbs[path] = info

    out_path = os.path.join(SCRIPT_DIR, output_filename)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {flow_name}\n\n")

        # ---------------- Routers ----------------
        f.write("## Routers\n")
        if not routers:
            f.write("(none found)\n")
        else:
            for path, info in routers.items():
                f.write(f"- {path}\n")
                for fn in info["functions"]:
                    f.write(f"  - {fn}()\n")
                for method, route in info["routes"]:
                    f.write(f"  - ROUTE {method} {route}\n")
        f.write("\n")

        # ---------------- Services ----------------
        f.write("## Services\n")
        if not services:
            f.write("(none found)\n")
        else:
            for path, info in services.items():
                f.write(f"- {path}\n")
                for fn in info["functions"]:
                    f.write(f"  - {fn}()\n")
        f.write("\n")

        # ---------------- DB Layer ----------------
        f.write("## DB Layer\n")
        if not dbs:
            f.write("(none found)\n")
        else:
            for path, info in dbs.items():
                f.write(f"- {path}\n")
                for fn in info["functions"]:
                    f.write(f"  - {fn}()\n")
        f.write("\n")

    print("✅ Flow file generated:", out_path)



# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":

    # 1) Full backend but excluding junk
    generate_map_excluding("_map_full_clean.txt")

    # 2) Focused maps
    generate_map_for_subtree("backend/api/services", "Services Layer", "_map_services.txt")
    generate_map_for_subtree("backend/api/db_layer", "DB Layer", "_map_db_layer.txt")
    generate_map_for_subtree("backend/api/routers", "Routers", "_map_routers.txt")

    # 3) Mermaid Architecture Atlas (files to paste into https://mermaid.live)
    write_mermaid_atlas("architecture_atlas")

    generate_flow_txt(
        flow_name="Table View Flow",
        keywords=["table_view", "reports"],
        output_filename="10_table_view_flow.txt"
    )