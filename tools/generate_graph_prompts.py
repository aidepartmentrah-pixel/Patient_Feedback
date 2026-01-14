import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Where your flow txt files are
FLOW_FILES_DIR = SCRIPT_DIR

# Where we will write the prompt files
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "graph_prompts")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def nice_name_from_filename(filename: str) -> str:
    """
    10_table_view_flow.txt -> TableView
    11_insert_flow.txt -> Insert
    12_followup_flow.txt -> Followup
    """
    name = filename.lower()
    name = re.sub(r"^\d+_", "", name)
    name = name.replace("_flow.txt", "")
    name = name.replace(".txt", "")
    parts = name.split("_")
    return "".join(p.capitalize() for p in parts)


def build_prompt(flow_name: str, content: str) -> str:
    return f"""
You are a senior software architect.

Below is a REAL architecture slice extracted from a FastAPI backend.

Your task:
1) Identify the main business flow (Router → Service → DB)
2) Ignore secondary endpoints and noise
3) Generate a CLEAN Mermaid flowchart for the MAIN use-case
4) Use REAL module names (not abstract ones)
5) Keep the diagram simple and readable

Name this diagram:
Graph_For_{flow_name}

ARCHITECTURE SLICE:
===================
{content}

OUTPUT:
- Return ONLY the Mermaid diagram code.
""".strip()


def main():
    ensure_dir(OUTPUT_DIR)

    generated = 0

    for file in os.listdir(FLOW_FILES_DIR):
        if not file.endswith("_flow.txt"):
            continue

        flow_path = os.path.join(FLOW_FILES_DIR, file)

        with open(flow_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        flow_name = nice_name_from_filename(file)

        prompt_text = build_prompt(flow_name, content)

        output_filename = f"Graph_For_{flow_name}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        with open(output_path, "w", encoding="utf-8") as out:
            out.write(prompt_text)

        print("✅ Generated:", output_path)
        generated += 1

    print()
    print("🎉 Done.")
    print("Generated", generated, "graph prompt files in:")
    print(" ", OUTPUT_DIR)


if __name__ == "__main__":
    main()
