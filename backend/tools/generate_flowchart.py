from typing import Any, Dict, Optional

async def generate_flowchart(
    diagram_type: str,
    mermaid_code: Optional[str] = None,
    schema_data: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    30_ToolSpecifications.md §30.4 & 12_FlowchartArchitecture.md - generate_flowchart tool implementation.
    Supports Path A (Auto-ER from schema_data) and Path B (Mermaid pass-through).
    """
    if diagram_type not in ["er", "flowchart", "sequence"]:
        return {
            "success": False,
            "error": f"diagram_type '{diagram_type}' is not supported. Use: er, flowchart, sequence"
        }

    # Path A: Auto-ER from schema_data (or fetch if not provided)
    if diagram_type == "er" and not schema_data and not mermaid_code:
        from tools.get_schema import get_schema
        schema_data = await get_schema()

    if schema_data and "tables" in schema_data:
        mermaid_lines = ["erDiagram"]
        tables = schema_data["tables"]
        for t in tables:
            tname = t["name"].upper()
            mermaid_lines.append(f"    {tname} {{")
            for c in t.get("columns", []):
                ctype = c.get("type", "TEXT")
                cname = c.get("name")
                pk = "PK" if c.get("pk") else ""
                mermaid_lines.append(f"        {ctype} {cname} {pk}".strip())
            mermaid_lines.append("    }")

            for fk in t.get("foreign_keys", []):
                target = fk.get("target_table", "").upper() or fk.get("table", "").upper()
                if target:
                    mermaid_lines.append(f"    {tname} ||--o{{ {target} : \"references\"")
        
        return {
            "success": True,
            "diagram_type": "er",
            "title": title or "Database ER Diagram",
            "mermaid": "\n".join(mermaid_lines)
        }

    # Path B: User/LLM provided mermaid_code
    if mermaid_code:
        code = mermaid_code.strip()
        if diagram_type == "flowchart" and not (code.startswith("flowchart") or code.startswith("graph")):
            code = f"flowchart TD\n    {code}"
        elif diagram_type == "er" and not code.startswith("erDiagram"):
            code = f"erDiagram\n    {code}"
        elif diagram_type == "sequence" and not code.startswith("sequenceDiagram"):
            code = f"sequenceDiagram\n    {code}"
        return {
            "success": True,
            "diagram_type": diagram_type,
            "title": title or f"{diagram_type.capitalize()} Diagram",
            "mermaid": code
        }

    return {
        "success": False,
        "error": "Either mermaid_code or schema_data must be provided to generate a flowchart."
    }
