"""
DataFlow AI — Dataset Ingestion Tool
Handles dataset uploads (ZIP, CSV, SQLite) and Kaggle dataset code/URL imports.
"""

import os
import io
import zipfile
import csv
import re
import json
import base64
import tempfile
import sqlite3
from pathlib import Path
from typing import Dict, Any, List

from db.adapters.sqlite import _resolve_db_path


def get_target_db_path() -> str:
    """Resolve active target SQLite DB path."""
    return _resolve_db_path(None)


def sanitize_table_name(name: str) -> str:
    """Sanitize filename to a valid SQLite table name."""
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean or "uploaded_dataset"


def sanitize_column_names(headers: List[str]) -> List[str]:
    """Sanitize column headers and ensure unique names."""
    seen = {}
    clean_cols = []
    for i, h in enumerate(headers):
        clean = sanitize_table_name(h.strip()) if h and h.strip() else f"col_{i+1}"
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 1
        clean_cols.append(clean)
    return clean_cols


def decode_bytes(content: bytes) -> str:
    """Safely decode bytes trying common text encodings."""
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


async def ingest_csv_bytes(csv_content: bytes, table_name: str) -> Dict[str, Any]:
    """Parse CSV bytes and insert as a table into SQLite database."""
    try:
        table_name = sanitize_table_name(table_name)
        text = decode_bytes(csv_content)
        reader = csv.reader(io.StringIO(text))
        headers = next(reader, None)
        if not headers:
            return {"success": False, "error": "CSV file is empty or invalid."}

        clean_cols = sanitize_column_names(headers)
        db_path = get_target_db_path()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Drop existing table if exists
        cursor.execute(f"DROP TABLE IF EXISTS \"{table_name}\"")

        # Create table with sanitized columns
        col_defs = ", ".join([f"\"{c}\" TEXT" for c in clean_cols])
        cursor.execute(f"CREATE TABLE \"{table_name}\" ({col_defs})")

        # Insert rows
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
        if rows:
            placeholders = ", ".join(["?"] * len(clean_cols))
            normalized_rows = [
                row[:len(clean_cols)] + [""] * max(0, len(clean_cols) - len(row))
                for row in rows
            ]
            cursor.executemany(f"INSERT INTO \"{table_name}\" VALUES ({placeholders})", normalized_rows)

        conn.commit()
        conn.close()

        return {
            "success": True,
            "table_name": table_name,
            "tables_added": [table_name],
            "rows_inserted": len(rows),
            "columns": clean_cols,
            "message": f"Successfully imported table '{table_name}' with {len(rows)} rows."
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to ingest CSV: {str(e)}"}


async def ingest_sqlite_bytes(sqlite_content: bytes) -> Dict[str, Any]:
    """Ingest tables from an uploaded SQLite database file into active database."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
            tmp.write(sqlite_content)
            tmp_path = tmp.name

        target_db = get_target_db_path()
        src_conn = sqlite3.connect(tmp_path)
        src_cursor = src_conn.cursor()
        src_cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = src_cursor.fetchall()

        dst_conn = sqlite3.connect(target_db)
        dst_cursor = dst_conn.cursor()

        tables_added = []
        for tbl_name, tbl_sql in tables:
            clean_tbl = sanitize_table_name(tbl_name)
            src_cursor.execute(f"SELECT * FROM \"{tbl_name}\"")
            rows = src_cursor.fetchall()
            col_names = [d[0] for d in src_cursor.description]
            clean_cols = sanitize_column_names(col_names)

            dst_cursor.execute(f"DROP TABLE IF EXISTS \"{clean_tbl}\"")
            col_defs = ", ".join([f"\"{c}\" TEXT" for c in clean_cols])
            dst_cursor.execute(f"CREATE TABLE \"{clean_tbl}\" ({col_defs})")

            if rows:
                placeholders = ", ".join(["?"] * len(clean_cols))
                dst_cursor.executemany(f"INSERT INTO \"{clean_tbl}\" VALUES ({placeholders})", rows)
            tables_added.append(clean_tbl)

        dst_conn.commit()
        dst_conn.close()
        src_conn.close()

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return {
            "success": True,
            "tables_added": tables_added,
            "message": f"Successfully ingested {len(tables_added)} table(s) from SQLite database."
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to ingest SQLite database: {str(e)}"}


async def ingest_zip_bytes(zip_content: bytes) -> Dict[str, Any]:
    """Extract ZIP file and ingest contained CSV/SQLite files."""
    try:
        tables_added = []
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            for filename in z.namelist():
                if filename.startswith('__MACOSX/') or filename.endswith('/'):
                    continue
                lower_fn = filename.lower()
                if lower_fn.endswith('.csv'):
                    base_name = Path(filename).stem
                    data = z.read(filename)
                    res = await ingest_csv_bytes(data, base_name)
                    if res.get("success") and res.get("table_name"):
                        tables_added.append(res["table_name"])
                elif lower_fn.endswith('.sqlite') or lower_fn.endswith('.db'):
                    data = z.read(filename)
                    res = await ingest_sqlite_bytes(data)
                    if res.get("success") and res.get("tables_added"):
                        tables_added.extend(res["tables_added"])

        if not tables_added:
            return {"success": False, "error": "No CSV or SQLite database files found in ZIP archive."}

        return {
            "success": True,
            "tables_added": tables_added,
            "table_name": tables_added[0] if tables_added else None,
            "message": f"Successfully ingested {len(tables_added)} table(s): {', '.join(tables_added)}"
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to ingest ZIP archive: {str(e)}"}


async def ingest_file(filename: str, content: bytes) -> Dict[str, Any]:
    """Ingest a file based on its extension (ZIP, CSV, SQLite)."""
    lower_fn = filename.lower()
    if lower_fn.endswith('.zip') or zipfile.is_zipfile(io.BytesIO(content)):
        return await ingest_zip_bytes(content)
    elif lower_fn.endswith('.csv'):
        base_name = Path(filename).stem
        return await ingest_csv_bytes(content, base_name)
    elif lower_fn.endswith('.sqlite') or lower_fn.endswith('.db'):
        return await ingest_sqlite_bytes(content)
    else:
        # Try CSV parser first, then ZIP parser
        csv_res = await ingest_csv_bytes(content, Path(filename).stem)
        if csv_res.get("success"):
            return csv_res
        return {"success": False, "error": f"Unsupported file type for '{filename}'. Please upload a ZIP, CSV, or SQLite file."}


def _get_kaggle_auth_headers() -> Dict[str, str]:
    """Extract Kaggle authentication credentials if present."""
    headers = {"User-Agent": "Mozilla/5.0"}

    # 1. Check ~/.kaggle/kaggle.json
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json_path):
        try:
            with open(kaggle_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            token = data.get("token") or data.get("key")
            username = data.get("username")
            if token and not username:
                # Kaggle API Bearer Token
                headers["Authorization"] = f"Bearer {token}"
                return headers
            elif username and token:
                # Kaggle Basic Auth
                cred = base64.b64encode(f"{username}:{token}".encode()).decode()
                headers["Authorization"] = f"Basic {cred}"
                return headers
        except Exception:
            pass

    # 2. Check environment variables
    api_token = os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_TOKEN")
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
        return headers

    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    if kaggle_user and kaggle_key:
        cred = base64.b64encode(f"{kaggle_user}:{kaggle_key}".encode()).decode()
        headers["Authorization"] = f"Basic {cred}"
        return headers

    return headers


async def download_kaggle_dataset(kaggle_url_or_code: str) -> Dict[str, Any]:
    """
    Import dataset from Kaggle code or dataset URL.
    Supports formats:
    - https://www.kaggle.com/datasets/user/dataset-name
    - kaggle datasets download -d user/dataset-name
    - user/dataset-name
    """
    try:
        raw_input = kaggle_url_or_code.strip()
        # Extract user/dataset-name slug
        match = re.search(r'(?:kaggle\.com/datasets/|download -d\s+|^)([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)', raw_input)
        slug = match.group(1) if match else raw_input
        # Remove trailing slashes or subpaths if present
        slug = "/".join(slug.split("/")[:2])
        dataset_name = slug.split('/')[-1]

        download_url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
        headers = _get_kaggle_auth_headers()

        import requests
        response = requests.get(download_url, headers=headers, allow_redirects=True, timeout=60)
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Kaggle download returned status {response.status_code}: {response.text[:200]}"
            }

        content = response.content
        if not content:
            return {"success": False, "error": "Downloaded Kaggle dataset is empty."}

        if zipfile.is_zipfile(io.BytesIO(content)):
            res = await ingest_zip_bytes(content)
            res["dataset_slug"] = slug
            return res
        else:
            res = await ingest_csv_bytes(content, dataset_name)
            res["dataset_slug"] = slug
            return res
    except Exception as e:
        return {
            "success": False,
            "error": f"Unable to fetch Kaggle dataset '{kaggle_url_or_code}': {str(e)}"
        }
