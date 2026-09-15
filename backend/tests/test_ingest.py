import pytest
import io
import zipfile
from tools.ingest_dataset import ingest_csv_bytes, ingest_zip_bytes, ingest_file, sanitize_table_name
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_sanitize_table_name():
    assert sanitize_table_name("My Custom Dataset 2026!") == "my_custom_dataset_2026"

@pytest.mark.asyncio
async def test_ingest_csv_bytes():
    csv_data = b"product_id,product_name,price\n101,Gadget,49.99\n102,Widget,19.99\n"
    res = await ingest_csv_bytes(csv_data, "test_products_upload")
    assert res.get("success") is True
    assert res.get("table_name") == "test_products_upload"
    assert res.get("rows_inserted") == 2

@pytest.mark.asyncio
async def test_ingest_zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("orders_sample.csv", "order_id,amount\n1,50.0\n2,100.0\n")
    res = await ingest_zip_bytes(buf.getvalue())
    assert res.get("success") is True
    assert "orders_sample" in res.get("tables_added", [])

def test_api_upload_dataset_csv():
    csv_content = b"user_id,username,score\n1,alice,95\n2,bob,88\n"
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("leaderboard.csv", csv_content, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") is True
    assert data.get("table_name") == "leaderboard"

def test_api_upload_dataset_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("customers_v2.csv", "id,name\n1,Charlie\n")
    response = client.post(
        "/api/dataset/upload",
        files={"file": ("customers_v2.zip", buf.getvalue(), "application/zip")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("success") is True
    assert "customers_v2" in data.get("tables_added", [])
