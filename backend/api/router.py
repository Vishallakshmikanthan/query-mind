from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from agent.session import session_store
from agent.orchestrator import agent_orchestrator
from api.auth import require_auth, get_current_user, _verify_basic_auth, _load_auth_config, create_access_token
from api.rate_limit import check_rate_limit
from api.schemas import (
    ChatRequest,
    HistoryResponse,
    MessageRecord,
    SchemaResponse,
    LoginRequest,
    TokenResponse,
    UserResponse,
)
from tools.get_schema import get_schema
from tools.execute_query import execute_query

router = APIRouter(prefix="/api")


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: Request, credentials: LoginRequest):
    """POST /api/auth/login - Exchange username/password for a JWT access token."""
    config = _load_auth_config()
    basic_creds = type("BasicCreds", (), {"username": credentials.username, "password": credentials.password})
    if not config["enabled"] or not _verify_basic_auth(basic_creds, config):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_CREDENTIALS", "message": "Invalid username or password."}
        )
    token = create_access_token(credentials.username, config)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=config["token_expire_hours"] * 3600
    )


@router.get("/auth/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    """GET /api/auth/me - Return the currently authenticated user."""
    return UserResponse(username=current_user.get("sub", "anonymous"), role=current_user.get("role", "guest"))

@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    user: dict = Depends(require_auth),
    _: None = Depends(check_rate_limit)
):
    """
    POST /api/chat - Main conversational streaming endpoint powered by AgentOrchestrator.
    Emits an SSE stream of typed events (08_APIArchitecture.md & 05_AgentArchitecture.md).
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_MESSAGE", "message": "Message cannot be empty."}
        )

    return StreamingResponse(
        agent_orchestrator.process_message_stream(
            message=request.message,
            session_id=request.session_id,
            show_sql=request.options.show_sql
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/session/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(
    session_id: str,
    user: dict = Depends(require_auth)
):
    """GET /api/session/{id}/history - Retrieves message history for a session."""
    messages = session_store.get_messages(session_id)
    records = [MessageRecord(**msg) for msg in messages]
    return HistoryResponse(
        session_id=session_id,
        messages=records,
        message_count=len(records)
    )

@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    user: dict = Depends(require_auth)
):
    """DELETE /api/session/{id} - Clears message history for a session."""
    success = session_store.clear_session(session_id)
    return {"success": success, "message": f"Session {session_id} cleared."}

@router.get("/schema", response_model=SchemaResponse)
async def direct_schema_introspection(user: dict = Depends(require_auth)):
    """GET /api/schema - Direct PRAGMA schema introspection for UI schema panel."""
    res = await get_schema()
    if not res.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DB_ERROR", "message": res.get("error", "Failed to fetch schema")}
        )
    return SchemaResponse(
        success=True,
        tables=res["tables"],
        total_tables=res["total_tables"]
    )

import csv
import io
from fastapi.responses import Response
from api.schemas import ExportRequest

@router.post("/export/csv")
async def export_csv(
    req: ExportRequest,
    user: dict = Depends(require_auth)
):
    """POST /api/export/csv - Execute SELECT query and stream CSV file download."""
    res = await execute_query(req.sql)
    if not res.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "SQL_UNSAFE" if "Forbidden" in res.get("error", "") else "DB_ERROR", "message": res.get("error")}
        )
    
    output = io.StringIO()
    writer = csv.writer(output)
    cols = res.get("columns", [])
    writer.writerow(cols)
    for row in res.get("rows", []):
        writer.writerow([row.get(col, "") for col in cols])
        
    filename = req.filename or "export"
    if not filename.endswith(".csv"):
        filename = f"{filename}.csv"
        
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


from fastapi import UploadFile, File
from pydantic import BaseModel
from tools.ingest_dataset import ingest_file, download_kaggle_dataset


class KaggleImportRequest(BaseModel):
    url_or_code: str


@router.post("/dataset/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    user: dict = Depends(require_auth)
):
    """POST /api/dataset/upload - Ingest uploaded ZIP, CSV, or SQLite dataset file."""
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_FILE", "message": "Uploaded file is empty."}
        )
    res = await ingest_file(file.filename or "uploaded_dataset.csv", content)
    if not res.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INGEST_FAILED", "message": res.get("error", "Failed to ingest dataset")}
        )
    return res


@router.post("/dataset/kaggle")
async def kaggle_dataset(
    req: KaggleImportRequest,
    user: dict = Depends(require_auth)
):
    """POST /api/dataset/kaggle - Fetch and ingest Kaggle dataset by URL or code."""
    if not req.url_or_code or not req.url_or_code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_KAGGLE_INPUT", "message": "Kaggle URL or code cannot be empty."}
        )
    res = await download_kaggle_dataset(req.url_or_code.strip())
    if not res.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "KAGGLE_IMPORT_FAILED", "message": res.get("error", "Failed to import Kaggle dataset")}
        )
    return res

