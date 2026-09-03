from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.app.responses import ok
from backend.app.services.pipeline_service import get_batch, list_batches, process_batch


router = APIRouter(prefix="/api/v1", tags=["JD批次处理"])


@router.post("/jd-batches")
async def create_jd_batch(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="JD 批次文件为空")
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="JD 批次不能超过 20 MB")
    try:
        return ok(await run_in_threadpool(process_batch, file.filename or "jobs.jsonl", content))
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/jd-batches")
def batches() -> dict:
    return ok(list_batches())


@router.get("/jd-batches/{batch_id}")
def batch_detail(batch_id: str) -> dict:
    try:
        return ok(get_batch(batch_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
