from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter

from ..contracts import AsyncSubmission
from ...services.dispatch import (
    cancel_job,
    get_job,
    submit_async,
    submit_unified_async,
)

router = APIRouter(tags=["async-jobs"])


@router.post("/async/jobs")
def submit_unified_async_route(submission: AsyncSubmission) -> dict[str, Any]:
    # Keep the exact route before the legacy dynamic route so ``/async/jobs``
    # cannot be interpreted as the old ``{kind}`` path parameter.
    return submit_unified_async(submission)


@router.post("/async/{kind}")
def submit_async_route(
    kind: Literal["tn_estimate", "tn_amplitudes", "sample"],
    payload: dict[str, Any],
) -> dict[str, Any]:
    return submit_async(kind, payload)


@router.get("/async/jobs/{job_id}")
def get_job_route(job_id: str) -> dict[str, Any]:
    return get_job(job_id)


@router.post("/async/jobs/{job_id}/cancel")
def cancel_job_route(job_id: str) -> dict[str, Any]:
    return cancel_job(job_id)
