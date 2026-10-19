import json
from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from miscellaneous.initial_messages import get_initial_message
from models.schemas import ExplainRequest
from rag.interview_generator import astream_interview
from rag.assistant_generator import astream_explain, query_explain


router = APIRouter()


def _encode(token: str) -> str:
    return token.replace("\n", "<|newline|>")


async def _token_stream(request: ExplainRequest) -> AsyncGenerator[str, None]:
    metadata: dict = {}
    retrieval_meta_emitted = False
    async for token in astream_explain(request, metadata):
        if not retrieval_meta_emitted:
            retrieval_log = metadata.pop("retrieval_log", None)
            if retrieval_log is not None:
                yield f"event: retrieval_metadata\ndata: {json.dumps(retrieval_log)}\n\n"
            retrieval_meta_emitted = True
        yield f"data: {_encode(token)}\n\n"
    yield f"event: done\ndata: {json.dumps(metadata)}\n\n"


@router.post("/explanations")
async def explain(request: ExplainRequest):
    """
    Streams the answer provided by an LLM to a query

    Each SSE event contains one token. The stream ends with an SSE event
    `event: done` whose data is a JSON object containing metadata.
    """
    return StreamingResponse(
        _token_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/explanations/interview")
async def explain_interview(request: ExplainRequest):
    """Streaming endpoint for "interview" condition."""

    return StreamingResponse(
        _interview_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _interview_stream(request: ExplainRequest) -> AsyncGenerator[str, None]:
    metadata: dict = {}
    retrieval_meta_emitted = False
    async for token in astream_interview(request, metadata):
        if not retrieval_meta_emitted:
            retrieval_log = metadata.pop("retrieval_log", None)
            if retrieval_log is not None:
                yield f"event: retrieval_metadata\ndata: {json.dumps(retrieval_log)}\n\n"
            retrieval_meta_emitted = True
        yield f"data: {_encode(token)}\n\n"
    if metadata.get('show_buttons'):
        yield f"event: show_buttons\ndata: {{}}\n\n"
    yield f"event: done\ndata: {json.dumps(metadata)}\n\n"


@router.get(
    "/explanations/interview/initial-message",
    responses={404: {"description": "No initial message configured for this question index"}},
)
async def get_interview_initial_message(
    question_index: Annotated[int, Query()],
    language: Annotated[str, Query()] = "de",
):
    """Returns the predefined opening message for a given question in the interview condition."""
    content = get_initial_message(question_index, language)
    if content is None:
        raise HTTPException(status_code=404, detail="No initial message for this question index")
    return {"content": content}


@router.post("/explanations/completions")
async def explain_synchronous(request: ExplainRequest):
    '''
    Non-streaming version of /explanations. 
    Used for debugging 
    '''
    res = await query_explain(request)
    return Response(res)
