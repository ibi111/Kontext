from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from db.postgres import (
    create_chatroom, list_chatrooms,
    get_messages, get_or_create_chatroom
)

router = APIRouter(prefix="/chatrooms", tags=["chatrooms"])


class CreateChatroomRequest(BaseModel):
    session_id: str
    title: Optional[str] = "New chat"


@router.post("")
def create_new_chatroom(request: CreateChatroomRequest):
    """Create a new chatroom for a session."""
    chatroom_id = create_chatroom(request.session_id, request.title)
    return {"chatroom_id": chatroom_id, "title": request.title}


@router.get("")
def get_chatrooms(session_id: str):
    """List all chatrooms for a session, newest first."""
    return list_chatrooms(session_id)


@router.get("/{chatroom_id}/messages")
def get_chatroom_messages(chatroom_id: int, session_id: str):
    """
    Return full message history for a chatroom.
    session_id is required to prevent cross-session access.
    """
    rooms = list_chatrooms(session_id)
    if not any(r["id"] == chatroom_id for r in rooms):
        raise HTTPException(
            status_code=404,
            detail="Chatroom not found or does not belong to this session."
        )
    return get_messages(chatroom_id)
