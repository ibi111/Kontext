"""
Postgres data layer.

Tables:
    chatrooms  - one row per chat thread, owned by a session_id
    messages   - every user/assistant/tool message in a chatroom, in order.
                 Tool calls are persisted here (name, args, result as JSONB)
                 exactly as they happen
    documents  - one row per ingested PDF
    chunks     - one row per chunk: text + metadata + qdrant_point_id.
                 The actual embedding vector is NOT stored here, it lives
                 in Qdrant.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from config import POSTGRES_URL

engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Chatroom(Base):
    __tablename__ = "chatrooms"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)  # no auth atm
    title = Column(String, nullable=False, default="New chat")
    created_at = Column(DateTime, server_default=func.now())

    messages = relationship("Message", back_populates="chatroom", order_by="Message.id")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    chatroom_id = Column(Integer, ForeignKey("chatrooms.id"), nullable=False, index=True)
    role = Column(String, nullable=False)          # 'user' | 'assistant' | 'tool'
    content = Column(Text, nullable=True)           # assistant/user text; tool rows may leave this null
    tool_name = Column(String, nullable=True)        # set only for role='tool'
    tool_args = Column(JSONB, nullable=True)
    tool_result = Column(JSONB, nullable=True)
    model_used = Column(String, nullable=True)       # which OpenRouter model produced this (assistant rows)
    step_number = Column(Integer, nullable=True)     # position within one agent turn, for ordered replay
    created_at = Column(DateTime, server_default=func.now())

    chatroom = relationship("Chatroom", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending | processing | ready | failed
    created_at = Column(DateTime, server_default=func.now())


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    page = Column(Integer)
    section = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    qdrant_point_id = Column(String, nullable=False)  # pointer into Qdrant; vector itself lives there


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()



# Chatrooms

def create_chatroom(session_id: str, title: str = "New chat") -> int:
    session = get_session()
    room = Chatroom(session_id=session_id, title=title[:80])
    session.add(room)
    session.commit()
    room_id = room.id
    session.close()
    return room_id


def get_or_create_chatroom(session_id: str, chatroom_id: int | None = None) -> int:
    """
    If chatroom_id is given and belongs to this session, reuse it.
    Otherwise create a fresh chatroom for this session — mirrors how the
    baseline always had a conversation_id to write into before streaming
    started.
    """
    session = get_session()
    if chatroom_id is not None:
        room = session.query(Chatroom).filter_by(id=chatroom_id, session_id=session_id).first()
        if room:
            session.close()
            return chatroom_id
    room = Chatroom(session_id=session_id, title="New chat")
    session.add(room)
    session.commit()
    room_id = room.id
    session.close()
    return room_id


def list_chatrooms(session_id: str) -> list[dict]:
    session = get_session()
    rooms = (
        session.query(Chatroom)
        .filter_by(session_id=session_id)
        .order_by(Chatroom.id.desc())
        .all()
    )
    out = [{"id": r.id, "title": r.title, "created_at": r.created_at.isoformat()} for r in rooms]
    session.close()
    return out


# Messages (including tool call persistence)


def add_message(
    chatroom_id: int,
    role: str,
    content: str | None = None,
    tool_name: str | None = None,
    tool_args: dict | None = None,
    tool_result: dict | None = None,
    model_used: str | None = None,
    step_number: int | None = None,
) -> int:
    session = get_session()
    msg = Message(
        chatroom_id=chatroom_id,
        role=role,
        content=content,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        model_used=model_used,
        step_number=step_number,
    )
    session.add(msg)
    session.commit()
    msg_id = msg.id
    session.close()
    return msg_id


def get_messages(chatroom_id: int) -> list[dict]:
    session = get_session()
    rows = (
        session.query(Message)
        .filter_by(chatroom_id=chatroom_id)
        .order_by(Message.id.asc())
        .all()
    )
    out = [{
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "tool_name": m.tool_name,
        "tool_args": m.tool_args,
        "tool_result": m.tool_result,
        "model_used": m.model_used,
        "step_number": m.step_number,
        "created_at": m.created_at.isoformat(),
    } for m in rows]
    session.close()
    return out


def history_to_chat_messages(chatroom_id: int) -> list[dict]:
    """
    Rebuild prior turns as OpenAI-format chat messages (user/assistant text
    only — tool payloads stay in the DB for the UI but aren't replayed into
    the prompt, same choice the baseline made).
    """
    messages = []
    for m in get_messages(chatroom_id):
        if m["role"] == "user":
            messages.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m["content"]:
            messages.append({"role": "assistant", "content": m["content"]})
    return messages