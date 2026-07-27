

import json
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="Kontext Test Console", layout="wide")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit-{uuid.uuid4().hex[:8]}"
if "chatroom_id" not in st.session_state:
    st.session_state.chatroom_id = None
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # list of {"role", "content", "tool_events"?, "meta"?}
if "_rooms" not in st.session_state:
    st.session_state._rooms = None
if "_docs" not in st.session_state:
    st.session_state._docs = None


def fetch_rooms(base_url: str, force: bool = False):
    if force or st.session_state._rooms is None:
        try:
            r = requests.get(
                f"{base_url}/chatrooms",
                params={"session_id": st.session_state.session_id},
                timeout=15,
            )
            r.raise_for_status()
            st.session_state._rooms = r.json()
        except Exception as e:
            st.session_state._rooms = []
            st.error(f"Request failed: {e}")
    return st.session_state._rooms


def load_room_messages(base_url: str, chatroom_id: int):
    try:
        r = requests.get(
            f"{base_url}/chatrooms/{chatroom_id}/messages",
            params={"session_id": st.session_state.session_id},
            timeout=15,
        )
        r.raise_for_status()
        log = []
        for m in r.json():
            if m["role"] == "tool":
                # attach tool messages to the previous assistant entry if present
                if log and log[-1]["role"] == "assistant":
                    log[-1].setdefault("tool_events", []).append(
                        {"name": m["tool_name"], "args": m["tool_args"], "result": m["tool_result"]}
                    )
                continue
            entry = {"role": m["role"], "content": m["content"] or ""}
            if m.get("model_used"):
                entry["meta"] = {"model_used": m["model_used"]}
            log.append(entry)
        st.session_state.chat_log = log
    except Exception as e:
        st.error(f"Request failed: {e}")


# --------------------------------------------------------------------------
# Sidebar -- connection + session controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Kontext Console")
    base_url = st.text_input("API base URL", value="http://localhost:8000").rstrip("/")

    st.divider()
    st.subheader("Session")
    st.text_input("session_id", key="session_id")
    if st.button("New session_id", use_container_width=True):
        st.session_state.session_id = f"streamlit-{uuid.uuid4().hex[:8]}"
        st.session_state.chatroom_id = None
        st.session_state.chat_log = []
        st.session_state._rooms = None
        st.rerun()

    st.divider()
    if st.button("Check /health", use_container_width=True):
        try:
            r = requests.get(f"{base_url}/health", timeout=10)
            st.session_state["_health_preview"] = r.json()
        except Exception as e:
            st.session_state["_health_preview"] = {"error": str(e)}
    if "_health_preview" in st.session_state:
        st.json(st.session_state["_health_preview"], expanded=False)

tab_chat, tab_ingest, tab_health = st.tabs(["Chat", "Ingest", "Health"])

# --------------------------------------------------------------------------
# Chat tab -- ChatGPT-style layout: room list on the left, conversation +
# input box on the right. Combines the old Chatrooms and Chat tabs.
# --------------------------------------------------------------------------
with tab_chat:
    room_col, chat_col = st.columns([1, 3])

    with room_col:
        if st.button("New chat", use_container_width=True, type="primary"):
            st.session_state.chatroom_id = None
            st.session_state.chat_log = []
            st.rerun()

        st.caption("Chatrooms")
        rooms = fetch_rooms(base_url)
        for room in rooms:
            selected = room["id"] == st.session_state.chatroom_id
            label = room["title"] or "New chat"
            if st.button(
                label,
                key=f"room_{room['id']}",
                use_container_width=True,
                type="secondary" if not selected else "primary",
            ):
                st.session_state.chatroom_id = room["id"]
                load_room_messages(base_url, room["id"])
                st.rerun()

        if not rooms:
            st.caption("No chatrooms yet for this session.")

    with chat_col:
        for entry in st.session_state.chat_log:
            with st.chat_message(entry["role"]):
                st.write(entry["content"])
                if entry.get("tool_events"):
                    with st.expander("Tool calls"):
                        for te in entry["tool_events"]:
                            st.json(te, expanded=False)
                if entry.get("meta"):
                    m = entry["meta"]
                    st.caption(
                        f"model={m.get('model_used')}  |  "
                        f"cache_hit={m.get('cache_hit')}  |  "
                        f"usage={m.get('usage')}"
                    )

        query = st.chat_input("Message Kontext...")

        if query:
            st.session_state.chat_log.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                tool_box = st.expander("Tool calls (live)", expanded=False)
                accumulated = ""
                tool_events = []
                final_meta = {}

                payload = {
                    "query": query,
                    "session_id": st.session_state.session_id,
                    "chatroom_id": st.session_state.chatroom_id,
                }

                try:
                    with requests.post(f"{base_url}/chat", json=payload, stream=True, timeout=300) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line or not line.startswith("data:"):
                                continue
                            raw = line[len("data:"):].strip()
                            if not raw:
                                continue
                            try:
                                chunk = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            ctype = chunk.get("type")

                            if ctype == "token":
                                accumulated += chunk.get("content") or ""
                                placeholder.markdown(accumulated)

                            elif ctype == "tool_start":
                                tool_events.append({"event": "tool_start", **chunk})
                                with tool_box:
                                    st.write(f"calling: {chunk.get('content')}")
                                    st.json(chunk.get("meta", {}), expanded=False)

                            elif ctype == "tool_end":
                                tool_events.append({"event": "tool_end", **chunk})
                                with tool_box:
                                    st.write(f"returned: {chunk.get('content')}")
                                    st.json(chunk.get("meta", {}), expanded=False)

                            elif ctype == "cache_hit":
                                accumulated = chunk.get("content") or ""
                                placeholder.markdown(accumulated)
                                st.session_state.chatroom_id = chunk.get("chatroom_id", st.session_state.chatroom_id)
                                final_meta = {"cache_hit": True}

                            elif ctype == "blocked":
                                accumulated = chunk.get("content") or ""
                                placeholder.markdown(f"[blocked] {accumulated}")

                            elif ctype == "done":
                                if not accumulated:
                                    accumulated = chunk.get("content") or ""
                                placeholder.markdown(accumulated)
                                final_meta = chunk.get("meta", {})
                                st.session_state.chatroom_id = chunk.get("chatroom_id", st.session_state.chatroom_id)

                            elif ctype == "error":
                                accumulated = f"Error: {chunk.get('content')}"
                                placeholder.markdown(accumulated)

                except Exception as e:
                    accumulated = f"Request failed: {e}"
                    placeholder.markdown(accumulated)

                st.session_state.chat_log.append({
                    "role": "assistant",
                    "content": accumulated,
                    "tool_events": tool_events,
                    "meta": final_meta,
                })

            # a chatroom may have just been created by this turn -- refresh the list
            fetch_rooms(base_url, force=True)
            st.rerun()

# --------------------------------------------------------------------------
# Ingest tab
# --------------------------------------------------------------------------
with tab_ingest:
    st.header("Document ingestion")

    up_col, list_col = st.columns([1, 1.4])

    with up_col:
        st.subheader("Upload a PDF")
        uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
        if st.button("Ingest document", disabled=uploaded is None, type="primary"):
            with st.spinner("Uploading & running ingestion graph (validate -> parse -> chunk -> enrich -> index)..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                    r = requests.post(f"{base_url}/ingest", files=files, timeout=600)
                    if r.status_code >= 400:
                        st.error(f"{r.status_code}: {r.text}")
                    else:
                        st.success("Ingestion complete")
                        st.json(r.json())
                        st.session_state._docs = None
                except Exception as e:
                    st.error(f"Request failed: {e}")

    with list_col:
        st.subheader("Ingested documents")
        if st.button("Refresh list"):
            st.session_state._docs = None
        if st.session_state._docs is None:
            try:
                r = requests.get(f"{base_url}/ingest", timeout=15)
                r.raise_for_status()
                st.session_state._docs = r.json()
            except Exception as e:
                st.session_state._docs = []
                st.error(f"Request failed: {e}")

        docs = st.session_state._docs or []
        if not docs:
            st.info("No documents ingested yet.")
        for d in docs:
            c1, c2, c3, c4 = st.columns([0.5, 3, 1.3, 1])
            c1.write(f"#{d['id']}")
            c2.write(d["filename"])
            c3.write(d["status"])
            if c4.button("Delete", key=f"del_{d['id']}"):
                try:
                    r = requests.delete(f"{base_url}/ingest/{d['id']}", timeout=30)
                    r.raise_for_status()
                    st.session_state._docs = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

# --------------------------------------------------------------------------
# Health tab
# --------------------------------------------------------------------------
with tab_health:
    st.header("Service health")
    st.caption("GET /health -- checks Postgres, Qdrant, Redis")

    if st.button("Check health now"):
        try:
            r = requests.get(f"{base_url}/health", timeout=15)
            r.raise_for_status()
            data = r.json()
            overall = data.get("status", "unknown")
            (st.success if overall == "ok" else st.error)(f"Overall status: {overall}")

            cols = st.columns(len(data.get("services", {})) or 1)
            for col, (name, status) in zip(cols, data.get("services", {}).items()):
                with col:
                    ok = isinstance(status, str) and status.startswith("ok")
                    st.metric(name, "ok" if ok else "error")
                    if not ok:
                        st.code(status, language=None)
        except Exception as e:
            st.error(f"Request failed: {e}")
