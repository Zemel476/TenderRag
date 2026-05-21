from datetime import datetime


class ConversationStore:
    """In-memory conversation store keyed by session_id."""

    def __init__(self, max_turns: int = 20):
        self._store: dict[str, list[dict]] = {}
        self.max_turns = max_turns

    def get_messages(self, session_id: str) -> list[dict]:
        return self._store.get(session_id, [])

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ):
        if session_id not in self._store:
            self._store[session_id] = []

        self._store[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

        if len(self._store[session_id]) > self.max_turns:
            self._store[session_id] = self._store[session_id][-self.max_turns:]

    def clear(self, session_id: str):
        self._store.pop(session_id, None)


memory_store = ConversationStore()
