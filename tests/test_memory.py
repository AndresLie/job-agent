from career_copilot.memory import MemoryStore


def test_memory_deduplicates_by_summary(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    assert store.add("Use pytest for verification", tags=["testing"]) is True
    assert store.add("Use pytest for verification", tags=["testing"]) is False
    assert len(store.items) == 1


def test_memory_retrieves_relevant_item(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    store.add("Prefer backend AI roles using retrieval and evaluation")
    store.add("Favorite editor theme is dark mode")
    hits = store.retrieve("retrieval evaluation role", k=1)
    assert hits[0].summary.startswith("Prefer backend")
