from lingxi.brain.memory_tools import MEMORY_TOOLS, TOOL_NAMES


def test_five_tools_defined():
    assert TOOL_NAMES == {
        "archival_memory_search", "archival_memory_insert",
        "core_memory_append", "core_memory_replace", "conversation_search",
    }


def test_each_tool_has_valid_schema():
    for t in MEMORY_TOOLS:
        assert "name" in t and "description" in t
        assert t["input_schema"]["type"] == "object"
        assert "properties" in t["input_schema"]
