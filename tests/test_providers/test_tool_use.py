from lingxi.providers.claude import ClaudeProvider


def _provider():
    # api-key mode; we only test pure-function _build_body + _parse_content
    return ClaudeProvider(api_key="sk-test", model="claude-x")


def test_build_body_includes_tools():
    p = _provider()
    tools = [{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    body = p._build_body([{"role": "user", "content": "hi"}], None, 100, 0.7,
                         tools=tools, tool_choice={"type": "auto"})
    assert body["tools"] == tools
    assert body["tool_choice"] == {"type": "auto"}


def test_build_body_omits_tools_when_none():
    p = _provider()
    body = p._build_body([{"role": "user", "content": "hi"}], None, 100, 0.7)
    assert "tools" not in body
    assert "tool_choice" not in body


def test_parse_content_extracts_tool_use():
    p = _provider()
    blocks = [
        {"type": "text", "text": "thinking"},
        {"type": "tool_use", "id": "tu_1", "name": "archival_memory_search",
         "input": {"query": "stars"}},
    ]
    text, tool_calls = p._parse_content(blocks)
    assert text == "thinking"
    assert tool_calls == [{"id": "tu_1", "name": "archival_memory_search",
                           "input": {"query": "stars"}}]
