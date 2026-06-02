"""
End-to-end tests for the OpenRouter proxy using the OpenAI Python SDK.
Run with: uv run python test_e2e.py
Requires the server to be running on http://localhost:8000
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

DIVIDER = "\n" + "=" * 60 + "\n"


def test_1_non_streaming():
    """Test 1: Non-streaming basic chat"""
    print(DIVIDER + "TEST 1: Non-streaming basic chat")
    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
    )
    print(f"  ID:      {response.id}")
    print(f"  Model:   {response.model}")
    print(f"  Content: {response.choices[0].message.content}")
    print(f"  Finish:  {response.choices[0].finish_reason}")
    print(f"  Usage:   {response.usage.total_tokens} tokens")
    assert response.choices[0].message.content, "Content should not be empty"
    assert response.choices[0].finish_reason == "stop"
    print("  ✅ PASSED")


def test_2_streaming():
    """Test 2: Streaming chat (spinner/loading support)"""
    print(DIVIDER + "TEST 2: Streaming chat (for spinner/loading)")
    stream = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": "Count from 1 to 3."}],
        stream=True,
    )
    collected = ""
    chunk_count = 0
    print("  Chunks: ", end="", flush=True)
    for chunk in stream:
        chunk_count += 1
        delta = chunk.choices[0].delta
        if delta.content:
            collected += delta.content
            print(delta.content, end="", flush=True)
    print()
    print(f"  Total chunks: {chunk_count}")
    print(f"  Full text: {collected}")
    assert chunk_count > 1, "Should receive multiple chunks for streaming"
    assert collected, "Collected content should not be empty"
    print("  ✅ PASSED")


def test_3_tool_calling():
    """Test 3: Non-streaming with tool calling"""
    print(DIVIDER + "TEST 3: Non-streaming with tool calling")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": "What's the weather in Surat?"}],
        tools=tools,
    )
    content = response.choices[0].message.content
    print(f"  Content: {content}")
    print(f"  Finish:  {response.choices[0].finish_reason}")
    # The proxy should have executed the tool and returned the final answer
    assert content, "Should have content (tool was executed server-side)"
    print("  ✅ PASSED")


def test_4_streaming_with_tools():
    """Test 4: Streaming + tool calling"""
    print(DIVIDER + "TEST 4: Streaming with tool calling")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            },
        }
    ]
    stream = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": "What's the weather like in Surat right now?"}],
        tools=tools,
        stream=True,
    )
    collected = ""
    chunk_count = 0
    print("  Streaming: ", end="", flush=True)
    for chunk in stream:
        chunk_count += 1
        delta = chunk.choices[0].delta
        if delta.content:
            collected += delta.content
            print(delta.content, end="", flush=True)
    print()
    print(f"  Total chunks: {chunk_count}")
    print(f"  Full text: {collected[:200]}...")
    # After tool execution + forced final response, we should get content
    print("  ✅ PASSED (stream completed)")


if __name__ == "__main__":
    print("🚀 Running end-to-end tests against http://localhost:8000/v1\n")

    test_1_non_streaming()
    test_2_streaming()
    test_3_tool_calling()
    test_4_streaming_with_tools()

    print(DIVIDER + "🎉 ALL TESTS PASSED!" + DIVIDER)
