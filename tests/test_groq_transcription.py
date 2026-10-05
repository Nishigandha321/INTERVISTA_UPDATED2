from types import SimpleNamespace

import main


def immediate_result(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("The mocked transcription endpoint should not suspend")


def test_transcription_uses_groq_turbo_with_original_audio(monkeypatch):
    recorded = {}

    async def fake_to_thread(function, request_args):
        recorded["executor"] = function
        recorded.update(request_args)
        return SimpleNamespace(text="A clear spoken answer")

    monkeypatch.setattr(main.llm_service.settings, "groq_api_key", "test-key")
    monkeypatch.setattr(main.asyncio, "to_thread", fake_to_thread)
    class MemoryUpload:
        filename = "answer.webm"

        async def read(self):
            return b"webm-audio"

    result = immediate_result(main.transcribe_endpoint(MemoryUpload(), "FastAPI, Java"))

    assert result == {"transcript": "A clear spoken answer"}
    assert recorded["executor"] is main._groq_transcribe_audio
    assert recorded["model"] == "whisper-large-v3-turbo"
    assert recorded["language"] == "en"
    assert recorded["file"] == ("recording.webm", b"webm-audio")
    assert recorded["prompt"] == "FastAPI, Java"
    assert recorded["response_format"] == "json"
