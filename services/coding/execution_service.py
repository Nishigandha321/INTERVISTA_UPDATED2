"""Small async client for the OnlineCompiler synchronous execution API."""

from __future__ import annotations

from typing import Any

import httpx

from config.settings import settings


class CodingExecutionError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class CodingExecutionConfigurationError(CodingExecutionError):
    pass


class CodingExecutionService:
    """Synchronous OnlineCompiler client; each POST is one execution."""

    COMPILERS = {"cpp": "g++-15", "python": "python-3.14"}

    def __init__(self, client_factory=httpx.AsyncClient):
        self.client_factory = client_factory

    def _base_url(self) -> str:
        base_url = settings.onlinecompiler_base_url.strip().rstrip("/")
        if not base_url:
            raise CodingExecutionConfigurationError(
                "Coding execution is not configured yet. Please try again later."
            )
        if not base_url.startswith(("https://", "http://")):
            raise CodingExecutionConfigurationError("Coding execution configuration is invalid.")
        return base_url

    def _headers(self) -> dict[str, str]:
        api_key = settings.onlinecompiler_api_key.strip()
        if not api_key or "\r" in api_key or "\n" in api_key:
            raise CodingExecutionConfigurationError(
                "Coding execution authentication is not configured correctly."
            )
        return {"Authorization": api_key, "Content-Type": "application/json"}

    def validate_language(self, language: str) -> str:
        """Validate provider configuration/language without making a paid execution call."""
        if language not in self.COMPILERS:
            raise ValueError("Unsupported coding language.")
        self._base_url()
        self._headers()
        return self.COMPILERS[language]

    @staticmethod
    def _looks_like_compile_error(compiler: str, error: str) -> bool:
        message = error.lower()
        if compiler == "g++-15":
            return any(marker in message for marker in (
                "fatal error:", "error:", "undefined reference", "collect2:",
                "compilation terminated", "cannot find", "no such file or directory",
            ))
        return any(marker in message for marker in (
            "syntaxerror", "indentationerror", "taberror", "invalid syntax",
            "unexpected eof while parsing",
        ))

    @classmethod
    def _normalize_result(cls, compiler: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise CodingExecutionError("OnlineCompiler returned an invalid execution result.", retryable=True)

        provider_status = str(payload.get("status") or "").strip().lower()
        exit_code = payload.get("exit_code")
        signal = payload.get("signal")
        output = payload.get("output")
        error = payload.get("error")
        stdout = output if isinstance(output, str) else ""
        error_text = error if isinstance(error, str) else ""
        runtime_time = payload.get("time")
        memory = payload.get("memory")

        if provider_status == "success" and exit_code == 0:
            status_id, description, compile_output, stderr = 3, "Accepted", None, error_text
        elif provider_status in {"timeout", "timed_out"} or exit_code == 124:
            status_id, description, compile_output, stderr = 5, "Time Limit Exceeded", None, error_text
        elif provider_status == "error" and cls._looks_like_compile_error(compiler, error_text):
            status_id, description, compile_output, stderr = 6, "Compilation Error", error_text, ""
        elif provider_status == "error" or (isinstance(exit_code, int) and exit_code != 0) or signal is not None:
            status_id, description, compile_output, stderr = 11, "Runtime Error (NZEC)", None, error_text
        else:
            raise CodingExecutionError("OnlineCompiler returned an invalid execution result.", retryable=True)

        return {
            "status": {"id": status_id, "description": description},
            "status_id": status_id,
            "stdout": stdout,
            "stderr": stderr,
            "compile_output": compile_output,
            "time": runtime_time,
            "memory": memory,
        }

    async def execute(self, source_code: str, language: str, stdin: str) -> dict[str, Any]:
        """Submit exactly one test case to OnlineCompiler's bounded sync endpoint."""
        compiler = self.validate_language(language)
        base_url = self._base_url()
        headers = self._headers()
        timeout = httpx.Timeout(
            timeout=min(max(float(settings.onlinecompiler_timeout_seconds), 31.0), 40.0),
            connect=5.0,
        )
        try:
            async with self.client_factory(timeout=timeout) as client:
                response = await client.post(
                    f"{base_url}/api/run-code-sync/",
                    headers=headers,
                    json={"compiler": compiler, "code": source_code, "input": stdin},
                )
                response.raise_for_status()
                return self._normalize_result(compiler, response.json())
        except CodingExecutionError:
            raise
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {401, 403}:
                raise CodingExecutionConfigurationError(
                    "OnlineCompiler authentication failed. Check the server-side API key."
                ) from exc
            if exc.response.status_code == 429 or exc.response.status_code >= 500:
                raise CodingExecutionError(
                    "OnlineCompiler is busy or unavailable. Please retry later.", retryable=True
                ) from exc
            raise CodingExecutionError(
                "OnlineCompiler could not process this test right now.", retryable=True
            ) from exc
        except httpx.TimeoutException as exc:
            raise CodingExecutionError(
                "OnlineCompiler timed out while processing this test.", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise CodingExecutionError(
                "OnlineCompiler could not process this test right now.", retryable=True
            ) from exc
        except ValueError as exc:
            raise CodingExecutionError(
                "OnlineCompiler returned an invalid execution result.", retryable=True
            ) from exc


coding_execution_service = CodingExecutionService()
