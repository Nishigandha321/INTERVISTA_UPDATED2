# core/llm_service.py
import asyncio
import json
import time
import hashlib
import re
from typing import Optional, Dict, Any
from groq import Groq
from openai import OpenAI
from config.settings import Settings
import logging


logger = logging.getLogger(__name__)


class LLMCache:
    """Simple TTL-based cache for LLM responses."""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 500):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
    
    def _make_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        key = self._make_key(prompt)
        entry = self.cache.get(key)
        
        if entry and (time.time() - entry["ts"]) < self.ttl_seconds:
            logger.debug(f"Cache HIT for key: {key[:16]}...")
            return entry["value"]
        
        self.cache.pop(key, None)
        return None
    
    def set(self, prompt: str, value: str):
        if len(self.cache) > self.max_size:
            oldest = min(self.cache, key=lambda k: self.cache[k]["ts"])
            self.cache.pop(oldest, None)
        
        key = self._make_key(prompt)
        self.cache[key] = {"value": value, "ts": time.time()}
        logger.debug(f"Cache SET for key: {key[:16]}...")
    
    def clear(self):
        self.cache.clear()


class LLMRateLimitBackoff(RuntimeError):
    """Raised locally while the provider's last rate-limit cooldown is active."""


class LLMService:
    """Provider-specific LLM client; callers choose the provider per use case."""

    def __init__(
        self,
        settings: Settings = None,
        provider: str = "openrouter",
        model_name: Optional[str] = None,
        purpose: str = "LLM",
    ):
        self.settings = settings or Settings()
        self.provider = provider.strip().lower()
        if self.provider not in {"groq", "openrouter"}:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        self.model_name = model_name or self.settings.model_name
        self.provider_name = "Groq" if self.provider == "groq" else "OpenRouter"
        if self.provider == "groq":
            self.client = (
                Groq(api_key=self.settings.groq_api_key)
                if self.settings.groq_api_key else None
            )
        else:
            self.client = (
                OpenAI(
                    api_key=self.settings.openrouter_api_key,
                    base_url="https://openrouter.ai/api/v1",
                )
                if self.settings.openrouter_api_key else None
            )
        self.cache = LLMCache(
            ttl_seconds=self.settings.cache_ttl_seconds,
            max_size=self.settings.cache_max_size
        )
        self._rate_limit_until = 0.0
        self._rate_limit_message = ""
        logger.info("%s → %s / %s", purpose, self.provider_name, self.model_name)

    @staticmethod
    def _response_content(response, provider: str) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            error = getattr(response, "error", None)
            error_message = (
                error.get("message") if isinstance(error, dict)
                else getattr(error, "message", None)
            )
            detail = f": {error_message}" if error_message else ""
            raise RuntimeError(f"{provider} returned no completion choices{detail}")

        message = choices[0].message
        content = message.content
        if isinstance(content, list):
            content = "".join(
                part.get("text", part.get("content", "")) if isinstance(part, dict)
                else getattr(part, "text", "")
                for part in content
            )
        if not isinstance(content, str) or not content.strip():
            for field in ("output_text", "text"):
                alternate = getattr(message, field, None)
                if isinstance(alternate, str) and alternate.strip():
                    content = alternate
                    break
        if not isinstance(content, str) or not content.strip():
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            usage = getattr(response, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            reasoning_tokens = getattr(
                getattr(usage, "completion_tokens_details", None),
                "reasoning_tokens",
                None,
            )
            details = [f"finish_reason={finish_reason or 'unknown'}"]
            if completion_tokens is not None:
                details.append(f"completion_tokens={completion_tokens}")
            if reasoning_tokens is not None:
                details.append(f"reasoning_tokens={reasoning_tokens}")
            raise ValueError(
                f"{provider} returned empty message content ({', '.join(details)})"
            )
        return content.strip()

    async def _create_completion(self, request_args: dict) -> tuple[str, str]:
        """Execute through this service's configured LLM provider only."""
        if self.client is None:
            key_name = "GROQ_API_KEY" if self.provider == "groq" else "OPENROUTER_API_KEY"
            raise RuntimeError(f"{key_name} environment variable not set")
        if time.monotonic() < self._rate_limit_until:
            raise LLMRateLimitBackoff(
                self._rate_limit_message or f"{self.provider_name} rate limit cooldown is active"
            )

        # Keep each provider's generation budget separate from its prompt size.
        minimum_generation_tokens = int(
            request_args.pop("_minimum_generation_tokens", 1000)
        )
        requested_generation_tokens = (
            request_args.get("max_completion_tokens") or request_args.get("max_tokens") or 0
        )
        generation_budget = max(int(requested_generation_tokens), minimum_generation_tokens)
        if self.provider == "groq":
            # Groq's GPT-OSS endpoint uses max_completion_tokens.
            # Honor the shared application budget instead of inheriting a small
            # caller-specific question limit that could crowd out reasoning.
            generation_budget = max(generation_budget, int(self.settings.max_tokens or 0))
            request_args.pop("max_tokens", None)
            request_args["max_completion_tokens"] = generation_budget
        else:
            request_args["max_tokens"] = generation_budget
            # Keep Nemotron's reasoning configuration local to OpenRouter.
            extra_body = dict(request_args.get("extra_body") or {})
            reasoning = dict(extra_body.get("reasoning") or {})
            reasoning.setdefault(
                "effort", request_args.get("reasoning_effort") or "high"
            )
            extra_body["reasoning"] = reasoning
            request_args["extra_body"] = extra_body

        logger.info(
            "%s request model=%s generation_budget=%s",
            self.provider_name, request_args.get("model"), generation_budget,
        )

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create, **request_args
            )
            return self._response_content(response, self.provider_name), self.provider_name
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            error_text = str(exc).casefold()
            if status_code == 429 or any(
                marker in error_text
                for marker in ("rate_limit_exceeded", "tokens per minute", "tokens per day")
            ):
                self._record_rate_limit(exc)
            logger.error("%s LLM request failed: %s", self.provider_name, exc)
            raise

    def _record_rate_limit(self, exc: Exception) -> None:
        message = str(exc)
        seconds = 5.0
        minute_match = re.search(r"try again in\s+(\d+)m([\d.]+)s", message, re.IGNORECASE)
        second_match = re.search(r"try again in\s+([\d.]+)s", message, re.IGNORECASE)
        if minute_match:
            seconds = int(minute_match.group(1)) * 60 + float(minute_match.group(2))
        elif second_match:
            seconds = float(second_match.group(1))
        self._rate_limit_until = max(self._rate_limit_until, time.monotonic() + max(seconds, 1.0))
        self._rate_limit_message = message
    
    async def invoke(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 4000,
        use_cache: bool = True,
        json_mode: bool = False,
        response_format: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        request_timeout_seconds: Optional[float] = None,
        min_generation_tokens: Optional[int] = None,
    ) -> str:
        """Execute single LLM call."""

        if use_cache and self.settings.cache_enabled:
            cached = self.cache.get(prompt)
            if cached:
                return cached
        
        try:
            logger.info(
                "Invoking %s LLM (json_mode=%s, tokens=%s)",
                self.provider_name, json_mode, max_tokens,
            )
            
            messages = []
            
            if json_mode:
                messages.append({
                    "role": "system",
                    "content": "Return ONLY valid JSON. No markdown, no explanation."
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            request_args = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                request_args["max_tokens"] = max_tokens
            if response_format:
                request_args["response_format"] = response_format
            if reasoning_effort:
                request_args["reasoning_effort"] = reasoning_effort
            if request_timeout_seconds is not None:
                request_args["timeout"] = request_timeout_seconds
            if min_generation_tokens is not None:
                request_args["_minimum_generation_tokens"] = min_generation_tokens

            result, provider = await self._create_completion(
                request_args
            )
            
            if use_cache and self.settings.cache_enabled and result:
                self.cache.set(prompt, result)
            
            logger.info("%s response (len=%s) received", provider, len(result))
            return result
        
        except Exception as e:
            logger.error(f"LLM invocation failed: {str(e)}")
            raise
    
    async def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 1000
    ) -> str:
        """Execute multi-turn conversation LLM call."""
        
        try:
            logger.info(
                "Invoking %s LLM chat completion (messages=%s, tokens=%s)",
                self.provider_name, len(messages), max_tokens,
            )
            
            result, provider = await self._create_completion({
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            })
            
            logger.info("%s chat response (len=%s) received", provider, len(result))
            return result
        
        except Exception as e:
            logger.error(f"LLM chat completion failed: {str(e)}")
            raise
    
    async def invoke_with_context(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = 4000,
        json_mode: bool = False
    ) -> str:
        """Execute LLM call with additional context."""
        augmented_prompt = f"""CONTEXT:
{context}

QUESTION:
{prompt}"""
        
        return await self.invoke(
            prompt=augmented_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=False,
            json_mode=json_mode
        )
    
    def clear_cache(self):
        """Clear response cache."""
        self.cache.clear()
        logger.info("LLM cache cleared")
