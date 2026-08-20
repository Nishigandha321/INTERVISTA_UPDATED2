# core/llm_service.py
import asyncio
import json
import time
import hashlib
from typing import Optional, Dict, Any
from groq import Groq
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


class LLMService:
    """Centralized LLM service - single point of all LLM interactions."""
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.client = Groq(api_key=self.settings.groq_api_key)
        self.cache = LLMCache(
            ttl_seconds=self.settings.cache_ttl_seconds,
            max_size=self.settings.cache_max_size
        )
        logger.info(f"LLMService initialized with model: {self.settings.model_name}")
    
    async def invoke(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        use_cache: bool = True,
        json_mode: bool = False,
        response_format: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None
    ) -> str:
        """Execute single LLM call."""
        
        if use_cache and self.settings.cache_enabled:
            cached = self.cache.get(prompt)
            if cached:
                return cached
        
        try:
            logger.info(f"Invoking LLM (json_mode={json_mode}, tokens={max_tokens})")
            
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
                "model": self.settings.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                request_args["response_format"] = response_format
            if reasoning_effort:
                request_args["reasoning_effort"] = reasoning_effort

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                **request_args,
            )
            
            message = response.choices[0].message
            content = message.content
            if isinstance(content, list):
                content = "".join(
                    part.get("text", part.get("content", "")) if isinstance(part, dict)
                    else getattr(part, "text", "")
                    for part in content
                )
            if not isinstance(content, str) or not content.strip():
                for field in ("reasoning", "reasoning_content", "output_text", "text"):
                    alternate = getattr(message, field, None)
                    if isinstance(alternate, str) and alternate.strip():
                        content = alternate
                        logger.info("Using alternate Groq message.%s for JSON parsing", field)
                        break
            if not isinstance(content, str) or not content.strip():
                logger.error(
                    "LLM response contained no message content (finish_reason=%s)",
                    getattr(response.choices[0], "finish_reason", None),
                )
                raise ValueError("LLM returned empty message content")
            result = content.strip()
            
            if use_cache and self.settings.cache_enabled and result:
                self.cache.set(prompt, result)
            
            logger.info(f"LLM response (len={len(result)}) received")
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
            logger.info(f"Invoking LLM chat completion (messages={len(messages)}, tokens={max_tokens})")
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.settings.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            
            logger.info(f"LLM chat response (len={len(result)}) received")
            return result
        
        except Exception as e:
            logger.error(f"LLM chat completion failed: {str(e)}")
            raise
    
    async def invoke_with_context(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
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
