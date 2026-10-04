# chains/base_chain.py
# Base chain implementation - pipeline orchestration


import json
import re
import time
from typing import Optional, Dict, Any, List, Callable
from core.llm.llm_service import LLMService
from core.prompts.prompt_manager import PromptManager, PromptCategory
from services.rag.retriever import Retriever, get_retriever
from utils.logger import Logger

logger = Logger(__name__)


class ChainResult:
    """Result from chain execution."""
    
    def __init__(
        self,
        output: str,
        status: str = "success",
        metadata: Dict[str, Any] = None,
        intermediate_steps: List[Dict] = None
    ):
        self.output = output
        self.status = status  # "success", "error", "partial"
        self.metadata = metadata or {}
        self.intermediate_steps = intermediate_steps or []
    
    def to_dict(self) -> Dict:
        return {
            "output": self.output,
            "status": self.status,
            "metadata": self.metadata,
            "intermediate_steps": self.intermediate_steps
        }


class BaseChain:
    """
    Base chain class for composable AI pipelines.
    
    Typical flow:
    1. Validate input
    2. Optionally retrieve context (RAG)
    3. Format prompt
    4. Call LLM
    5. Parse output
    6. Return result
    """
    
    def __init__(
        self,
        llm_service: LLMService = None,
        prompt_manager: PromptManager = None,
        retriever: Retriever = None,
        name: str = "BaseChain"
    ):
        self.llm_service = llm_service or LLMService()
        self.prompt_manager = prompt_manager or PromptManager()
        self.retriever = retriever or get_retriever()
        self.name = name
        self.intermediate_steps: List[Dict] = []
        logger.info(f"Chain initialized: {name}")
    
    async def invoke(
        self,
        input_data: Dict[str, Any],
        use_rag: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        json_mode: bool = False,
        use_cache: bool = True,
        response_format: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None
    ) -> ChainResult:
        """
        Execute chain.
        
        Args:
            input_data: Input data dict (will be passed to prompt template)
            use_rag: Whether to retrieve context first
            temperature: LLM sampling temperature
            max_tokens: Max tokens in response
            json_mode: Force JSON output
            use_cache: Cache LLM responses
        
        Returns:
            ChainResult
        """
        
        try:
            self.intermediate_steps = []
            
            # Step 2: Optionally retrieve context
            context = ""
            if use_rag:
                context = await self._retrieve_context(input_data)
            
            # Step 3: Format prompt
            prompt = self._format_prompt(input_data, context)
            
            # Step 4: Call LLM
            invoke_args = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "use_cache": use_cache,
            }
            if response_format is not None:
                invoke_args["response_format"] = response_format
            if reasoning_effort is not None:
                invoke_args["reasoning_effort"] = reasoning_effort
            output = await self.llm_service.invoke(
                **invoke_args,
            )
            
            # Step 5: Parse output with retry logic
            parsed = await self._safe_parse_output(output, json_mode)
            
            return ChainResult(
                output=parsed if isinstance(parsed, str) else json.dumps(parsed),
                status="success",
                metadata={"chain": self.name},
                intermediate_steps=self.intermediate_steps
            )
        
        except Exception as e:
            if isinstance(e, ValueError) and "json parsing failed" in str(e).casefold():
                logger.warning(f"Chain returned malformed JSON; caller may retry: {str(e)}")
            else:
                logger.error(f"Chain execution failed: {str(e)}")
            self._log_step("error", f"Chain failed", {"error": str(e)})
            
            return ChainResult(
                output="",
                status="error",
                metadata={"chain": self.name, "error": str(e)},
                intermediate_steps=self.intermediate_steps
            )
    
    def _log_step(self, step_type: str, description: str, data: Dict = None):
        """Log intermediate step."""
        self.intermediate_steps.append({
            "type": step_type,
            "description": description,
            "data": data or {}
        })
    
    async def _safe_parse_output(self, output: str, json_mode: bool = False) -> Any:
        """
        Safely parse LLM output with retry on failure.
        
        For JSON mode: tries multiple strategies, retries LLM once if needed.
        """
        
        if not json_mode:
            return output
        
        # Clean invalid control characters first
        cleaned_output = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', output)
        
        # Try basic JSON parsing first
        try:
            return json.loads(cleaned_output.strip())
        except json.JSONDecodeError:
            pass
        
        # Try extraction strategies
        for strategy in [self._extract_json_strict, self._extract_json_flexible]:
            try:
                result = strategy(cleaned_output)
                if result:
                    return result
            except Exception:
                pass
        
        # If all strategies fail, raise error (no silent fallback)
        logger.warning(f"Cannot parse JSON from LLM output: {cleaned_output[:200]}")
        raise ValueError("JSON parsing failed on all strategies")
    
    def _extract_json_strict(self, text: str) -> Optional[Dict]:
        """Extract JSON from text strictly."""
        cleaned = text.strip()
        # Remove invalid control characters
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
        # Remove markdown blocks
        if cleaned.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
        
        # Find JSON block
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        
        return None
    
    def _extract_json_flexible(self, text: str) -> Optional[Dict]:
        """Extract JSON from text with relaxed rules."""
        # Remove invalid control characters
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # Try to find any balanced JSON-like structure
        for match in re.finditer(r'\{[^{}]*\}', cleaned):
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None
    
    async def _retrieve_context(self, input_data: Dict) -> str:
        """
        Retrieve context from RAG system.
        
        Override this for custom retrieval logic.
        """
        
        query = input_data.get("query") or input_data.get("input") or ""
        
        if not query:
            return ""
        
        try:
            context = await self.retriever.retrieve_context(query, k=5)
            return context
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {str(e)}")
            return ""
    
    def _format_prompt(self, input_data: Dict, context: str = "") -> str:
        """
        Format prompt template with input data and context.
        
        Override for custom prompt formatting.
        """
        
        prompt_name = input_data.get("prompt_name", "")
        
        if not prompt_name:
            return str(input_data)
        
        try:
            # Get base prompt
            prompt = self.prompt_manager.get_prompt(prompt_name, **input_data)
            
            # Append context if available
            if context:
                prompt = f"""CONTEXT:
{context}

ORIGINAL PROMPT:
{prompt}"""
            
            return prompt
        
        except Exception as e:
            logger.error(f"Prompt formatting failed: {str(e)}")
            return str(input_data)
    
    def _parse_output(self, output: str, json_mode: bool = False) -> Any:
        """
        Parse LLM output.
        
        Override for custom parsing logic.
        """
        
        if not json_mode:
            return output
        
        # Try to extract JSON from output
        try:
            # Remove markdown code blocks if present
            cleaned = output.strip()
            # Remove invalid control characters
            cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
            if cleaned.startswith("```"):
                cleaned = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
                if cleaned:
                    cleaned = cleaned.group(1)
            
            # Find JSON boundaries
            start = cleaned.find("{")
            if start == -1:
                start = cleaned.find("[")
            
            if start == -1:
                return output  # No JSON found
            
            end = cleaned.rfind("}")
            if end == -1:
                end = cleaned.rfind("]")
            
            if end == -1:
                return output
            
            json_str = cleaned[start:end + 1]
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            return {"raw_output": output, "parse_error": str(e)}


class InterviewQuestionChain(BaseChain):
    """Specialized chain for generating interview questions with RAG context."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="InterviewQuestionChain", **kwargs)
        self._interview_context_cache: Dict[tuple[str, str], tuple[float, str]] = {}
    
    async def invoke(self, input_data: Dict[str, Any], **kwargs) -> ChainResult:
        """Generate interview questions with relevant context from RAG."""
        
        # Prepare input
        previous_questions = input_data.get("previous_questions", [])
        used_categories = input_data.get("used_categories", [])
        
        prepared = {
            "prompt_name": "interviewer_system_prompt",
            "role": input_data.get("role", "Software Engineer"),
            "level": input_data.get("level", "Junior"),
            "interview_type": input_data.get("interview_type", "technical"),
            "count": input_data.get("count", 5),
            "resume_text": input_data.get("resume_text", ""),
            "company_name": input_data.get("company_name") or "N/A",
            "job_description": input_data.get("job_description") or "N/A",
            "matched_skills": ", ".join(input_data.get("matched_skills", [])) if isinstance(input_data.get("matched_skills", []), list) else str(input_data.get("matched_skills", "") or "N/A"),
            "missing_skills": ", ".join(input_data.get("missing_skills", [])) if isinstance(input_data.get("missing_skills", []), list) else str(input_data.get("missing_skills", "") or "N/A"),
            "completed_modules": ", ".join(input_data.get("completed_modules", [])) if isinstance(input_data.get("completed_modules", []), list) else str(input_data.get("completed_modules", "")),
            "course_topics": ", ".join(input_data.get("course_topics", [])) if isinstance(input_data.get("course_topics", []), list) else str(input_data.get("course_topics", "")),
            "current_question": input_data.get("current_question", "N/A"),
            "candidate_latest_answer": input_data.get("candidate_latest_answer", "N/A"),
            "answer_analysis": json.dumps(input_data.get("answer_analysis", {}), ensure_ascii=False),
            "recent_conversation": json.dumps(input_data.get("recent_conversation", []), ensure_ascii=False),
            "candidate_state": json.dumps(input_data.get("candidate_state", {}), ensure_ascii=False),
            "previous_questions": "\n".join(previous_questions) if isinstance(previous_questions, list) else str(previous_questions),
            "used_categories": ", ".join(used_categories) if isinstance(used_categories, list) else str(used_categories),
            "instruction_override": input_data.get("instruction_override") or "None",
        }
        
        # Use RAG to retrieve similar interview questions and best practices
        use_rag = kwargs.pop("use_rag", True)
        max_tokens = kwargs.pop("max_tokens", 450)
        
        return await super().invoke(
            prepared,
            use_rag=use_rag,
            json_mode=True,
            response_format={"type": "json_object"},
            use_cache=False,  # Always disable cache for dynamic questions
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def _retrieve_context(self, input_data: Dict) -> str:
        """Retrieve interview questions and rubrics for this role/level."""
        from services.rag.rag_config import RetrievalContext, DocumentCategory
        
        role = input_data.get("role", "")
        level = input_data.get("level", "")
        cache_key = (str(role).casefold().strip(), str(level).casefold().strip())
        cached = self._interview_context_cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 1800:
            return cached[1]
        
        query = f"interview questions for {role} at {level} level"
        
        context_obj = RetrievalContext(
            query=query,
            category=DocumentCategory.INTERVIEW_QUESTIONS,
            role=role.lower() if role else None,
            level=level.lower() if level else None,
            k=3
        )
        
        try:
            from services.rag.rag_pipeline import get_or_create_rag_pipeline
            pipeline = get_or_create_rag_pipeline()
            
            if pipeline.initialized:
                context = await pipeline.retrieve_context(query, context_obj)
                self._interview_context_cache[cache_key] = (time.monotonic(), context or "")
                return context or ""
        except Exception as e:
            logger.debug(f"RAG retrieval skipped: {str(e)}")
        
        return ""


class OnlineAssessmentChain(BaseChain):
    """Generate placement screening MCQs for the Online Assessment round."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="OnlineAssessmentChain", **kwargs)

    async def invoke(self, input_data: Dict[str, Any], **kwargs) -> ChainResult:
        prepared = {
            "prompt_name": "oa_generate_prompt",
            "role": input_data.get("role", "Software Engineer"),
            "level": input_data.get("level", "Junior"),
        }
        prompt = self._format_prompt(prepared)
        self._validate_oa_prompt(prompt)
        kwargs.pop("max_tokens", None)
        kwargs.pop("json_mode", None)
        kwargs.pop("reasoning_effort", None)
        logger.info("OA LLM call started")
        logger.info("OA requested max_tokens=1200")
        return await super().invoke(
            prepared,
            use_rag=False,
            json_mode=True,
            use_cache=False,
            max_tokens=1200,
            reasoning_effort="low",
            **kwargs,
        )

    def _validate_oa_prompt(self, prompt: str) -> None:
        """Reject malformed schema instructions before consuming an LLM call."""
        start = prompt.find("{\n  \"questions\"")
        if start < 0:
            raise ValueError("OA prompt is missing its JSON schema example")
        try:
            example, _ = json.JSONDecoder().raw_decode(prompt[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("OA prompt contains malformed JSON schema instructions") from exc
        if set(example) != {"questions"} or not isinstance(example["questions"], list):
            raise ValueError("OA prompt schema must contain only questions")
        if len(example["questions"]) != 1 or set(example["questions"][0]) != {
            "question", "option_a", "option_b", "option_c", "option_d",
            "correct_answer", "explanation", "topic",
        }:
            raise ValueError("OA prompt question schema is invalid")


class EvaluationChain(BaseChain):
    """Specialized chain for evaluating interview answers with RAG rubrics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="EvaluationChain", **kwargs)
    
    async def invoke(self, input_data: Dict[str, Any], **kwargs) -> ChainResult:
        """Evaluate answer with relevant rubrics and examples from RAG."""
        
        prepared = {
            "prompt_name": "evaluate_answer",
            "role": input_data.get("role", "Software Engineer"),
            "level": input_data.get("level", "Junior"),
            "question": input_data.get("question", ""),
            "answer": input_data.get("answer", ""),
            "company_name": input_data.get("company_name") or "N/A",
            "job_description": input_data.get("job_description") or "N/A",
            "instruction_override": input_data.get("instruction_override") or "None",
        }
        
        # Use RAG to retrieve evaluation rubrics
        use_rag = kwargs.pop("use_rag", True)
        
        # Disable cache for unique evaluations
        kwargs.pop("max_tokens", None)
        return await super().invoke(
            prepared,
            use_rag=use_rag,
            json_mode=True,
            use_cache=False,
            max_tokens=900,
            **kwargs
        )
    
    async def _retrieve_context(self, input_data: Dict) -> str:
        """Retrieve evaluation rubrics and communication guidelines."""
        from services.rag.rag_config import RetrievalContext, DocumentCategory
        
        role = input_data.get("role", "")
        question = input_data.get("question", "")[:100]  # First 100 chars for query
        
        query = f"evaluation rubrics communication best practices for {role}"
        
        context_obj = RetrievalContext(
            query=query,
            category=DocumentCategory.EVALUATION_RUBRICS,
            k=4
        )
        
        try:
            from services.rag.rag_pipeline import get_or_create_rag_pipeline
            pipeline = get_or_create_rag_pipeline()
            
            if pipeline.initialized:
                return await pipeline.retrieve_context(query, context_obj)
        except Exception as e:
            logger.debug(f"RAG retrieval skipped: {str(e)}")
        
        return ""


class RAGChain(BaseChain):
    """Specialized chain with RAG retrieval."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="RAGChain", **kwargs)
    
    async def invoke(self, input_data: Dict[str, Any], **kwargs) -> ChainResult:
        """Execute RAG-augmented chain."""
        
        return await super().invoke(input_data, use_rag=True, **kwargs)


class SummaryChain(BaseChain):
    """Specialized chain for generating summaries with skill framework context."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, name="SummaryChain", **kwargs)
    
    async def invoke(self, input_data: Dict[str, Any], **kwargs) -> ChainResult:
        """Generate performance summary with relevant skill frameworks."""
        
        prepared = {
            "prompt_name": "performance_summary",
            "role": input_data.get("role", ""),
            "score": input_data.get("score", 0),
            "weak_topics": ", ".join(input_data.get("weak_topics", [])),
            "attempted": input_data.get("attempted", 0)
        }
        
        # Use RAG for skill frameworks
        use_rag = kwargs.pop("use_rag", True)
        
        return await super().invoke(
            prepared,
            use_rag=use_rag,
            json_mode=False,
            **kwargs
        )
    
    async def _retrieve_context(self, input_data: Dict) -> str:
        """Retrieve skill frameworks and improvement recommendations."""
        from services.rag.rag_config import RetrievalContext, DocumentCategory
        
        role = input_data.get("role", "")
        weak_topics = input_data.get("weak_topics", [])
        
        query = f"skill framework for {role} focusing on {', '.join(weak_topics[:3])}"
        
        context_obj = RetrievalContext(
            query=query,
            category=DocumentCategory.SKILL_FRAMEWORKS,
            k=3
        )
        
        try:
            from services.rag.rag_pipeline import get_or_create_rag_pipeline
            pipeline = get_or_create_rag_pipeline()
            
            if pipeline.initialized:
                return await pipeline.retrieve_context(query, context_obj)
        except Exception as e:
            logger.debug(f"RAG retrieval skipped: {str(e)}")
        
        return ""
