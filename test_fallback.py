import asyncio
from core.llm.llm_service import LLMService
from config.settings import Settings


async def main():
    settings = Settings()

    # Force Groq to fail
    settings.model_name = "invalid-groq-model-for-testing"

    llm = LLMService(settings)

    result = await llm.invoke(
        "Reply with exactly: GEMINI FALLBACK WORKED",
        use_cache=False
    )

    print("\nFINAL RESULT:")
    print(result)


asyncio.run(main())