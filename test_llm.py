import asyncio
from core.llm.llm_service import LLMService


async def main():
    llm = LLMService()

    result = await llm.invoke(
        "Reply with exactly: GROQ TEST",
        use_cache=False
    )

    print("\nRESULT:")
    print(result)


asyncio.run(main())