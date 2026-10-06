# refactored_main.py
# Main entry point - complete refactored system

import asyncio
import json
import sys
from typing import Dict, Any
from config.settings import Settings, settings
from core.llm.llm_service import LLMService
from core.prompts.prompt_manager import PromptManager, PromptCategory
from services.rag.retriever import get_retriever
from core.agents.agent import Agent
from core.chains.base_chain import (
    InterviewQuestionChain,
    EvaluationChain,
    SummaryChain
)
from evaluation.evaluator import ChainEvaluator
from utils.logger import Logger

logger = Logger(__name__)


class InterviewSystem:
    """Complete interview AI system - high-level orchestrator."""
    
    def __init__(self):
        """Initialize the complete system."""
        
        # Validate configuration
        try:
            Settings.validate(provider="groq")
        except ValueError as e:
            logger.error(f"Configuration error: {str(e)}")
            raise
        
        # Initialize components
        self.llm_service = LLMService(
            settings,
            provider="groq",
            model_name=settings.interview_model_name,
            purpose="Interview LLM",
        )
        self.prompt_manager = PromptManager()
        self.retriever = get_retriever()
        self.agent = Agent(
            llm_service=self.llm_service,
            prompt_manager=self.prompt_manager,
            retriever=self.retriever,
            name="InterviewCoordinator"
        )
        
        logger.info("InterviewSystem initialized successfully")
        self.display_system_info()
    
    def display_system_info(self):
        """Display system configuration."""
        info = self.agent.get_agent_info()
        logger.info(f"System Info: {json.dumps(info, indent=2)}")
    
    async def generate_interview_questions(
        self,
        role: str,
        level: str,
        count: int = 5,
        resume_text: str = ""
    ) -> Dict[str, Any]:
        """
        Generate interview questions.
        
        Args:
            role: Job role (e.g., "Software Engineer")
            level: Experience level (e.g., "Junior", "Senior")
            count: Number of questions
            resume_text: Optional resume text for personalization
        
        Returns:
            Response with generated questions
        """
        
        logger.info(f"Generating {count} questions for {level} {role}")
        
        input_data = {
            "action": "generate",
            "intent": "generate_questions",
            "role": role,
            "level": level,
            "count": count,
            "resume_text": resume_text[:600]  # Truncate for performance
        }
        
        result = await self.agent.run(input_data)
        return result
    
    async def evaluate_interview_answer(
        self,
        role: str,
        level: str,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate interview answer.
        
        Args:
            role: Job role
            level: Experience level
            question: Interview question
            answer: Candidate's answer
        
        Returns:
            Evaluation result with score and feedback
        """
        
        logger.info(f"Evaluating answer for {role} ({level})")
        
        input_data = {
            "action": "evaluate",
            "intent": "evaluate_answer",
            "role": role,
            "level": level,
            "question": question[:120],  # Truncate
            "answer": answer[:1500]  # Truncate
        }
        
        result = await self.agent.run(input_data)
        return result
    
    async def generate_performance_summary(
        self,
        role: str,
        score: int,
        weak_topics: list,
        attempted: int
    ) -> Dict[str, Any]:
        """
        Generate performance summary.
        
        Args:
            role: Job role
            score: Overall score (0-100)
            weak_topics: List of weak areas
            attempted: Number of questions attempted
        
        Returns:
            Summary text
        """
        
        logger.info(f"Generating summary for {role} (score: {score})")
        
        input_data = {
            "action": "summary",
            "intent": "generate_summary",
            "role": role,
            "score": score,
            "weak_topics": weak_topics,
            "attempted": attempted
        }
        
        result = await self.agent.run(input_data)
        return result
    
    async def retrieve_documents(
        self,
        query: str,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents from RAG system.
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            Retrieved documents with scores
        """
        
        logger.info(f"Retrieving documents for: '{query}'")
        
        results = await self.retriever.retrieve(query, k)
        
        return {
            "query": query,
            "results": [r.to_dict() for r in results],
            "count": len(results)
        }
    
    def add_rag_documents(self, documents: list) -> Dict[str, Any]:
        """
        Add documents to RAG system.
        
        Args:
            documents: List of {"content": ..., "metadata": ...}
        
        Returns:
            Status and doc_ids
        """
        
        doc_ids = self.retriever.add_documents(documents)
        logger.info(f"Added {len(doc_ids)} documents to RAG system")
        
        return {
            "status": "success",
            "documents_added": len(doc_ids),
            "doc_ids": doc_ids
        }
    
    async def run_evaluation_tests(self) -> Dict[str, Any]:
        """
        Run evaluation tests on chains.
        
        Returns:
            Evaluation report
        """
        
        logger.info("Running evaluation tests...")
        
        # Test case 1: Question Generation
        evaluator = ChainEvaluator(InterviewQuestionChain(
            self.llm_service,
            self.prompt_manager,
            self.retriever
        ))
        
        await evaluator.evaluate_testcase(
            "question_generation_basic",
            {
                "role": "Software Engineer",
                "level": "Junior",
                "count": 3
            }
        )
        
        # Test case 2: Answer Evaluation
        evaluator2 = ChainEvaluator(EvaluationChain(
            self.llm_service,
            self.prompt_manager,
            self.retriever
        ))
        
        await evaluator2.evaluate_testcase(
            "answer_evaluation_basic",
            {
                "role": "Software Engineer",
                "level": "Junior",
                "question": "What is a REST API?",
                "answer": "A REST API is an interface that uses HTTP requests to perform CRUD operations."
            }
        )
        
        return {
            "status": "complete",
            "test_results": [evaluator.get_summary(), evaluator2.get_summary()]
        }


async def main():
    """Main CLI interface."""
    
    try:
        # Initialize system
        system = InterviewSystem()
        
        # Interactive menu
        while True:
            print("\n" + "="*80)
            print("REFACTORED INTERVIEW AI SYSTEM")
            print("="*80)
            print("1. Generate Interview Questions")
            print("2. Evaluate Interview Answer")
            print("3. Generate Performance Summary")
            print("4. Retrieve Documents (RAG)")
            print("5. Run Evaluation Tests")
            print("6. Display System Info")
            print("7. Exit")
            print("="*80)
            
            choice = input("Select option (1-7): ").strip()
            
            if choice == "1":
                role = input("Job role: ").strip() or "Software Engineer"
                level = input("Experience level (Junior/Mid/Senior): ").strip() or "Junior"
                count = int(input("Number of questions (1-10): ").strip() or "5")
                
                result = await system.generate_interview_questions(role, level, count)
                print("\n" + json.dumps(result, indent=2))
            
            elif choice == "2":
                role = input("Job role: ").strip() or "Software Engineer"
                level = input("Experience level: ").strip() or "Junior"
                question = input("Question: ").strip()
                answer = input("Answer: ").strip()
                
                if question and answer:
                    result = await system.evaluate_interview_answer(role, level, question, answer)
                    print("\n" + json.dumps(result, indent=2))
            
            elif choice == "3":
                role = input("Job role: ").strip()
                try:
                    score = int(input("Overall score (0-100): ").strip())
                    weak_topics_input = input("Weak topics (comma-separated): ").strip()
                    weak_topics = [t.strip() for t in weak_topics_input.split(",")]
                    attempted = int(input("Questions attempted: ").strip())
                    
                    result = await system.generate_performance_summary(role, score, weak_topics, attempted)
                    print("\n" + json.dumps(result, indent=2))
                except ValueError:
                    print("Invalid input")
            
            elif choice == "4":
                query = input("Search query: ").strip()
                if query:
                    result = await system.retrieve_documents(query)
                    print("\n" + json.dumps(result, indent=2))
            
            elif choice == "5":
                print("Running tests...")
                result = await system.run_evaluation_tests()
                print("\n" + json.dumps(result, indent=2))
            
            elif choice == "6":
                system.display_system_info()
            
            elif choice == "7":
                print("Exiting...")
                break
            
            else:
                print("Invalid option")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
