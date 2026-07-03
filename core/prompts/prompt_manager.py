# prompts/prompt_manager.py
# Centralized prompt template management with versioning support

from typing import Dict, Any, Optional
from enum import Enum
from utils.logger import Logger

logger = Logger(__name__)


class PromptCategory(Enum):
    """Enum for prompt categories."""
    INTERVIEW = "interview"
    EVALUATION = "evaluation"
    COURSE = "course"
    SUMMARY = "summary"
    PROFILE = "profile"
    RAG_RETRIEVAL = "rag_retrieval"


class PromptTemplate:
    """Single prompt template with versioning."""
    
    def __init__(self, name: str, template: str, category: PromptCategory, version: str = "1.0"):
        self.name = name
        self.template = template
        self.category = category
        self.version = version
    
    def render(self, **variables) -> str:
        """Render template with variables."""
        try:
            return self.template.format(**variables)
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            raise


class PromptManager:
    """
    Centralized management of all prompts.
    
    Features:
    - Template versioning
    - Category organization
    - Easy variable substitution
    - Audit trail of prompt changes
    """
    
    def __init__(self):
        self.prompts: Dict[str, Dict[str, PromptTemplate]] = {}
        self._register_all_prompts()
        logger.info("PromptManager initialized with all templates")
    
    def _register_all_prompts(self):
        """Register all prompt templates."""
        
        # INTERVIEW PROMPTS
        self._register_agentic_interviewer()
        self._register_oa_generate()
        
        # EVALUATION PROMPTS
        self._register_evaluation_content()
        self._register_performance_summary()
        
        # COURSE PROMPTS
        self._register_course_outline()
        self._register_course_module_detail()
        
        # PROFILE PROMPTS
        self._register_resume_skill_profile()
        self._register_resume_jd_skill_gap()
        
        # RAG PROMPTS
        self._register_rag_retrieval()
    
    def _register_agentic_interviewer(self):
        """Agentic interviewer system prompt for resume-aware multi-turn interviews."""
        template = """You are an elite technical recruiter and interviewer.

Analyze the role, designation, resume, learning context, and optional job targeting:
Role: {role}
Designation: {level}
Company (optional): {company_name}
Job Description (optional): {job_description}
Resume context: {resume_text}
Matched resume skills vs JD: {matched_skills}
Missing JD skills (gaps): {missing_skills}
Completed module titles: {completed_modules}
Course topics: {course_topics}
Previous questions asked:
{previous_questions}

Avoid repeating any of the above questions.

Used categories so far:
{used_categories}

Ask a NEW question from a DIFFERENT category if possible.

WHEN Job Description is provided and not "N/A":
- Target company: {company_name}
- Use this question distribution across the full interview (pick the category least used so far):
  * 30% resume-based — probe skills/experience on the resume (category: resume-based)
  * 30% JD-based — test concepts and responsibilities from the job description (category: jd-based)
  * 20% missing-skill-based — focus on JD skills absent or weak on the resume (category: missing-skill)
  * 20% behavioral/company-fit — culture, collaboration, ownership aligned with JD and company (category: behavioral-company-fit)
- Example: resume skill React + JD skill AWS → ask about React experience, AWS concepts, or AWS scenarios as appropriate.
- Behavioral questions should reflect company expectations in the JD.

WHEN Job Description is empty or "N/A":
- Ignore JD-specific distribution and use standard interview behavior.
- Prioritize weak areas, course topics, and role-specific applied scenarios.
- Categories: behavioral|situational|technical|logical|project-specific

TECHNICAL INTERVIEW ROUND (8-10 questions total):
- Mix question types across the full interview — resume, projects, skills, JD, company-specific, coding (discussion format), and technical concepts.
- Avoid repeating categories; vary depth and difficulty.
- For coding questions: ask in discussion format (e.g. "Explain how you would solve Two Sum", "Write pseudocode for Binary Search", "Explain BFS vs DFS"). Do NOT require runnable code or test cases.
- Coding category: use category "coding" for algorithm/DS questions answered via explanation, pseudocode, or code snippets.

Rules:
- NEVER repeat previous questions
- ALWAYS generate a unique, fresh question
- If repeating category, increase difficulty or depth
- Use course topics and completed modules to make questions relevant and practical

Additional instructions: {instruction_override}

Return ONLY JSON:
{{
 "question": "...",
 "category": "resume-based|jd-based|missing-skill|behavioral-company-fit|behavioral|situational|technical|logical|project-specific|coding|company-specific|skill-based|project-based"
}}
"""
        
        self._register(
            "interviewer_system_prompt",
            template,
            PromptCategory.INTERVIEW,
            "3.1"
        )

    def _register_oa_generate(self):
        """Generate placement screening MCQs for Online Assessment round."""
        template = """You are an expert placement test designer creating a generic Online Assessment (OA) screening round.

This assessment should NOT depend heavily on a specific resume, company, or job description.
Focus on general aptitude and CS fundamentals suitable for campus/placement screening.

Generate EXACTLY 10 multiple-choice questions with this distribution:
- 2 Aptitude (quantitative/logical reasoning)
- 1 OOPS (object-oriented programming concepts)
- 1 C++ (C++ language fundamentals)
- 1 SQL (SQL queries and concepts)
- 2 DBMS (database management systems)
- 1 Operating Systems
- 1 Computer Networks
- 1 DSA (data structures & algorithms)

Each question MUST have exactly four options (A, B, C, D) and one correct answer.

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "option_a": "First option",
      "option_b": "Second option",
      "option_c": "Third option",
      "option_d": "Fourth option",
      "correct_answer": "A",
      "explanation": "Brief explanation of why this answer is correct.",
      "topic": "Aptitude"
    }}
  ]
}}

Rules:
- correct_answer must be exactly one of: A, B, C, or D
- topic must be one of: Aptitude, OOPS, C++, SQL, DBMS, Operating Systems, Computer Networks, DSA
- All 10 questions must be unique and non-trivial
- Options must be plausible; only one correct answer per question
- STRICT: Return JSON only. No markdown."""
        
        self._register(
            "oa_generate_prompt",
            template,
            PromptCategory.INTERVIEW,
            "1.0"
        )
    
    def _register_evaluation_content(self):
        """Evaluate interview answer with CKFS metrics for RL integration."""
        template = """You are evaluating an interview answer for a {level} {role} candidate.

IMPORTANT: The answer is from speech-to-text. Ignore spelling/grammar/STT mistakes.
Judge intent and substance. Do not penalize homophones or minor word errors.

Target company (optional): {company_name}
Job Description context (optional): {job_description}

If a Job Description is provided and not "N/A":
- Briefly consider whether the answer addresses JD expectations and required skills.
- Use JD only as additional context — do NOT drastically change scoring weights.

Scoring rules:
- 0 → skipped or empty
- 10-30 → wrong or no substance
- 50-70 → partially correct, weak structure
- 80-90 → solid, relevant, reasonably structured
- 90-100 → excellent depth and structure

Dimension scoring (each 0-100, must reflect THIS answer):
- relevance_score: Does the answer address the question asked?
- explanation_depth_score: Technical/role depth and examples
- star_method_score: Behavioral answers — clear Situation, Task, Action, Result (0 if not behavioral)
- structured_thinking_score: Logical flow, steps, cause-effect, signposting
- problem_solving_score: Approach, trade-offs, solution quality (0 if not a problem-solving question)

Question:
{question}

Candidate Answer:
{answer}

Generate an ideal candidate response in first-person. The ideal answer should be a polished, interview-ready reply with concrete examples, clear structure, and personal ownership. Do NOT include guidance, instructions, evaluation commentary, or interviewer perspective.

Return ONLY valid JSON:
{{
 "score": <number 0-100>,
 "relevance_score": <number 0-100>,
 "explanation_depth_score": <number 0-100>,
 "star_method_score": <number 0-100>,
 "structured_thinking_score": <number 0-100>,
 "problem_solving_score": <number 0-100>,
 "strengths": ["..."],
 "weaknesses": ["..."],
 "ideal_answer": "A concise first-person candidate answer with a strong example, not guidance.",
 "weak_topics": ["topic1", "topic2"],
 "C": <float 0.0-1.0>,
 "K": <float 0.0-1.0>,
 "F": <float 0.0-1.0>,
 "S": <float 0.0-1.0>
}}

STRICT: JSON only. No markdown."""
        
        self._register(
            "evaluate_answer",
            template,
            PromptCategory.EVALUATION,
            "4.0"
        )
    
    def _register_performance_summary(self):
        """Generate performance summary."""
        template = """You are a technical recruiter evaluating interview performance.

Candidate Role: {role}
Overall Score: {score}/100

Top Weak Topics: {weak_topics}

Attempted Questions: {attempted}/5

Write a brief professional summary (3-4 sentences) recommending:
1. Key strength areas
2. Priority improvement areas

Be concise and actionable.

STRICT: Return ONLY the summary text. No JSON, no markdown, no extra text."""
        
        self._register(
            "performance_summary",
            template,
            PromptCategory.SUMMARY,
            "1.0"
        )
    
    def _register_course_outline(self):
        """Generate course outline."""
        template = """You are a senior curriculum designer.

Role/skill: {skill}
Learner designation: {level}
Duration: {duration_hours} hours
REQUIRED curriculum difficulty: {target_difficulty}
Bandit learning path: {bandit_action}
Suggested course title (use exactly or very close): {title_hint}

{strict_requirements}

Rules:
- Module depth MUST match REQUIRED curriculum difficulty (easy=foundational, hard=advanced/expert).
- Do NOT produce a beginner course when difficulty is hard/advanced.
- Do NOT produce an expert course when difficulty is easy.
- First 2 modules must cover the mandated weak topics.
- course_title must reflect difficulty and topics (not a generic name).

Return JSON:

{{
 "course_title": "...",
 "description": "...",
 "learning_objectives": ["..."],
 "modules": [
  {{
   "module_title": "...",
   "duration_minutes": 60,
   "topics": ["..."]
  }}
 ],
 "assessments": ["..."]
}}"""
        
        self._register(
            "course_outline",
            template,
            PromptCategory.COURSE,
            "2.0"
        )
    
    def _register_course_module_detail(self):
        """Generate detailed course module with strict JSON output."""
        template = """You are a technical instructor creating detailed learning content.

Create comprehensive content for:
Skill: {skill}
Module: {module}
Level: {level}
Final module: {is_final}
Previous modules: {previous_modules}

Audience level guidance:
- Intern: skip beginner-level basics and start at intermediate applied concepts.
- Junior: deliver intermediate content plus applied problems and real examples.
- Senior: provide advanced concepts, system-level thinking, and architectural tradeoffs.

STRICT CONTENT REQUIREMENTS:
1. Depth over definition: Avoid generic definitions and beginner-level explanations.
   - Provide real-world explanation, practical use cases, edge cases, and best practices.
   - Include at least one meaningful code example that solves a real problem.
   - Include one real-world scenario or case study.

2. Practical Learning Depth: The 'content_markdown' must be at least 5-6 paragraphs with detailed technical explanations.
   - Use clear headings, step-by-step reasoning, and concrete examples.
   - Avoid shallow descriptions and list-style content.

3. Practice Links: Add a mandatory field "practice_links".
   - It must be a list of 2-4 valid resources.
   - Each item must include "title" and "url".
   - URLs must start with http.
   - Prefer real resources from LeetCode, GeeksforGeeks, HackerRank, or other trusted technical learning sites.

4. Active Recall Quiz: Ensure quiz questions focus on application and problem solving.
5. Quiz must have EXACTLY 3 multiple-choice questions.
6. Each question has 4 options and 1 correct answer.
7. Answer field contains the LETTER of the correct option (A, B, C, or D).

Return ONLY valid JSON (no markdown wrapper, no extra text):

{{
 "module_title": "Actual module title here",
 "content_markdown": "Write full markdown explanation with headings, paragraphs and code block",
 "quiz": [
  {{
   "question": "Application-focused question here",
   "options": ["Option A", "Option B", "Option C", "Option D"],
   "answer": "A"
  }},
  {{
   "question": "Application-focused question here",
   "options": ["Option A", "Option B", "Option C", "Option D"],
   "answer": "B"
  }},
  {{
   "question": "Application-focused question here",
   "options": ["Option A", "Option B", "Option C", "Option D"],
   "answer": "C"
  }}
 ],
 "practice_links": [
  {{
   "title": "LeetCode problem title",
   "url": "https://leetcode.com/problems/example"
  }}
 ]
}}

CRITICAL: Return ONLY the JSON object. No markdown code blocks. No explanations."""
        
        self._register(
            "course_module_detail",
            template,
            PromptCategory.COURSE,
            "2.0"
        )
    
    def _register_resume_skill_profile(self):
        """Extract skills from resume."""
        template = """You are analyzing a professional resume.

Target Role: {role}
Experience Level: {experience_level}

Resume (first 800 chars):
{resume_text}

Extract and analyze. Return ONLY JSON:

{{
 "identified_skills": ["..."],
 "skill_gaps": ["..."],
 "missing_for_role": ["..."],
 "strength_percentage": <0-100>,
 "improvement_suggestions": "short paragraph"
}}"""
        
        self._register(
            "resume_skill_profile",
            template,
            PromptCategory.PROFILE,
            "2.0"
        )

    def _register_resume_jd_skill_gap(self):
        """Compare resume skills against a job description for interview targeting."""
        template = """You are an ATS-style resume screener.

Compare the candidate resume against the job description.
Identify concrete skills, tools, and technologies (not vague traits).

Resume:
{resume_text}

Job Description:
{job_description}

Return ONLY valid JSON:
{{
 "matched_skills": ["skills present in both resume and JD"],
 "missing_skills": ["important JD skills weak or absent on resume"],
 "ats_score": <integer 0-100 approximate match>
}}

Rules:
- ats_score is approximate based on skill/requirement overlap.
- Keep lists concise (max 10 items each).
- Use specific skill names (e.g. "AWS", "React", "SQL").
- JSON only. No markdown."""
        
        self._register(
            "resume_jd_skill_gap",
            template,
            PromptCategory.PROFILE,
            "1.0"
        )
    
    def _register_rag_retrieval(self):
        """RAG retrieval relevance check."""
        template = """Given the following documents, determine if they contain relevant information to answer the user's question.

User Question:
{question}

Available Documents:
{documents}

Return JSON:

{{
 "relevant": <true/false>,
 "relevance_score": <0-100>,
 "relevant_sections": ["section1", "section2"],
 "reasoning": "brief explanation"
}}"""
        
        self._register(
            "rag_retrieval",
            template,
            PromptCategory.RAG_RETRIEVAL,
            "1.0"
        )
    
    def _register(self, name: str, template: str, category: PromptCategory, version: str):
        """Register a new prompt template."""
        if category.value not in self.prompts:
            self.prompts[category.value] = {}
        
        self.prompts[category.value][name] = PromptTemplate(
            name=name,
            template=template,
            category=category,
            version=version
        )
        logger.debug(f"Registered prompt: {category.value}/{name} v{version}")
    
    def get_prompt(self, name: str, category: PromptCategory = None, **variables) -> str:
        """
        Get and render prompt template.
        
        Args:
            name: Prompt name
            category: Prompt category (optional, for disambiguation)
            **variables: Template variables
        
        Returns:
            Rendered prompt string
        """
        
        # Search across all categories if not specified
        for cat_name, cat_prompts in self.prompts.items():
            if name in cat_prompts:
                prompt = cat_prompts[name]
                rendered = prompt.render(**variables)
                logger.debug(f"Retrieved prompt: {name} (v{prompt.version})")
                return rendered
        
        logger.error(f"Prompt not found: {name}")
        raise ValueError(f"Prompt template '{name}' not found")
    
    def list_prompts(self, category: PromptCategory = None) -> Dict[str, str]:
        """List all registered prompts with versions."""
        result = {}
        
        for cat_name, cat_prompts in self.prompts.items():
            if category and cat_name != category.value:
                continue
            
            for name, prompt in cat_prompts.items():
                result[f"{cat_name}/{name}"] = prompt.version
        
        return result
    
    def get_template_preview(self, name: str, max_chars: int = 200) -> str:
        """Get preview of template (first N chars)."""
        for cat_prompts in self.prompts.values():
            if name in cat_prompts:
                template = cat_prompts[name].template
                return template[:max_chars] + "..." if len(template) > max_chars else template
        
        return "Template not found"
