
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    ForeignKey,
    DateTime,
    Text,
    Float,
    JSON,
)
from database import Base
from datetime import datetime


# =========================
# USERS
# =========================
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)


class UserProfile(Base):
    """
    Extended user profile data and serialized skill profile.

    This keeps auth credentials (`User`) separate from richer profile info
    like resume, skill strengths/gaps, and improvement suggestions.
    """

    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, index=True)

    # Basic identity
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    email = Column(String, nullable=True)

    # Targeting and resume metadata
    role_applied_for = Column(String, nullable=True)
    current_designation = Column(String, nullable=True)
    company_name = Column(String, nullable=True)
    job_description = Column(Text, nullable=True)
    resume_file_path = Column(String, nullable=True)

    # Aggregated skill data
    extracted_skills = Column(Text, nullable=True)  # JSON array of skills
    skill_strength_percentage = Column(Float, default=0.0)
    skill_gaps = Column(Text, nullable=True)  # JSON array of weak topics
    improvement_suggestions = Column(Text, nullable=True)

    # Full serialized UserSkillProfile (Pydantic) for richer analysis
    profile_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class UserTarget(Base):
    __tablename__ = "user_targets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)

    role = Column(String, nullable=False)
    level = Column(String, nullable=False)  # Intern/Junior/Senior
    updated_at = Column(DateTime, default=datetime.utcnow)


# =========================
# SKILL TRACKING
# =========================
class SkillProgress(Base):
    __tablename__ = "skill_progress"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    skill = Column(String)
    attempts = Column(Integer, default=0)
    weak = Column(Boolean, default=False)


# =========================
# INTERVIEW
# =========================
class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    role = Column(String, nullable=False)
    level = Column(String, nullable=False)

    status = Column(String, default="active")  # active/completed
    created_at = Column(DateTime, default=datetime.utcnow)


class InterviewAttempt(Base):
    __tablename__ = "interview_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # ✅ FIXED

    role = Column(String)
    topic = Column(String)
    difficulty = Column(String)
    answer = Column(Text)       # better than String for long answers
    feedback = Column(Text)     # better than String for long feedback
    timestamp = Column(DateTime, default=datetime.utcnow)


class Interview(Base):
    """
    High-level record of completed interviews per user and role.

    Stores the overall score and the complete report JSON so that
    past interviews can be browsed from the profile page.
    """

    __tablename__ = "interviews"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)

    role = Column(String, nullable=False)
    date = Column(DateTime, default=datetime.utcnow, index=True)
    score = Column(Float, default=0.0)
    report_json = Column(Text, nullable=False)


# =========================
# COURSE BUILDER - SKELETON-FIRST ARCHITECTURE
# =========================
class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    role = Column(String, nullable=False)          # e.g., "Software Engineer", "Data Scientist"
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    level = Column(String, default="beginner")     # beginner/intermediate/advanced
    status = Column(String, default="draft")       # draft/generated
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Module(Base):
    """
    DEPRECATED — legacy skeleton-first modules (pre resource-pipeline courses).

    New courses use CourseResource rows only. Kept for backward compatibility with
    existing DB rows; do not use for new course creation.
    """
    __tablename__ = "modules"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), index=True)

    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    order_index = Column(Integer, default=0)

    # Content generation (on-demand, cached)
    content = Column(Text, nullable=True)          # JSON: markdown content
    quiz = Column(JSON, nullable=True)             # List of quiz questions with answers

    # Progress tracking
    is_unlocked = Column(Boolean, default=False)   # User can access this module
    is_completed = Column(Boolean, default=False)  # User passed the quiz
    is_final = Column(Boolean, default=False)      # Final assessment module

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModuleAttempt(Base):
    """DEPRECATED — legacy module quiz attempts. See Module."""
    __tablename__ = "module_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    module_id = Column(Integer, ForeignKey("modules.id"), index=True)

    score = Column(Integer, default=0)             # Number of correct answers
    total_questions = Column(Integer, default=0)   # Total quiz questions
    answers = Column(JSON, nullable=True)          # User's answers for review
    created_at = Column(DateTime, default=datetime.utcnow)


class Chapter(Base):
    """DEPRECATED — unused legacy course structure."""
    __tablename__ = "chapters"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"))

    title = Column(String, nullable=False)
    order_index = Column(Integer, default=0)


class Unit(Base):
    """DEPRECATED — unused legacy course structure."""
    __tablename__ = "units"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(Integer, ForeignKey("chapters.id"))

    title = Column(String, nullable=False)
    order_index = Column(Integer, default=0)

    content = Column(Text, nullable=True)
    status = Column(String, default="pending")
    estimated_minutes = Column(Integer, default=10)


# =========================
# QUIZ
# =========================
class QuizAttempt(Base):
    """DEPRECATED — unused legacy quiz attempts."""
    __tablename__ = "quiz_attempts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    unit_id = Column(Integer, ForeignKey("units.id"))
    score = Column(Integer, default=0)

    wrong_concepts = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# =========================
# USER SKILL PROFILES
# =========================
class UserSkillProfileRow(Base):
    """Stores the aggregated skill vectors for a user."""

    __tablename__ = "user_skill_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, index=True)

    technical_skills = Column(JSON, nullable=False, default=dict)
    interview_skills = Column(JSON, nullable=False, default=dict)
    communication_skills = Column(JSON, nullable=False, default=dict)

    overall_score = Column(Float, default=0.0)
    interview_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# =========================
# REINFORCEMENT LEARNING
# =========================
class QTable(Base):
    __tablename__ = "q_table"

    id = Column(Integer, primary_key=True, index=True)
    state_id = Column(String, index=True, nullable=False)
    action_id = Column(String, index=True, nullable=False)
    q_value = Column(Float, default=0.0)
    visit_count = Column(Integer, default=0)


class UserState(Base):
    __tablename__ = "user_states"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    state_id = Column(String, nullable=False)
    weak_topics = Column(JSON, nullable=True)
    avg_score = Column(Float, default=0.0)
    last_score = Column(Float, default=0.0)
    current_proficiency = Column(Integer, default=0)
    session_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# =========================
# LEARNING RESOURCES
# =========================
class CourseResource(Base):
    """
    Persists recommended learning resources linked to a course.

    Each row maps a single external resource (YouTube video, GFG article, etc.)
    to the course and the weak concept it addresses, along with the semantic
    ranking score computed at retrieval time.
    """

    __tablename__ = "course_resources"

    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), index=True, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=True)

    # Which weak topic this resource addresses
    concept = Column(String, nullable=False)

    # Resource identity
    resource_id = Column(String, nullable=False)        # source_hash format (e.g. "youtube_abc123")
    title = Column(String, nullable=False)
    url = Column(String, nullable=False)
    source = Column(String, nullable=False)             # youtube | geeksforgeeks | coursera | etc.

    # Ranking
    rank_score = Column(Float, default=0.0)             # Semantic ranking score (0.0–1.0)
    priority_order = Column(Integer, default=0)         # Weak-concept priority (0 = highest)

    # LLM-generated explanation of why this resource was recommended (1–2 sentences)
    explanation = Column(Text, nullable=True)

    # Progress tracking
    opened_at = Column(DateTime, nullable=True)
    marked_complete = Column(Boolean, default=False)
    completed_at = Column(DateTime, nullable=True)

    # Full resource metadata (LearningResource.to_dict())
    resource_metadata = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

