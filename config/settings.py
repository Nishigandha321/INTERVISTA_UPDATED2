# config/settings.py
# Centralized configuration management

import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings:
    """
    Application settings loaded from environment variables.
    
    All configuration in one place - easy to override for testing/deployment.
    """
    
    # Learning Resources Configuration
    youtube_api_key: str = os.getenv("YOUTUBE_API_KEY", "")

    # LLM Configuration
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4000"))
    
    # Cache Configuration
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "500"))
    
    # RAG Configuration
    rag_enabled: bool = os.getenv("RAG_ENABLED", "true").lower() == "true"
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "faiss")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store.pkl")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    retrieval_score_threshold: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "30.0"))
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format: str = os.getenv("LOG_FORMAT", "json")  # json or text
    
    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")

    # Isolated Coding Round execution provider. Credentials stay server-side.
    onlinecompiler_base_url: str = os.getenv(
        "ONLINECOMPILER_BASE_URL", "https://api.onlinecompiler.io"
    ).strip().rstrip("/")
    onlinecompiler_api_key: str = os.getenv("ONLINECOMPILER_API_KEY", "")
    onlinecompiler_timeout_seconds: float = 35.0
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_debug: bool = os.getenv("API_DEBUG", "false").lower() == "true"
    
    # Feature Flags
    enable_streaming: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    enable_rag: bool = os.getenv("ENABLE_RAG", "true").lower() == "true"
    enable_caching: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    enable_monitoring: bool = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    
    # Timeout Configuration
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    retrieval_timeout_seconds: int = int(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "30"))
    
    # Application Configuration
    app_name: str = os.getenv("APP_NAME", "Interview AI System")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    environment: str = os.getenv("ENVIRONMENT", "development")  # development, staging, production
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration."""
        
        if not cls.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        return True
    
    @classmethod
    def to_dict(cls) -> dict:
        """Return all settings as dictionary."""
        return {
            "groq_api_key": "***" if cls.groq_api_key else None,
            "model_name": cls.model_name,
            "temperature": cls.temperature,
            "max_tokens": cls.max_tokens,
            "cache_enabled": cls.cache_enabled,
            "cache_ttl_seconds": cls.cache_ttl_seconds,
            "cache_max_size": cls.cache_max_size,
            "rag_enabled": cls.rag_enabled,
            "embeddings_model": cls.embeddings_model,
            "vector_store_type": cls.vector_store_type,
            "log_level": cls.log_level,
            "environment": cls.environment,
            "app_name": cls.app_name,
            "app_version": cls.app_version
        }
    
    @classmethod
    def display(cls):
        """Display current settings."""
        import json
        settings_dict = cls.to_dict()
        print(json.dumps(settings_dict, indent=2))


# Global settings instance
settings = Settings()
