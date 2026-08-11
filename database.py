from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("database")


def _build_engine(database_url: str):
    if not database_url:
        database_url = "sqlite:///./jobprepmate.db"

    try:
        if database_url.startswith("sqlite"):
            return create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                pool_pre_ping=True,
            )

        return create_engine(database_url, pool_pre_ping=True)
    except Exception as exc:
        if database_url.startswith("sqlite"):
            raise

        logger.warning(
            "Failed to initialize database engine for %s (%s); falling back to SQLite at jobprepmate.db",
            database_url,
            exc,
        )
        return create_engine(
            "sqlite:///./jobprepmate.db",
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )


DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or "sqlite:///./jobprepmate.db"
engine = _build_engine(DATABASE_URL)
if str(engine.url).startswith("sqlite") and not DATABASE_URL.startswith("sqlite"):
    DATABASE_URL = "sqlite:///./jobprepmate.db"

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
Base = declarative_base()


def _fallback_to_sqlite():
    global DATABASE_URL, engine, SessionLocal

    if DATABASE_URL.startswith("sqlite"):
        return engine

    sqlite_url = "sqlite:///./jobprepmate.db"
    logger.warning("Database initialization failed; falling back to SQLite at %s", sqlite_url)
    DATABASE_URL = sqlite_url
    engine = _build_engine(DATABASE_URL)
    SessionLocal.configure(bind=engine)
    return engine


def initialize_database():
    global engine

    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        if str(engine.url).startswith("sqlite"):
            raise
        logger.warning("Database schema initialization failed (%s); falling back to SQLite", exc)
        engine = _fallback_to_sqlite()
        Base.metadata.create_all(bind=engine)

    return engine