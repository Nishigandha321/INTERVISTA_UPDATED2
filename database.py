from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

def _build_engine(database_url: str):
    if not database_url:
        database_url = "sqlite:///./jobprepmate.db"

    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )

    # A configured PostgreSQL/Supabase URL is authoritative. Engine creation or
    # connection errors must surface instead of redirecting writes to a second DB.
    return create_engine(database_url, pool_pre_ping=True)


DATABASE_URL = os.getenv("DATABASE_URL", "").strip() or "sqlite:///./jobprepmate.db"
engine = _build_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
Base = declarative_base()


def initialize_database():
    global engine
    # Do not mask a configured primary database failure by creating or using a
    # local SQLite database. Startup should fail visibly in that case.
    Base.metadata.create_all(bind=engine)
    return engine
