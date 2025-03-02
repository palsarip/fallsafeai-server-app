from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.api.core.config import settings

# Use the encoded URL
engine = create_engine(settings.db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Initialize DB
def init_db():
    from app.api.models import user
    Base.metadata.create_all(bind=engine)

# Dependency for getting a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
