from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.api.dependencies.db import Base

class DummyData(Base):
    __tablename__ = "dummydata"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now()) 