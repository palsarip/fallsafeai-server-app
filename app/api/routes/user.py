from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.dependencies.db import get_db
from app.api.models.user import User
from app.api.schemas.user import UserCreate, UserResponse

router = APIRouter()

@router.post("/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    new_user = User(username=user.username, email=user.email, hashed_password=user.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
