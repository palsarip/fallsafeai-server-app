from sqlalchemy.orm import Session
from app.api.models.user import User
from app.api.schemas.user import UserCreate

# Create a new user
def create_user(db: Session, user: UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"  # Example hashing
    db_user = User(name=user.name, email=user.email, hashed_password=fake_hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Get a user by ID
def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

# Get all users
def get_users(db: Session, skip: int = 0, limit: int = 10):
    return db.query(User).offset(skip).limit(limit).all()
