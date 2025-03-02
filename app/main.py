from fastapi import FastAPI
from app.api.routes import user, auth
from app.api.dependencies.db import init_db

app = FastAPI()

# Initialize the database (Create tables if not exist)
init_db()

# Include routes
app.include_router(user.router, prefix="/users", tags=["users"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])

@app.get("/")
async def root():
    return {"message": "Hello, World!"}
