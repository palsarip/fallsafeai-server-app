from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    class Config:
        env_file = ".env"

    @property
    def db_url(self) -> str:
        # Parse and rebuild the URL with proper encoding
        user = "fallsafeai"
        password = quote_plus("P@ssw0rd")  # URL encode the password
        host = "localhost"
        port = "5432"
        db = "fallsafeai"
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

settings = Settings()
