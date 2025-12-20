from typing import Optional
from .schemas import UserInDB
from .security import verify_password

fake_users_db = {
    "alice": {
        "username": "alice",
        "hashed_password": "$2b$12$KIXQ1j1F8pYQxvFJH3hE5O6ZLZK9Gz7Hh9c6mV1XyN0k7P6f4nJqS",
        "disabled": False,
    }
}


def get_user(username: str) -> Optional[UserInDB]:
    user = fake_users_db.get(username)
    if user:
        return UserInDB(**user)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user
