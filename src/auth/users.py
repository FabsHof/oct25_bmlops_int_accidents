from typing import Optional
from .schemas import UserInDB


fake_users_db = {
    "alice": {
        "username": "alice",
        "hashed_password": "$2b$12$ud/UQOrRn5ZLMR3i/Lng7e.9d4vqnWtXcqMr2XK4fGDoUgky.ACie",
        # the hashed password comes from the password "secret". which you need to use in the login form
        "disabled": False,
    }
}


def get_user(username: str) -> Optional[UserInDB]:
    user = fake_users_db.get(username)
    if user:
        return UserInDB(**user)
    return None


