from fastapi import Depends, Request, HTTPException, status
import jwt
import os

def is_authorized(request: Request):
    """
    Takes authorization token and makes sure it belongs to a dev
    """
    if os.getenv("stage", "False") in ["dev", "test"]:
        auth = request.headers.get("authorization")
        decoded = jwt.decode(
            auth, algorithms=["RS256"], options={"verify_signature": False}
        )

        if "dev" in decoded["cognito:groups"]:
            return True
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )
    else:
        return True
