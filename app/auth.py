import os
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)

def admin_required(key: str = Security(api_key_header)):
    admin_key = os.getenv("ADMIN_API_KEY", "")
    if not admin_key or key != admin_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
