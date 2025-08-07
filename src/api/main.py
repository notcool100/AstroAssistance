"""
Main FastAPI application for AstroAssistance.
"""
import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import jwt
from typing import Dict, Any, Optional

from src.core.config import config_manager
from src.core.logger import app_logger
from src.api.routes import (
    tasks_router, reminders_router, goals_router,
    preferences_router, recommendations_router
)


# Create FastAPI app
app = FastAPI(
    title="AstroAssistance API",
    description="API for AstroAssistance, a self-learning productivity assistant",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT settings
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key")  # In production, use a secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Mock user database for demonstration purposes
# In a real implementation, this would be replaced with a proper database
USERS_DB = {
    "user@example.com": {
        "id": "user123",
        "email": "user@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
        "full_name": "John Doe"
    }
}


def get_user(email: str) -> Optional[Dict[str, Any]]:
    """
    Get a user by email.
    
    Args:
        email: User email
        
    Returns:
        User dictionary or None if not found
    """
    return USERS_DB.get(email)


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user.
    
    Args:
        email: User email
        password: User password
        
    Returns:
        User dictionary or None if authentication fails
    """
    user = get_user(email)
    if not user:
        return None
    
    # In a real implementation, verify the password hash
    # For demonstration, we'll just check if the password is "password"
    if password != "password":
        return None
    
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Token data
        expires_delta: Token expiration time
        
    Returns:
        JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Get an access token.
    
    Args:
        form_data: OAuth2 form data
        
    Returns:
        Access token
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "user_id": user["id"]},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


# Include routers
app.include_router(tasks_router)
app.include_router(reminders_router)
app.include_router(goals_router)
app.include_router(preferences_router)
app.include_router(recommendations_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to AstroAssistance API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    app_logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "details": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    host = config_manager.get("api.host", "0.0.0.0")
    port = config_manager.get("api.port", 8000)
    debug = config_manager.get("api.debug", True)
    
    uvicorn.run("src.api.main:app", host=host, port=port, reload=debug)