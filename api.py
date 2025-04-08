import os
import numpy as np
import pickle
import secrets
import hashlib
import string
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, Query, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
import uvicorn
import jwt
from enum import Enum
from utils.data_loader import lr_model, dt_model, rf_model, xgb_model, nb_model, X_test, y_test
from utils.predict import predict_mental_health_risk
from utils.performance_metrics import get_model_metrics

# Initialize FastAPI app
app = FastAPI(
    title="Mental Health Prediction API",
    description="API for predicting mental health risk based on social media usage",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# JWT configuration
SECRET_KEY = "your-secret-key"  # Change this to a secure secret key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password reset token expiration (in minutes)
PASSWORD_RESET_EXPIRE_MINUTES = 30

# Models for API requests and responses
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    RESET = "reset"

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserDB(UserBase):
    hashed_password: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class UserResponse(UserBase):
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None
    expires_in: int = 1800  # 30 minutes in seconds

class TokenData(BaseModel):
    username: Optional[str] = None
    token_type: TokenType = TokenType.ACCESS
    exp: Optional[datetime] = None

class MentalHealthPredictionRequest(BaseModel):
    model_choice: str = Field(..., description="ML model to use for prediction")
    age: int = Field(..., description="Age of the person", ge=0, le=120)
    gender: int = Field(..., description="Gender (0: Male, 1: Female, 2: Non-binary)")
    relationship_status: int = Field(..., description="Relationship status (0: Single, 1: Relationship, 2: Married, 3: Divorced)")
    occupation_status: int = Field(..., description="Occupation status (0: University Student, 1: Salaried Worker, 2: School Student, 3: Retired)")
    use_social_media: int = Field(..., description="Whether the person uses social media (0: No, 1: Yes)")
    daily_social_media_time: float = Field(..., description="Daily time spent on social media in hours")
    Discord: int = Field(..., description="Whether the person uses Discord (0: No, 1: Yes)")
    Facebook: int = Field(..., description="Whether the person uses Facebook (0: No, 1: Yes)")
    Instagram: int = Field(..., description="Whether the person uses Instagram (0: No, 1: Yes)") 
    Pinterest: int = Field(..., description="Whether the person uses Pinterest (0: No, 1: Yes)")
    Reddit: int = Field(..., description="Whether the person uses Reddit (0: No, 1: Yes)")
    Snapchat: int = Field(..., description="Whether the person uses Snapchat (0: No, 1: Yes)")
    TikTok: int = Field(..., description="Whether the person uses TikTok (0: No, 1: Yes)")
    Twitter: int = Field(..., description="Whether the person uses Twitter (0: No, 1: Yes)")
    YouTube: int = Field(..., description="Whether the person uses YouTube (0: No, 1: Yes)")

class MentalHealthPredictionResponse(BaseModel):
    prediction: int
    prediction_text: str
    confidence: Optional[float] = None
    model_metrics: Optional[Dict[str, Any]] = None

# Mock user database - replace with proper database in production
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "email": "testuser@example.com",
        "full_name": "Test User",
        "hashed_password": hashlib.sha256("password".encode()).hexdigest(),
        "role": UserRole.USER,
        "is_active": True,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    },
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "hashed_password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": UserRole.ADMIN,
        "is_active": True,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
}

# Store password reset tokens
password_reset_tokens = {}

def get_password_hash(password: str) -> str:
    """Hash a password with SHA-256 (use bcrypt in production)"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return get_password_hash(plain_password) == hashed_password

def get_user(db, username: str):
    """Get a user from the database by username"""
    if username in db:
        user_dict = db[username]
        return user_dict
    return None

def get_user_by_email(db, email: str):
    """Get a user from the database by email"""
    for username, user_data in db.items():
        if user_data.get("email") == email:
            return user_data
    return None

def authenticate_user(db, username: str, password: str):
    """Authenticate a user with username and password"""
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_token(data: dict, expires_delta: Optional[timedelta] = None, token_type: TokenType = TokenType.ACCESS):
    """Create a JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "type": token_type})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create an access token"""
    return create_token(data, expires_delta, TokenType.ACCESS)

def create_refresh_token(data: dict):
    """Create a refresh token with a longer expiry"""
    return create_token(data, timedelta(days=7), TokenType.REFRESH)

def create_reset_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a password reset token"""
    return create_token(data, expires_delta, TokenType.RESET)

def generate_password_reset_token() -> str:
    """Generate a secure random token for password reset"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(64))

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get the current user from a JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type = payload.get("type")
        if username is None or token_type != TokenType.ACCESS:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    
    # Convert to response model excluding password
    return user

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """Verify that the current user is active"""
    if not current_user.get("is_active", False):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def send_password_reset_email(email: str, token: str):
    """
    Simulated function to send password reset email
    In a real application, you would use an email service
    """
    # In production, implement actual email sending
    # For now, just print the reset link
    reset_link = f"https://your-frontend.com/reset-password?token={token}"
    print(f"Sending password reset email to {email}")
    print(f"Reset link: {reset_link}")
    return reset_link

# API Routes

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate):
    """Register a new user"""
    if user.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email is already registered
    for existing_user in fake_users_db.values():
        if existing_user.get("email") == user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Hash the password
    hashed_password = get_password_hash(user.password)
    
    # Create user object
    db_user = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "role": UserRole.USER,
        "is_active": True,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    # Add user to database
    fake_users_db[user.username] = db_user
    
    # Return user without password
    return db_user

@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: UserLogin):
    """Login to get JWT access token"""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": user["username"]}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.post("/refresh-token", response_model=Token)
async def refresh_access_token(token: str = Depends(oauth2_scheme)):
    """Get a new access token using a refresh token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type = payload.get("type")
        
        if username is None or token_type != TokenType.REFRESH:
            raise credentials_exception
            
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    
    # Create new tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": user["username"]}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    """Request a password reset token"""
    user = get_user_by_email(fake_users_db, request.email)
    if not user:
        # Don't reveal that the email doesn't exist
        # Just return success to prevent email enumeration
        return {"message": "If your email is registered, you will receive a password reset link"}
    
    # Generate a reset token
    reset_token = generate_password_reset_token()
    
    # Store the token with the user info
    expiry = datetime.utcnow() + timedelta(minutes=PASSWORD_RESET_EXPIRE_MINUTES)
    password_reset_tokens[reset_token] = {
        "user": user["username"],
        "expires": expiry
    }
    
    # Send the email in the background
    background_tasks.add_task(
        send_password_reset_email,
        request.email,
        reset_token
    )
    
    return {"message": "If your email is registered, you will receive a password reset link"}

@app.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    """Reset password using a reset token"""
    # Check if token exists and is valid
    if request.token not in password_reset_tokens:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    token_data = password_reset_tokens[request.token]
    
    # Check if token has expired
    if datetime.utcnow() > token_data["expires"]:
        # Remove expired token
        del password_reset_tokens[request.token]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired"
        )
    
    # Get the user
    username = token_data["user"]
    if username not in fake_users_db:
        # This shouldn't happen if the token was properly created
        del password_reset_tokens[request.token]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not found"
        )
    
    # Update the password
    fake_users_db[username]["hashed_password"] = get_password_hash(request.new_password)
    fake_users_db[username]["updated_at"] = datetime.now()
    
    # Remove the used token
    del password_reset_tokens[request.token]
    
    return {"message": "Password has been reset successfully"}

@app.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_active_user)):
    """Get the current user's profile"""
    return current_user

@app.get("/models")
async def get_available_models(current_user: dict = Depends(get_current_active_user)):
    """Get a list of available prediction models"""
    return {
        "models": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "Naive Bayes"
        ]
    }

@app.post("/predict", response_model=MentalHealthPredictionResponse)
async def predict(request: MentalHealthPredictionRequest, current_user: dict = Depends(get_current_active_user)):
    """
    Predict mental health risk based on social media usage and personal attributes
    """
    try:
        prediction = predict_mental_health_risk(
            request.model_choice,
            request.age,
            request.gender,
            request.relationship_status,
            request.occupation_status,
            request.use_social_media,
            request.daily_social_media_time,
            request.Discord,
            request.Facebook,
            request.Instagram,
            request.Pinterest,
            request.Reddit,
            request.Snapchat,
            request.TikTok,
            request.Twitter,
            request.YouTube
        )
        
        # Get model metrics
        metrics = get_model_metrics(request.model_choice, X_test, y_test)
        
        # Format the response
        prediction_result = int(prediction[0])
        prediction_text = "High Mental Health Risk" if prediction_result == 1 else "Low Mental Health Risk"
        
        return {
            "prediction": prediction_result,
            "prediction_text": prediction_text,
            "confidence": float(metrics.get("accuracy", 0.0)),
            "model_metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/metrics/{model_name}")
async def get_metrics(model_name: str, current_user: dict = Depends(get_current_active_user)):
    """Get performance metrics for a specific model"""
    if model_name not in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Naive Bayes"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Model {model_name} not found"
        )
    
    metrics = get_model_metrics(model_name, X_test, y_test)
    return {"model": model_name, "metrics": metrics}

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 