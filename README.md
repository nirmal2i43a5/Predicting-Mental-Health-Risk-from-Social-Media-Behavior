# Mental Health Prediction API

This project uses FastAPI to provide a REST API for predicting mental health risks based on social media usage patterns. The API serves machine learning models trained to assess the correlation between social media usage and mental health outcomes.

## Features

- RESTful API using FastAPI
- Comprehensive Authentication System:
  - User registration and login
  - JWT-based authentication with access and refresh tokens
  - Password reset functionality
  - Account management
- Multiple ML models for prediction:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - Naive Bayes
- Model metrics endpoints
- Interactive Swagger documentation

## Requirements

See `requirements.txt` for the full list of dependencies, but the main requirements are:

- Python 3.8+
- FastAPI
- Uvicorn
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- PyJWT

## Installation

1. Clone the repository
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the API

Start the API server with:

```
python api.py
```

This will start the server at http://localhost:8000 by default.

Alternatively, you can use Uvicorn directly:

```
uvicorn api:app --reload
```

## API Documentation

Once the API is running, you can access the interactive documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Authentication

- `POST /register` - Register a new user
- `POST /login` - Login to get access and refresh tokens
- `POST /refresh-token` - Refresh an expired access token
- `POST /forgot-password` - Request a password reset
- `POST /reset-password` - Reset password with token
- `GET /me` - Get current user profile

### Prediction

- `GET /models` - List available prediction models
- `POST /predict` - Make a mental health risk prediction
- `GET /metrics/{model_name}` - Get metrics for a specific model
- `GET /health` - Server health check

## Authentication Flow

### User Registration

```python
# Register a new user
response = requests.post(
    "http://localhost:8000/register",
    json={
        "username": "newuser",
        "email": "user@example.com",
        "password": "securepassword",
        "full_name": "New User"
    }
)
```

### Login

```python
# Login to obtain tokens
response = requests.post(
    "http://localhost:8000/login",
    json={
        "username": "newuser",
        "password": "securepassword"
    }
)
tokens = response.json()
access_token = tokens["access_token"]
refresh_token = tokens["refresh_token"]
```

### Using Access Tokens

```python
# Call any protected endpoint
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.get("http://localhost:8000/me", headers=headers)
```

### Password Reset

```python
# 1. Request password reset
response = requests.post(
    "http://localhost:8000/forgot-password",
    json={"email": "user@example.com"}
)

# 2. User receives reset token via email

# 3. Reset password using token
response = requests.post(
    "http://localhost:8000/reset-password",
    json={
        "token": "received_token",
        "new_password": "new_secure_password"
    }
)
```

## Prediction Example

```python
# Make prediction
headers = {"Authorization": f"Bearer {access_token}"}
prediction_data = {
    "model_choice": "Logistic Regression",
    "age": 30,
    "gender": 0,  # Male
    "relationship_status": 1,  # Relationship
    "occupation_status": 1,  # Salaried Worker
    "use_social_media": 1,  # Yes
    "daily_social_media_time": 3.5,
    # Social media platform usage
    "Discord": 1,
    "Facebook": 1,
    # ... other platforms
}

response = requests.post(
    "http://localhost:8000/predict",
    headers=headers,
    json=prediction_data
)
prediction = response.json()
```

## Example Client

See the `client_example.py` file for a complete example showing how to:
- Register a new user
- Login with credentials
- Request password reset
- Reset password with token
- Get user profile
- Make predictions
- Get model metrics

## Security Notes

- For production use, replace the hard-coded secret key with a secure, environment-based key
- Implement proper password hashing (currently using SHA-256 for simplicity)
- Consider more comprehensive authentication mechanisms
- Restrict CORS settings in production
- Use HTTPS for all communications

## Next Steps

Potential improvements:
- Database integration for user management (PostgreSQL, MongoDB)
- Email service integration for password resets
- Rate limiting for API endpoints
- Request validation middleware
- Create a front-end application to interact with the API
- Add model re-training endpoints
- Implement session management
- Add multi-factor authentication 