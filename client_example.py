import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

def register_user(username, email, password, full_name=None):
    """Register a new user"""
    user_data = {
        "username": username,
        "email": email,
        "password": password
    }
    if full_name:
        user_data["full_name"] = full_name
        
    response = requests.post(
        f"{BASE_URL}/register",
        json=user_data
    )
    
    if response.status_code == 201:
        print("User registered successfully!")
        return response.json()
    else:
        print(f"Registration failed: {response.text}")
        return None

def login(username, password):
    """Login to get access and refresh tokens"""
    response = requests.post(
        f"{BASE_URL}/login",
        json={"username": username, "password": password}
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Login failed: {response.text}")
        return None

def refresh_token(refresh_token):
    """Get a new access token using a refresh token"""
    headers = {"Authorization": f"Bearer {refresh_token}"}
    response = requests.post(f"{BASE_URL}/refresh-token", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Token refresh failed: {response.text}")
        return None

def forgot_password(email):
    """Request a password reset"""
    response = requests.post(
        f"{BASE_URL}/forgot-password",
        json={"email": email}
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Password reset request failed: {response.text}")
        return None

def reset_password(reset_token, new_password):
    """Reset password using a reset token"""
    response = requests.post(
        f"{BASE_URL}/reset-password",
        json={"token": reset_token, "new_password": new_password}
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Password reset failed: {response.text}")
        return None

def get_user_profile(token):
    """Get the current user's profile"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/me", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get user profile: {response.text}")
        return None

def get_available_models(token):
    """Get list of available prediction models"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/models", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get models: {response.text}")
        return None

def predict_mental_health(token, data):
    """Make a prediction using the API"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        f"{BASE_URL}/predict",
        headers=headers,
        json=data
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Prediction failed: {response.text}")
        return None

def get_model_metrics(token, model_name):
    """Get metrics for a specific model"""
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/metrics/{model_name}", headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get metrics: {response.text}")
        return None

if __name__ == "__main__":
    # Example usage
    print("\n========== AUTHENTICATION DEMO ==========")
    
    # 1. Register a new user (uncomment to test)
    # new_user = register_user(
    #     username="newuser", 
    #     email="newuser@example.com", 
    #     password="securepassword",
    #     full_name="New Test User"
    # )
    
    # 2. Login with existing user
    print("\nLogging in...")
    auth_data = login("testuser", "password")
    
    if not auth_data:
        print("Login failed. Exiting.")
        exit()
        
    access_token = auth_data["access_token"]
    refresh_token = auth_data["refresh_token"]
    print(f"Login successful! Token expires in {auth_data['expires_in']} seconds")
    
    # 3. Get user profile
    print("\nFetching user profile...")
    profile = get_user_profile(access_token)
    if profile:
        print(f"Logged in as: {profile['username']} ({profile['email']})")
    
    # 4. Password reset flow (uncomment to test)
    # print("\nTesting password reset flow...")
    # forgot_result = forgot_password("testuser@example.com")
    # if forgot_result:
    #     print(forgot_result["message"])
    #     # In a real app, the user would receive the token via email
    #     # For testing, you would get the token from the console output
    #     reset_token = input("Enter the reset token from the console output: ")
    #     reset_result = reset_password(reset_token, "newpassword")
    #     if reset_result:
    #         print(reset_result["message"])
    #         # Login with new password
    #         auth_data = login("testuser", "newpassword")
    #         if auth_data:
    #             print("Login with new password successful!")
    #             access_token = auth_data["access_token"]
    
    # 5. Token refresh (uncomment to test after waiting for token expiration)
    # print("\nRefreshing token...")
    # new_tokens = refresh_token(refresh_token)
    # if new_tokens:
    #     print("Token refreshed successfully!")
    #     access_token = new_tokens["access_token"]
    #     refresh_token = new_tokens["refresh_token"]
    
    print("\n========== PREDICTION DEMO ==========")
    
    # Get available models
    models = get_available_models(access_token)
    if models:
        print(f"\nAvailable models: {models['models']}")
    
    # Example prediction data
    prediction_data = {
        "model_choice": "Logistic Regression",
        "age": 30,
        "gender": 0,  # Male
        "relationship_status": 1,  # Relationship
        "occupation_status": 1,  # Salaried Worker
        "use_social_media": 1,  # Yes
        "daily_social_media_time": 3.5,  # 3.5 hours per day
        "Discord": 1,
        "Facebook": 1,
        "Instagram": 1,
        "Pinterest": 0,
        "Reddit": 1,
        "Snapchat": 0,
        "TikTok": 1,
        "Twitter": 1,
        "YouTube": 1
    }
    
    # Make prediction
    print("\nMaking prediction...")
    prediction = predict_mental_health(access_token, prediction_data)
    if prediction:
        print(f"Prediction result: {prediction['prediction_text']}")
        print(f"Confidence: {prediction['confidence']:.2f}")
        
        # Print model metrics
        print("\nModel metrics:")
        for key, value in prediction["model_metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # Get detailed metrics for a specific model
    model_name = "Random Forest"
    print(f"\nGetting metrics for {model_name}...")
    metrics = get_model_metrics(access_token, model_name)
    if metrics:
        print(f"Metrics for {metrics['model']}:")
        for key, value in metrics["metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\nDone!") 