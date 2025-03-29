import os
import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image
from utils.data_loader import lr_model, X_test, y_test, dt_model, dt_model, rf_model, xgb_model, nb_model
from utils.predict import predict_mental_health_risk
from utils.performance_metrics import get_model_metrics, display_model_metrics
from utils.confusion_matrix import display_confusion_matrix


file_path = "mental_health.png"
if os.path.exists(file_path):
    image = Image.open(file_path)
else:
    print(f"File '{file_path}' not found!")

st.image(image, caption="Image")


models = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "Naive Bayes": nb_model,
    "XGBoost": xgb_model
}


gender_choices = {
    'Male': 0,
    'Female': 1,
    'Non-binary': 2
}

relationship_choices = {
    'Single': 0,
    'Relationship': 1,
    'Married': 2,
    'Divorced': 3
}

occupation_choices = {
    'University Student': 0,
    'Salaried Worker': 1,
    'School Student': 2,
    'Retired': 3
}

use_social_media_choices = {
    'No': 0,
    'Yes': 1
}

time_choices = {
    'Less than an Hour': 0.5,
    'Between 1 and 2 hours': 1.5,
    'Between 2 and 3 hours': 2.5,
    'Between 3 and 4 hours': 3.5,
    'Between 4 and 5 hours': 4.5,
    'More than 5 hours': 5.5
}



                
def main():
    # st.title("Social Media Mental Health Analysis")
    
    html_temp = """
    <div style="background-color:blue;padding:10px;border-radius:10px">
      <h2 style="color:white;text-align:center;">Predicting Mental Health From Social Media</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # -------------------------
    # Model Selection Section
    # -------------------------
    model_choice = st.selectbox("Select Prediction Algorithm", list(models.keys()))
    print(model_choice)
    # selected_model = models[model_choice]
    
    # -------------------------
    # Input Features Section
    # -------------------------
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
    with col2:
        gender_choice = st.selectbox("Gender", list(gender_choices.keys()), index=0)
        gender = gender_choices[gender_choice]
        
    col1, col2 = st.columns(2)
    with col1:
        relationship_status_choice = st.selectbox("Relationship Status", list(relationship_choices.keys()), index=0)
        relationship_status = relationship_choices[relationship_status_choice]
    with col2:
        occupation_status_choice = st.selectbox("Occupation Status", list(occupation_choices.keys()), index=0)
        occupation_status = occupation_choices[occupation_status_choice]
        
    col1, col2 = st.columns(2)
    with col1:
        use_social_media_choice = st.selectbox("Use Social Media", list(use_social_media_choices.keys()), index=0)
        use_social_media = use_social_media_choices[use_social_media_choice]
    with col2:
        daily_social_media_time_choice = st.selectbox("Daily Social Media Time", list(time_choices.keys()), index=0)
        daily_social_media_time = time_choices[daily_social_media_time_choice]
    
    col1, col2 = st.columns(2)
    with col1:
        Discord = st.number_input("Discord", min_value=0, value=0, step=1)
    with col2:
        Facebook = st.number_input("Facebook", min_value=0, value=0, step=1)
        
    col1, col2 = st.columns(2)
    with col1:
        Instagram = st.number_input("Instagram", min_value=0, value=0, step=1)
    with col2:
        Pinterest = st.number_input("Pinterest", min_value=0, value=0, step=1)
        
    col1, col2 = st.columns(2)
    with col1:
        Reddit = st.number_input("Reddit", min_value=0, value=0, step=1)
    with col2:
        Snapchat = st.number_input("Snapchat", min_value=0, value=0, step=1)
        
    col1, col2 = st.columns(2)
    with col1:
        TikTok = st.number_input("TikTok", min_value=0, value=0, step=1)
    with col2:
        Twitter = st.number_input("Twitter", min_value=0, value=0, step=1)
        
    YouTube = st.number_input("YouTube", min_value=0, value=0, step=1)
    
    result = ""
    # -------------------------
    # Prediction & Results Display
    # -------------------------
    if st.button("Predict"):
        # selected_model = models[model_choice]
        predicted_label = predict_mental_health_risk( 
                                                model_choice, age, gender, relationship_status, 
                                               occupation_status, use_social_media, 
                                               daily_social_media_time, Discord, 
                                               Facebook, Instagram, Pinterest, Reddit, 
                                               Snapchat, TikTok, Twitter, YouTube
                                               )
        
        print(predicted_label,'-------------I am testing predicted label-----------')
        if predicted_label == 1:
            st.error('Menatl Health Risk is High')
        else:
            st.success('Mental Health Risk is Low')

        metrics = get_model_metrics(model_choice, X_test, y_test)
        display_model_metrics(model_choice,metrics)
        display_confusion_matrix(model_choice, X_test, y_test)
    


if __name__ == '__main__':
    main()