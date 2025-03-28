  
import os
import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image


import os

import os
import pickle

# Get the current working directory and build the full path to the file
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'logistic.pkl')
print("Current Working Directory:", current_dir)
print("File path to pickle file:", file_path)

# Check if the file exists before attempting to open it
if os.path.exists(file_path):
    with open(file_path, 'rb') as pickle_in:
        lr_model = pickle.load(pickle_in)
    print("Pickle file loaded successfully!")
else:
    print(f"Error: File not found at {file_path}")


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



def predict_mental_health_risk(age, gender, relationship_status, occupation_status,
                                use_social_media, daily_social_media_time, Discord,
                                Facebook, Instagram, Pinterest, Reddit, Snapchat,
                                TikTok, Twitter, YouTube):

    
    features_array = np.array([
            age,
            gender,
            relationship_status,
            occupation_status,
            use_social_media,
            daily_social_media_time,
            Discord,
            Facebook,
            Instagram,
            Pinterest,
            Reddit,
            Snapchat,
            TikTok,
            Twitter,
            YouTube
])
    # (required for sklearn's predict method)
    prediction=lr_model.predict([features_array])
    print(prediction,"---------------------------------------")
    return prediction[0]


def main():
    st.title("Social Media Mental Health Analysis")
    
    html_temp = """
    <div style="background-color:blue;padding:10px">
      <h2 style="color:white;text-align:center;">Social Media Mental Health Analysis ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
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
    
    if st.button("Predict"):
        result = predict_mental_health_risk(age, gender, relationship_status, 
                                               occupation_status, use_social_media, 
                                               daily_social_media_time, Discord, 
                                               Facebook, Instagram, Pinterest, Reddit, 
                                               Snapchat, TikTok, Twitter, YouTube)
        if result == 1:
            st.error('Menatl Health Risk is High')
        else:
            st.success('Mental Health Risk is Low')

    
    if st.button("About"):
        st.text("Let's Learn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()