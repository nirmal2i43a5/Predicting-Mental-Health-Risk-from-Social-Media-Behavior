  
import os
import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils.data_loader import lr_model, X_test, y_test
from utils.predict import predict_mental_health_risk
from utils.performance_metrics import get_model_metrics, display_model_metrics


# Dictionary for model selection
models = {
    "Logistic Regression": lr_model,
    # "Decision Tree": dt_model,
    # "Random Forest": rf_model,
    # "XGBoost": xgb_model
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




def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Create a heatmap-style confusion matrix for Streamlit
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - classes: List of class labels
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a figure and set its size
    plt.figure(figsize=(8, 6))
    
    # Create heatmap using Seaborn
    sns.heatmap(cm, 
                annot=True,  # Show numerical values in each cell
                fmt='d',     # Integer formatting
                cmap='Blues',  # Color palette
                xticklabels=classes,
                yticklabels=classes)
    
    # Set labels and title
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Adjust layout and return the plot
    plt.tight_layout()
    
    # Return the plot to be used in Streamlit
    return plt

def display_confusion_matrix(model, X_test, y_test):
    """
    Display confusion matrix in Streamlit
    
    Parameters:
    - model: Trained machine learning model
    - X_test: Test features
    - y_test: True labels
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get classes from the model
    classes = model.classes_
    
    # Create the confusion matrix plot
    st.subheader("Confusion Matrix Visualization")
    
    # Plot the confusion matrix
    fig = plot_confusion_matrix(y_test, y_pred, classes)
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # # Display raw confusion matrix values
    # cm = confusion_matrix(y_test, y_pred)
    # st.write("Confusion Matrix Values:")
    # st.dataframe(cm)
                
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
    selected_model = models[model_choice]
    
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
        selected_model = models[model_choice]
        predicted_label = predict_mental_health_risk( age, gender, relationship_status, 
                                               occupation_status, use_social_media, 
                                               daily_social_media_time, Discord, 
                                               Facebook, Instagram, Pinterest, Reddit, 
                                               Snapchat, TikTok, Twitter, YouTube)
        
    
        if predicted_label == 1:
            st.error('Menatl Health Risk is High')
        else:
            st.success('Mental Health Risk is Low')

        metrics = get_model_metrics(selected_model, X_test, y_test)
        display_model_metrics(model_choice,metrics)
        display_confusion_matrix(lr_model, X_test, y_test)
    
    # if st.button("About"):
    #     st.text("Let's Learn")
    #     st.text("Built with Streamlit")

if __name__ == '__main__':
    main()