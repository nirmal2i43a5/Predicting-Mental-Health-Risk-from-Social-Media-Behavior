
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st 
from sklearn.metrics import confusion_matrix
from utils.data_loader import lr_model, dt_model, dt_model, rf_model, xgb_model, nb_model


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Create a heatmap-style confusion matrix for Streamlit

    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, 
                annot=True,  # Show numerical values in each cell
                fmt='d',     # Integer formatting
                cmap='Blues',  # Color palette
                xticklabels=classes,
                yticklabels=classes)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
       
    return plt

def display_confusion_matrix(model_choice, X_test, y_test):
    """
    Display confusion matrix in Streamlit
    
    """
    if model_choice == 'Logistic Regression':
        y_pred = lr_model.predict(X_test)
          # Get classes from the model
        classes = lr_model.classes_
        
    elif model_choice == 'Decision Tree':
        y_pred = dt_model.predict(X_test)
        classes = dt_model.classes_
        
        
    elif model_choice == 'Random Forest':
        y_pred = rf_model.predict(X_test)
        classes = rf_model.classes_
        
    elif model_choice == 'XGBoost':
        y_pred = xgb_model.predict(X_test)
        classes = xgb_model.classes_
        
    elif model_choice == 'Naive Bayes':
        y_pred = nb_model.predict(X_test)
        classes = nb_model.classes_
        
    else:
        return None

    
  
    
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