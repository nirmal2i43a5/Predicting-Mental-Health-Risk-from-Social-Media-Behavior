
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st 
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Create a heatmap-style confusion matrix for Streamlit

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