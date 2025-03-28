import streamlit as st 
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
)


def get_model_metrics(model, X_test, y_test):
    """
    Calculate and return comprehensive model metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics
    
def display_model_metrics(selected_model, metrics):
 
    st.subheader(f"Prediction  Performance  for {selected_model}")
    
    # Create columns for metrics
    cols = st.columns(len(metrics))
    
    # Display each metric
    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(
                label=metric_name, 
                value=f"{metric_value:.4f}"
            )