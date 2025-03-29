import streamlit as st 
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
)
from utils.data_loader import lr_model, dt_model, dt_model, rf_model, xgb_model, nb_model

def get_model_metrics(model_choice, X_test, y_test):
    """
    Calculate and return comprehensive model metrics
    """
    if model_choice == 'Logistic Regression':
        y_pred = lr_model.predict(X_test)
        
        
    elif model_choice == 'Decision Tree':
        y_pred = dt_model.predict(X_test)
        
    elif model_choice == 'Random Forest':
        y_pred = rf_model.predict(X_test)
        
    elif model_choice == 'XGBoost':
        y_pred = xgb_model.predict(X_test)
        
    elif model_choice == 'Naive Bayes':
        y_pred = nb_model.predict(X_test)
    else:
        return None
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
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