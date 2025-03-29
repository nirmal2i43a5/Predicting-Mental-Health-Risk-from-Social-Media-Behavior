import numpy as np 
from utils.data_loader import lr_model, dt_model, rf_model, xgb_model, nb_model

def predict_mental_health_risk( model_choice, age, gender, relationship_status, occupation_status,
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
    if model_choice == 'Logistic Regression':
        prediction = lr_model.predict([features_array])
        
    elif model_choice == 'Decision Tree':
        prediction = dt_model.predict([features_array])
        

    elif model_choice == 'Random Forest':
        prediction = rf_model.predict([features_array])
    
    elif model_choice == 'XGBoost':
        prediction = xgb_model.predict([features_array])
        
    elif model_choice == 'Naive Bayes':
        prediction = nb_model.predict([features_array])
        
    else:
        return None
       
    
    return prediction