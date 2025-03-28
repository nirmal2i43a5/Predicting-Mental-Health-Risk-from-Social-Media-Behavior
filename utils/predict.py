import numpy as np 
from utils.data_loader import lr_model

def predict_mental_health_risk( age, gender, relationship_status, occupation_status,
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
    prediction=lr_model.predict([features_array])
    return prediction