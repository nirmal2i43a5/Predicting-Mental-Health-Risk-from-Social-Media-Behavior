import pickle 



with open('lr_model.pkl', 'rb') as f:
    saved_data = pickle.load(f)

lr_model = saved_data['model']
X_test = saved_data['X_test']
y_test = saved_data['y_test']

    
with open('dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)
    
    
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
    
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
    
with open('nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)