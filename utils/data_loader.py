import pickle 

# Get the current working directory and build the full path to the file
# current_dir = os.getcwd()
# file_path = os.path.join(current_dir, 'logistic.pkl')
# print("Current Working Directory:", current_dir)
# print("File path to pickle file:", file_path)

# # Check if the file exists before attempting to open it
# if os.path.exists(file_path):
#     with open(file_path, 'rb') as pickle_in:
#         lr_model = pickle.load(pickle_in)
#     print("Pickle file loaded successfully!")
# else:
#     print(f"Error: File not found at {file_path}")
# Load your models (update file paths as needed)


with open('logistic.pkl', 'rb') as f:
    saved_data = pickle.load(f)

lr_model = saved_data['model']
X_test = saved_data['X_test']
y_test = saved_data['y_test']

# with open('lr_model.pkl', 'rb') as f:
#     lr_model = pickle.load(f)
# with open('dt_model.pkl', 'rb') as f:
#     dt_model = pickle.load(f)
# with open('rf_model.pkl', 'rb') as f:
#     rf_model = pickle.load(f)
# with open('xgb_model.pkl', 'rb') as f:
#     xgb_model = pickle.load(f)