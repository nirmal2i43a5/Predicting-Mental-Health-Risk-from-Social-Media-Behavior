import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from pathlib import Path

def load_or_create_config():
    """Load the config file or create a simple one if it doesn't exist"""
    config_path = Path("config.yaml")
    
    if not config_path.exists():
        # Create default config with admin user
        config = {
            'credentials': {
                'usernames': {
                    'admin': {
                        'name': 'Nirmal',
                        'password': 'password',  # Use Hasher if needed
                    }
                }
            },
            'cookie': {
                'expiry_days': 30,
                'key': 'some_signature_key',
                'name': 'auth_cookie'
            }
        }
        
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    
    # Load config
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
    
    return config

def authentication():
    st.title("Streamlit Simple Authentication")
    
    # Load configuration
    config = load_or_create_config()
    
    # Create the authenticator
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    print( config['cookie']['name'],"-------------------------------------")
    # Display login widget
    name, authentication_status, username = authenticator.login(location="main")
    
    
    # Handle authentication status
    if authentication_status == False:
        st.error('Username/password is incorrect')
        
    elif authentication_status == None:
        st.warning('Please enter your username and password')
        
    else:
        # User is logged in
        st.success(f'Welcome, {name}!')
        authenticator.logout('Logout', location='main')
        
        # Content only visible to logged in users
        st.header("Protected Content")
        st.write("This content is only visible to authenticated users.")
        
        # Add your App content here
        st.subheader("Your Application")
        st.write("Put your Streamlit App functionality here...")

# Run authentication
authentication()
