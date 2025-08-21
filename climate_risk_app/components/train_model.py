import streamlit as st
from config import Config
from api_utils import make_api_request, display_response

def show_train_model(base_url: str):
    st.header("Train Model")
    
    epochs = st.number_input("Epochs", min_value=10, max_value=500, value=100, key="train_model_epochs")
    learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, key="train_model_lr")
    
    if st.button("Train Model", key="train_model_button"):
        with st.spinner(f"Training model for {epochs} epochs..."):
            response = make_api_request(
                base_url,
                Config.ENDPOINTS["train"],
                "POST",
                {"epochs": epochs, "learning_rate": learning_rate}
            )
            display_response(response, "Model training completed!")