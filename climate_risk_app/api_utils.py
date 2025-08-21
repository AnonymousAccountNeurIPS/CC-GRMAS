import streamlit as st
import requests
from typing import Dict

@st.cache_data
def make_api_request(base_url: str, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    url = f"{base_url}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, params=data)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

def display_response(response: Dict, success_message: str = "Operation completed successfully"):
    if response["success"]:
        st.success(success_message)
        return response["data"]
    else:
        st.error(f"Error: {response['error']}")
        return None

def check_connection(base_url: str):
    if "connection_status" not in st.session_state:
        try:
            response = requests.get(f"{base_url}/")
            if response.status_code == 200:
                st.session_state.connection_status = "✅ API Connected"
            else:
                st.session_state.connection_status = f"❌ API Error: Status {response.status_code}"
        except requests.exceptions.ConnectionError:
            st.session_state.connection_status = "❌ API Disconnected: Unable to connect"
        except Exception as e:
            st.session_state.connection_status = f"❌ API Error: {str(e)}"
    
    st.sidebar.write(st.session_state.connection_status)