import streamlit as st
import os
# --- FIX: Using the dedicated langchain_cohere package for better compatibility ---
# This package ensures better integration and usually resolves version conflicts 
# that cause internal errors like 'token_count' attribute missing.
from langchain_cohere import ChatCohere # Updated import location
from langchain_core.messages import HumanMessage # Required for Chat models

# Configuration and Initialization
# The Cohere API key provided by the user has been set below.
COHERE_API_KEY = "COHERE_API_KEY"

# --- Recommended replacement model ---
# The model 'command-r' was deprecated on September 15, 2025.
# We are switching to 'command-r-08-2024', which is a modern, supported chat model.
NEW_MODEL_NAME = "command-r-08-2024"

def get_cohere_response(question):
    """
    Initializes the Cohere Chat Model and generates a response to the question.
    """
    try:
        # --- Instantiate ChatCohere using the direct package import ---
        llm = ChatCohere(
            cohere_api_key=COHERE_API_KEY,
            model=NEW_MODEL_NAME, # Updated to a supported model
            temperature=0.5
        )
        
        # Chat models require a list of message objects
        messages = [HumanMessage(content=question)]
        
        # The invoke method returns an AIMessage object.
        response = llm.invoke(messages)
        
        # We explicitly return only the string content.
        return response.content
        
    except Exception as e:
        # Catch and display any errors during the API call
        return f"An error occurred during Cohere Chat API call: {e}"

# --- Streamlit App Layout ---
st.set_page_config(page_title="Q&A Chatbot with Cohere", layout="centered")

st.title("🧠 LangChain + Cohere Chatbot (Fixed)")
st.markdown(f"The model has been updated from the deprecated `command-r` to the supported `{NEW_MODEL_NAME}`.")

# User Input
user_input = st.text_input("Enter your question:", key="user_input_key")

submit = st.button("Get Answer", key="submit_button")

# Response Logic
if submit:
    if not user_input or user_input.strip() == "":
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking..."):
            response = get_cohere_response(user_input)
        
        st.subheader("🤖 Response")
        # Display the response in a distinct block
        st.info(response)

st.sidebar.markdown(
    f"""
    **Model Details:**
    - **Provider:** Cohere (Using Direct Chat API Integration)
    - **Model:** `{NEW_MODEL_NAME}` (Updated)
    - **Temperature (Creativity):** 0.5
    """
)

# Instruction for running the app - Keeping the dependency fix instruction
st.caption("🚨 **ACTION REQUIRED (Dependency Fix):** If you still see a `TypeError: unhashable type: 'list'`, please run `pip install aiohttp==3.8.4` in your terminal to resolve the environment conflict, and then restart your Streamlit app.")