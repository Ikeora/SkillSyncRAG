"""
This code will act as the front-end 'main' make function
calls into the backend.
"""

# Fix for PyTorch/Streamlit compatibility
import asyncio
import sys

if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# UI libraries 
import streamlit as st
from urllib.error import URLError

from streamlit_chat import message 
from streamlit.components.v1 import html

import os
from PIL import Image

# Data Wrangling
import pandas as pd
import numpy as np

# Charting Libraries 
import plotly.io as pio
import json
# Needed for Plotly image export
pio.kaleido.scope.mathjax = None

# Custom agent 
from rag import query

# Disable problematic watchers:
from streamlit import config
config.set_option("server.fileWatcherType", "none")

# Add this before any PyTorch-related imports
import torch

# --- Build a sidebar for sessions --- 

# --- Initialize states for the site ---
st.session_state.setdefault(
    'past', 
    []
)
st.session_state.setdefault(
    'generated', 
    []
)

# --- Init the actions of a site ---
def on_input_change():
    user_input = st.session_state.user_input # Retreives registry from `st.text_input()` obj
    if user_input: # Only process if input exists 
        # Store user input 
        st.session_state.past.append(user_input)

        raw_response = query(user_input)

        # Create proper response structure 
        response = {
            'type': 'text', # Default to text response 
            'data': raw_response
        }
        
        # Check for special format 
        if "PLOT_DATA||" in raw_response and "||TEXT_DATA" in raw_response:
            plot_json, text = raw_response.split("||TEXT_DATA||")
            plot_json = plot_json.replace("PLOT_DATA||", "")

            response = {
                'type': 'plot',
                'plot': json.loads(plot_json),
                'text': text
            }
 
        # Store agent response 
        st.session_state.generated.append(response)

        # Clear input field
        st.session_state.user_input = ""

# Clear chat function
def clear_chat():
    st.session_state.past = []
    st.session_state.generated = []

# --- Init the site --- 

# Setting the page tab-title
st.set_page_config(
    page_title="SkillSyncRag",
    page_icon="💻➡️💼",
)

# Titles
# - Main page
st.title("SkillSyncRag💼")

chat_placeholder = st.empty()

# --- Handling message display ---
with chat_placeholder.container():    
    for i, response in enumerate(st.session_state['generated']):  
        # User message              
        message(st.session_state['past'][i], 
                is_user=True, 
                key=f"{i}_user")
        
        # Agent response 
        if response['type'] == 'plot':
            try:
                # Display plot
                fig = pio.from_json(response['plot'])
                st.plotly_chart(fig, use_container_width=True)

                # Display text
                st.markdown(f"```\n{response['text']}\n```")
            except Exception as e:
                st.error(f"Error displaying visualization: {str(e)}")
        else:
            message(response['data'], key=f"{i}")
    
# --- Input Section ---
with st.container():
    st.text_input("User Input:", 
                on_change=on_input_change, 
                key="user_input", # Registers input as `user_input`
                value="", # Add empty value to clear after input 
                )
    st.button("Clear Chat", on_click=clear_chat)

