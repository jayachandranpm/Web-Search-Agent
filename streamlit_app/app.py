import streamlit as st
import requests
import json
import re
from datetime import datetime
import logging
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system_frontend.log")
    ]
)
logger = logging.getLogger("rag_system_frontend")

# Set page configuration
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide"
)

# Define the Flask API endpoint
API_ENDPOINT = "http://localhost:5000/api/chat"
QUERY_TRANSFORM_ENDPOINT = "http://localhost:5000/api/transform_query"

# Initialize session state for chat history if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'sources' not in st.session_state:
    st.session_state.sources = []

if 'previous_query' not in st.session_state:
    st.session_state.previous_query = ""

# Title and description
st.title("RAG Assistant")
st.markdown("""
This assistant searches the web for information and provides answers with source citations.
Ask any question and get up-to-date information from multiple sources!
""")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info("""
    This RAG (Retrieval-Augmented Generation) system:
    - Searches the web using DuckDuckGo and Yahoo
    - Retrieves and processes content from top results
    - Generates responses using Google's Gemini 2.0-flash model
    - Maintains conversation memory for contextual follow-ups
    - Provides source citations for transparency
    """)
    
    # Add debug information in a collapsible section
    with st.expander("Debug Information", expanded=False):
        if st.session_state.sources:
            st.write(f"**Last Query Sources:** {len(st.session_state.sources)}")
            for source in st.session_state.sources:
                st.write(f"- [{source.get('id')}] {source.get('title')}")
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        logger.info("Clearing conversation")
        st.session_state.messages = []
        st.session_state.sources = []
        st.session_state.previous_query = ""
        st.experimental_rerun()

def format_response_with_sources(text, sources):
    logger.info("Formatting response with inline citations")
    
    # Create a mapping of source IDs to their URLs for quick lookup
    source_urls = {str(source.get('id', 0)): source.get('url', '') for source in sources}
    
    # First, handle multiple source citations like [SOURCE 4, 5]
    def replace_multiple_sources(match):
        source_ids = match.group(1).split(',')
        replacements = []
        for source_id in source_ids:
            source_id = source_id.strip()
            url = source_urls.get(source_id, '')
            if url:
                replacements.append(f'<sup><a href="{url}" target="_blank">[{source_id}]</a></sup>')
            else:
                replacements.append(f'<sup>[{source_id}]</sup>')
        return ' '.join(replacements)
    
    # Replace multiple source patterns
    text = re.sub(r'\[SOURCE ([0-9, ]+)\]', replace_multiple_sources, text)
    
    # Then replace single source citations
    def replace_single_source(match):
        source_id = match.group(1)
        url = source_urls.get(source_id, '')
        if url:
            return f'<sup><a href="{url}" target="_blank">[{source_id}]</a></sup>'
        return f'<sup>[{source_id}]</sup>'
    
    text = re.sub(r'\[SOURCE ([0-9]+)\]', replace_single_source, text)
    
    # Add cited sources to session state
    st.session_state.sources = sources
    
    # Then add a "Sources:" section at the end with all references
    if sources:
        text += "\n\n---\n\n**Sources:**\n"
        for source in sources:
            source_id = source.get('id', 0)
            source_title = source.get('title', f'Source {source_id}')
            source_url = source.get('url', '')
            if source_url:
                text += f"\n[{source_id}] <a href='{source_url}' target='_blank'>{source_title}</a>"
            else:
                text += f"\n[{source_id}] {source_title}"
    
    return text

def transform_followup_query_with_gemini(current_query, previous_query):
    logger.info(f"Transforming follow-up query. Current: '{current_query}', Previous: '{previous_query}'")
    
    try:
        # Call the API endpoint to transform the query
        response = requests.post(
            QUERY_TRANSFORM_ENDPOINT,
            json={"current_query": current_query, "previous_query": previous_query},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            transformed_query = result.get("transformed_query", current_query)
            logger.info(f"Query transformed to: '{transformed_query}'")
            return transformed_query
        else:
            logger.warning(f"Query transformation failed with status code {response.status_code}")
            return f"{previous_query} {current_query}"
    except Exception as e:
        logger.error(f"Error transforming query: {str(e)}")
        return f"{previous_query} {current_query}"

def check_api_health():
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# API health check
api_status = check_api_health()
if not api_status:
    st.error("⚠️ Backend API is not accessible. Please ensure the Flask backend is running.")

# Chat input
user_query = st.chat_input("Ask a question...", disabled=not api_status)

# Process user input
if user_query:
    logger.info(f"Received user query: '{user_query}'")
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Display assistant thinking
    with st.chat_message("assistant"):
        start_time = time.time()
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🔍 _Searching the web and generating response..._")
        
        # Determine if this is a follow-up question
        is_followup = len(st.session_state.messages) > 1 and st.session_state.previous_query
        logger.info(f"Is this a follow-up question? {is_followup}")
        
        # Transform the query if it's a follow-up using Gemini model
        transformed_query = user_query
        if is_followup:
            transformed_query = transform_followup_query_with_gemini(user_query, st.session_state.previous_query)
            # Show a small debug message about transformation
            if transformed_query != user_query:
                st.caption(f"Transformed query: '{transformed_query}'")
                logger.info(f"Query transformed from '{user_query}' to '{transformed_query}'")
        
        # Initialize progress bar for better visual feedback
        progress_bar = st.progress(0)
        
        # Send request to Flask backend
        try:
            logger.info(f"Sending request to backend API with query: '{transformed_query}'")
            
            # Update progress to show that the request is being sent
            progress_bar.progress(10)
            
            # Make the API request
            response = requests.post(
                API_ENDPOINT,
                json={
                    "query": transformed_query, 
                    "is_followup": is_followup, 
                    "original_query": user_query
                },
                timeout=120  # Allow up to 2 minutes for processing
            )
            
            # Update progress to show that we've received a response
            progress_bar.progress(90)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "Sorry, I couldn't generate a response.")
                sources = result.get("sources", [])
                processing_time = result.get("processing_time", 0)
                
                logger.info(f"Received response from API with {len(sources)} sources in {processing_time:.2f} seconds")
                
                # Check if any sources were returned
                if not sources:
                    logger.warning("No sources were returned with the response")
                    # If no sources but we have an answer, still show it
                    if answer:
                        formatted_answer = answer
                    else:
                        formatted_answer = "I searched the web but couldn't find relevant information for your query. Please try rephrasing your question or ask something else."
                else:
                    # Format the response with proper source citations
                    formatted_answer = format_response_with_sources(answer, sources)
                
                # Add assistant's response to chat history
                st.session_state.messages.append({"role": "assistant", "content": formatted_answer})
                
                # Update previous query for next follow-up
                st.session_state.previous_query = transformed_query
                
                # Display assistant's response
                thinking_placeholder.markdown(formatted_answer, unsafe_allow_html=True)
                
                # Show processing time
                end_time = time.time()
                total_time = end_time - start_time
                st.caption(f"Response generated in {total_time:.2f} seconds")
                
                # Log the successful response
                logger.info(f"Response displayed successfully in {total_time:.2f} seconds")
            else:
                logger.error(f"API returned error status code: {response.status_code}")
                error_message = f"Error: Received status code {response.status_code} from API"
                thinking_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                
        except Exception as e:
            logger.error(f"Error connecting to backend: {str(e)}")
            error_message = f"Error connecting to backend: {str(e)}"
            thinking_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
        
        # Complete the progress bar
        progress_bar.progress(100)
        # Remove the progress bar after a short delay
        time.sleep(0.5)
        progress_bar.empty()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Powered by Gemini 2.0 | Web search via DuckDuckGo & Yahoo</p>
    <p><small>Last updated: May 2025</small></p>
</div>
""", unsafe_allow_html=True)
