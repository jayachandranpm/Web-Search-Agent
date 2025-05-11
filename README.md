# Web-Search-Agent

# LLM RAG System

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that leverages Large Language Models (LLMs) to provide informative, context-aware answers. The system searches the web for relevant information, processes the content, and uses Google's Gemini model to generate responses with source citations. It features a Streamlit frontend for user interaction and a Flask backend to handle the logic.

## Features

-   **Advanced LLM:** Utilizes Google's Gemini 2.0-flash model for sophisticated language understanding and response generation.
-   **Comprehensive Web Search:** Employs DuckDuckGo and Yahoo search engines to retrieve diverse and up-to-date information.
-   **Conversational Context:** Maintains conversation memory for up to 10 recent messages (5 question-answer pairs) to allow for follow-up questions.
-   **Source Citations:** Provides inline and summarized source citations for all retrieved information, ensuring transparency and verifiability.
-   **Dynamic Query Transformation:** Follow-up questions are intelligently transformed into standalone queries for more effective searching.
-   **User-Friendly Interface:** Simple and intuitive Streamlit-based UI for easy interaction.

## System Architecture & Process Flow

1.  **User Input (Streamlit Frontend)**: The user enters a query into the Streamlit web interface.
2.  **API Call (Frontend to Backend)**: The query is sent to the Flask backend API.
3.  **Query Transformation (Backend)**: If it's a follow-up question, the backend uses the LLM to transform it into a standalone query using the conversation history.
4.  **Web Search & Content Scraping (Backend)**:
    *   The Flask backend searches the web using DuckDuckGo and Yahoo with the (potentially transformed) query.
    *   It retrieves top relevant article URLs and scrapes their content (textual data like headings and paragraphs).
5.  **Content Processing & Context Building (Backend)**: The scraped text from multiple sources is cleaned, truncated, and formatted into a context document, along with source identifiers.
6.  **LLM Response Generation (Backend)**:
    *   The processed context and the original user query (plus conversation history) are passed to the Gemini LLM via its API.
    *   The LLM generates a contextual answer, incorporating information from the provided sources and citing them.
7.  **Response to Frontend (Backend to Frontend)**: The Flask backend sends the LLM's generated answer and the list of sources back to the Streamlit frontend.
8.  **Display to User (Streamlit Frontend)**: The frontend displays the answer and clickable source links to the user.

## Tech Stack

-   **Frontend:** Streamlit
-   **Backend:** Flask, Python
-   **LLM:** Google Gemini 2.0-flash (via `google-generativeai` SDK)
-   **Web Scraping/Parsing:** `requests`, `beautifulsoup4`
-   **Search:** Custom integration with DuckDuckGo and Yahoo HTML search.
-   **Environment Management:** `python-dotenv`

## Prerequisites

-   Python 3.8 or higher
-   Access to the internet for web searches and LLM API calls.
-   A Google API Key with the Gemini API enabled.

## Setup Instructions

### Step 1: Clone or Download the Repository

If you have a Git repository URL:
```bash
git clone https://github.com/jayachandranpm/Web-Search-Agent.git # Replace with your actual repo URL
cd project naem # Replace with your project directory name
```
Or, if you downloaded the files, extract them to a project directory.

### Step 2: Set Up a Virtual Environment

It's highly recommended to use a virtual environment.

#### Using `venv`:
```bash
python -m venv env
source env/bin/activate  # On Linux/macOS
# env\Scripts\activate   # On Windows
```

#### Using `conda`:
```bash
conda create --name rag_env python=3.9 # Or your preferred Python version >= 3.8
conda activate rag_env
```

### Step 3: Install Requirements

Ensure your `requirements.txt` file is in the project root and lists all necessary packages (e.g., `streamlit`, `flask`, `requests`, `beautifulsoup4`, `google-generativeai`, `python-dotenv`, `flask-cors`).
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory of the project. Add your Google API Key to this file:

```env
# .env
GOOGLE_API_KEY="your_google_api_key_here"
```
**Important:** Replace `"your_google_api_key_here"` with your actual Google API key that has the Generative Language API (Gemini) enabled.

### Step 5: Organize Project Files

Ensure your project files are organized as follows. You might need to rename your initial `app.py` files.

```
llm_rag_system/
├── flask_app/
│   └── app.py        # Your Flask backend code
├── streamlit_app/
│   └── app.py        # Your Streamlit frontend code
├── .env              # For API keys
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
*   Place your Flask backend code into `flask_app/app.py`.
*   Place your Streamlit frontend code into `streamlit_app/app.py`.

### Step 6: Run the Flask Backend

Navigate to the `flask_app` directory and start the Flask server:

```bash
cd flask_app
python app.py
```
The backend server should start, typically on `http://localhost:5000`. Check the console output.

### Step 7: Run the Streamlit Frontend

Open a **new terminal window/tab**. Activate your virtual environment in this new terminal as well. Then, navigate to the `streamlit_app` directory and run the Streamlit app:

```bash
cd streamlit_app  # If you are in project_root, otherwise navigate accordingly
streamlit run app.py
```
The Streamlit app should open in your web browser, typically at `http://localhost:8501`.

### Step 8: Interact with the Application

Open `http://localhost:8501` (or the URL Streamlit provides) in your web browser. You can now ask questions and interact with the RAG assistant.

## Project Structure

```
llm_rag_system/
├── flask_app/
│   └── app.py            # Backend Flask application logic, API endpoints
├── streamlit_app/
│   └── app.py            # Frontend Streamlit application UI and logic
├── .env                  # Environment variables (e.g., API keys) - NOT version controlled
├── requirements.txt      # Project dependencies
└── README.md             # This documentation file
```

