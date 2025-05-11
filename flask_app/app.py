from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import json
import re
import logging
from datetime import datetime
from bs4 import BeautifulSoup
import google.generativeai as genai
from urllib.parse import quote_plus
import time
import concurrent.futures
import hashlib
from dotenv import load_dotenv
import random # Added for User-Agent rotation

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system_backend.log")
    ]
)
logger = logging.getLogger("rag_system_backend")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set")
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Initialize Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Global variables
MAX_SEARCH_RESULTS = 5
CONVERSATION_MEMORY = {}  # Store conversation history by session ID
MAX_CONTENT_LENGTH = 15000  # Maximum content length for context (characters)
SEARCH_TIMEOUT = 30
CONTENT_FETCH_TIMEOUT = 30  # Increased timeout for content fetching

# User agents for web requests
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"
]

# Search engines
SEARCH_ENGINES = {
    "duckduckgo": "https://duckduckgo.com/html/?q={}",
    "yahoo": "https://search.yahoo.com/search?p={}"
}

# Log startup information
logger.info("Environment variables loaded")
logger.info("API key configured successfully")

#------------------------
# Helper Functions
#------------------------

def get_session_id(request):
    """Generate a unique session ID based on client IP and user agent"""
    client_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', '')
    unique_string = f"{client_ip}_{user_agent}"
    session_id = hashlib.md5(unique_string.encode()).hexdigest()
    logger.debug(f"Generated session ID: {session_id}")
    return session_id

def extract_text_from_html(html_content):
    """Extract clean text from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style", "nav", "footer", "aside"]): # Added common non-content tags
        script.extract()
    
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def search_web(query):
    """Search the web using multiple search engines concurrently"""
    quoted_query = quote_plus(query)
    all_results = []
    
    logger.info(f"Starting web search for: '{query}'")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_engine = {
            executor.submit(search_with_engine, engine, quoted_query): engine
            for engine in SEARCH_ENGINES
        }
        
        for future in concurrent.futures.as_completed(future_to_engine):
            engine = future_to_engine[future]
            try:
                results = future.result()
                if results: # Only extend if results is not None and not empty
                    logger.info(f"Found {len(results)} results from {engine}")
                    all_results.extend(results)
                else:
                    logger.info(f"No results from {engine}")
            except Exception as e:
                logger.error(f"Error searching with {engine}: {str(e)}")
    
    unique_results = []
    seen_urls = set()
    
    for result in all_results:
        url = result.get('url')
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    logger.info(f"Found {len(unique_results)} unique results across all search engines")
    
    final_results = unique_results[:MAX_SEARCH_RESULTS]
    # Assign sequential IDs *after* deduplication and selection
    for i, res in enumerate(final_results):
        res['id'] = i + 1 
    
    logger.info(f"Using top {len(final_results)} results for processing with assigned IDs")
    return final_results

def search_with_engine(engine, quoted_query):
    """Search with a specific search engine"""
    url = SEARCH_ENGINES[engine].format(quoted_query)
    headers = {'User-Agent': random.choice(USER_AGENTS)} # Rotate user agent for search itself
    
    logger.info(f"Searching {engine} with URL: {url} using User-Agent: {headers['User-Agent']}")
    
    try:
        response = requests.get(url, headers=headers, timeout=SEARCH_TIMEOUT)
        
        if response.status_code == 200:
            logger.info(f"Successful response from {engine}")
            parsed_items = []
            if engine == "duckduckgo":
                parsed_items = parse_duckduckgo_results(response.text)
            elif engine == "yahoo":
                parsed_items = parse_yahoo_results(response.text)
            
            if not parsed_items:
                logger.warning(f"Parsed 0 results from {engine} (status 200). HTML might've changed or no results on page.")
            else:
                logger.info(f"Parsed {len(parsed_items)} results from {engine}")
            return parsed_items
        else:
            logger.warning(f"{engine} search returned status code {response.status_code}")
            return []
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout searching {engine} after {SEARCH_TIMEOUT} seconds for query: {quoted_query}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException searching {engine} for query {quoted_query}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Generic error searching {engine} for query {quoted_query}: {str(e)}")
        return []

def parse_duckduckgo_results(html_content):
    """Parse DuckDuckGo search results"""
    results = []
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for result_div in soup.select('div.result, div.results_links_deep'): # Common selectors
        try:
            title_elem = result_div.select_one('h2.result__title a.result__a, a.deep-result__title')
            url_elem = result_div.select_one('a.result__url, span.result__url') # URL might be in span or a
            snippet_elem = result_div.select_one('a.result__snippet, div.result__snippet')

            if title_elem:
                title = title_elem.get_text(strip=True)
                
                # Get URL. It might be directly in title_elem's href or in url_elem
                url = title_elem.get('href')
                if not url and url_elem: # Fallback if not in title_elem
                     url = url_elem.get('href', url_elem.get_text(strip=True))


                if url:
                    # Extract the actual URL from DuckDuckGo's redirect
                    match = re.search(r'uddg=([^&]+)', url)
                    if match:
                        url = requests.utils.unquote(match.group(1))
                    
                    # Ensure URL starts with http/https
                    if not url.startswith('http'):
                        if url.startswith("//"):
                            url = "https:" + url
                        elif url_elem and url_elem.get_text(strip=True).startswith("http"): # Check text if href is relative
                            url = url_elem.get_text(strip=True)
                        else: # Could be relative path, try to skip or make absolute if base known
                            logger.debug(f"Skipping DuckDuckGo result with potentially relative URL: {url}")
                            continue


                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                if title and url and url.startswith('http') and not any(blocked in url.lower() for blocked in ['.pdf', '.doc', '.xls', '.ppt', '.zip']):
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet
                        # ID is assigned later
                    })
                    logger.debug(f"Added DuckDuckGo result: {title} - {url}")
            
        except Exception as e:
            logger.error(f"Error parsing a DuckDuckGo result item: {str(e)}")
    
    return results

def parse_yahoo_results(html_content):
    """Parse Yahoo search results"""
    results = []
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for result_li in soup.select('li div.algo'): # Common Yahoo result item
        try:
            title_elem = result_li.select_one('h3 a')
            snippet_elem = result_li.select_one('div.compText p, span.fc-mrstandard') # Yahoo uses different snippet classes
            
            if title_elem:
                title = title_elem.get_text(strip=True)
                url = title_elem.get('href')
                
                if url:
                    # Extract the actual URL from Yahoo's redirect
                    if '/RU=' in url and '/RK=' in url: # Common Yahoo redirect pattern
                        match = re.search(r'/RU=([^/]+)', url)
                        if match:
                            url = requests.utils.unquote(match.group(1))
                    
                    # Ensure URL starts with http/https
                    if not url.startswith('http'):
                        logger.debug(f"Skipping Yahoo result with non-http URL: {url}")
                        continue

                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                if title and url and url.startswith('http') and not any(blocked in url.lower() for blocked in ['.pdf', '.doc', '.xls', '.ppt', '.zip']):
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet
                        # ID is assigned later
                    })
                    logger.debug(f"Added Yahoo result: {title} - {url}")
        
        except Exception as e:
            logger.error(f"Error parsing a Yahoo result item: {str(e)}")
            
    return results

def fetch_content(url):
    """Fetch and extract content from a URL"""
    logger.info(f"Fetching content from URL: {url}")
    
    try:
        user_agent = random.choice(USER_AGENTS)
        headers = {'User-Agent': user_agent}
        logger.debug(f"Using User-Agent: {user_agent} for {url}")

        response = requests.get(url, headers=headers, timeout=CONTENT_FETCH_TIMEOUT, allow_redirects=True)
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type:
                text_content = extract_text_from_html(response.text)
                
                if not text_content or len(text_content) < 50: # Check for minimal content length
                    logger.warning(f"Extracted text from {url} is too short (length: {len(text_content) if text_content else 0}). Discarding.")
                    return None

                trimmed_content = text_content[:MAX_CONTENT_LENGTH]
                logger.info(f"Extracted {len(text_content)} chars, trimmed to {len(trimmed_content)} chars from {url}")
                return trimmed_content
            else:
                logger.info(f"Skipping non-HTML content from {url}: {content_type}")
                return None
        else:
            logger.warning(f"Failed to fetch {url}, status code: {response.status_code}")
            return None
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error fetching content from {url} after {CONTENT_FETCH_TIMEOUT} seconds.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching content from {url}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Generic error fetching content from {url}: {str(e)}")
        return None

def format_context_with_sources(sources_for_context):
    """Format search results into context for the model with sources.
    Uses full content if available, otherwise uses snippet if marked."""
    context_parts = []
    
    for result in sources_for_context:
        source_id = result.get('id') # Should always have an ID now
        title = result.get('title', f'Source {source_id}')
        url = result.get('url', '')
        
        content_to_use = result.get('content')
        content_source_type = "full content"

        if not content_to_use and result.get('use_snippet_for_context', False):
            content_to_use = result.get('snippet')
            content_source_type = "snippet"
        
        if content_to_use:
            context_parts.append(f"[SOURCE {source_id}] Title: {title}\nURL: {url}\n(Source type: {content_source_type})\nContent:\n{content_to_use}\n")
    
    context = "\n---\n".join(context_parts) # Separate sources a bit more clearly
    if not context_parts:
        logger.warning("Formatted context is empty. No usable content or snippets found from sources.")
        return "No usable content found from search results."

    logger.info(f"Created context with {len(context_parts)} sources (mix of content/snippets), total size: {len(context)} chars")
    return context

def generate_response(query, context, conversation_history=None):
    """Generate a response using Gemini model with RAG context"""
    logger.info(f"Generating response for query: '{query}'")
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt_template = f"""
You are a helpful and informative AI assistant. You have been provided with web search results related to the user's query.
User's Query: "{query}"
{{history_section}}
Search Results Provided:
---
{context}
---
Instructions:
1. Based *only* on the search results and conversation history (if any), provide a comprehensive answer to the user's query.
2. Cite your sources using [SOURCE X] notation inline after the information from that source (e.g., "The sky is blue [SOURCE 1].").
3. If information comes from multiple sources, cite all relevant ones (e.g., "Water boils at 100°C [SOURCE 1, 2].").
4. Synthesize information from multiple sources when appropriate to provide a complete picture.
5. If the provided search results do not contain enough information to answer the query, clearly state that. Do not invent information.
6. Focus on directly answering the user's current query, considering the conversational context if available.
7. Be objective and stick to the facts presented in the sources.

User Query: {query}
Response:
"""
        history_section_text = ""
        if conversation_history:
            logger.info("Including conversation history in prompt")
            history_section_text = f"Conversation History:\n{conversation_history}\n"
        
        final_prompt = prompt_template.format(history_section=history_section_text)
        
        logger.info(f"Generated prompt with length: {len(final_prompt)} chars")
        # logger.debug(f"Full prompt to Gemini:\n{final_prompt}") # Potentially very long
        
        response = model.generate_content(final_prompt)
        
        # Log safety ratings if available (useful for debugging refusals)
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            logger.warning(f"Prompt blocked by Gemini. Reason: {response.prompt_feedback.block_reason}")
            return f"I'm sorry, I couldn't generate a response due to content safety reasons related to the prompt. (Reason: {response.prompt_feedback.block_reason})"
        if not response.candidates or not response.candidates[0].content.parts:
             logger.warning("Gemini response was empty or malformed.")
             if response.candidates and response.candidates[0].finish_reason:
                 logger.warning(f"Gemini finish reason: {response.candidates[0].finish_reason}")
                 if response.candidates[0].finish_reason.name == "SAFETY":
                      return "I'm sorry, I couldn't generate a response due to content safety filters regarding the generated content."
             return "I'm sorry, I received an empty or malformed response from the AI model."


        logger.info(f"Received response from Gemini API.")
        return response.text
    
    except Exception as e:
        logger.error(f"Error generating response with Gemini: {str(e)}")
        return f"I'm sorry, but I encountered an error while generating a response. Error: {str(e)}"

def transform_followup_query(current_query, previous_query):
    """Transform a follow-up query into a complete, standalone query using Gemini"""
    logger.info(f"Transforming follow-up query: '{current_query}' with previous: '{previous_query}'")
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
Given the previous query and the current follow-up query, rewrite the follow-up query as a standalone, complete question that incorporates necessary context from the previous query. The goal is to make the new query understandable and searchable on its own.

Previous query: "{previous_query}"
Follow-up query: "{current_query}"

Standalone query:
"""
        response = model.generate_content(prompt)
        transformed_query = response.text.strip().strip('"')
        logger.info(f"Transformed query: '{transformed_query}'")
        return transformed_query
        
    except Exception as e:
        logger.error(f"Error transforming query with Gemini: {str(e)}")
        fallback_query = f"{previous_query} {current_query}"
        logger.info(f"Falling back to simple concatenation for query transformation: '{fallback_query}'")
        return fallback_query

#------------------------
# API Endpoints
#------------------------

@app.route('/api/chat', methods=['POST'])
def chat():
    start_time = time.time()
    try:
        data = request.json
        query = data.get('query', '') # This is the transformed query if it was a follow-up
        is_followup = data.get('is_followup', False)
        original_query = data.get('original_query', query) # This is what user typed
        
        if not query:
            logger.warning("Received empty query")
            return jsonify({"error": "Query is required"}), 400
        
        logger.info(f"Processing query: '{query}' (Original: '{original_query}', Follow-up: {is_followup})")
        
        session_id = get_session_id(request)
        conversation_history = CONVERSATION_MEMORY.get(session_id, [])
        
        search_results = search_web(query)
        if not search_results:
            logger.warning(f"No search results found by search_web for query: '{query}'")
            return jsonify({
                "answer": "I'm sorry, but I couldn't find any relevant web pages for your query. Please try rephrasing or a different topic.",
                "sources": [],
                "processing_time": time.time() - start_time
            })

        logger.info(f"Fetching content for {len(search_results)} search results...")
        content_fetch_start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_SEARCH_RESULTS) as executor:
            future_to_result = {executor.submit(fetch_content, res['url']): res for res in search_results}
            for future in concurrent.futures.as_completed(future_to_result):
                res = future_to_result[future]
                try:
                    content = future.result()
                    if content:
                        res['content'] = content
                except Exception as exc:
                    logger.error(f"Error fetching/processing content for {res['url']}: {exc}")
        logger.info(f"Content fetching completed in {time.time() - content_fetch_start_time:.2f}s")

        sources_for_llm_context = []
        for res in search_results:
            if res.get('content'):
                sources_for_llm_context.append(res)
            elif res.get('snippet'): # Fallback to snippet
                res['use_snippet_for_context'] = True
                sources_for_llm_context.append(res)
        
        if not sources_for_llm_context:
            logger.warning(f"No usable content or snippets extracted from {len(search_results)} search results for query: '{query}'.")
            return jsonify({
                "answer": "I found search results, but I couldn't extract useful information or snippets from them. This might be due to issues accessing or parsing the content of the web pages. Please try rephrasing your question.",
                "sources": [], # Optionally, could return search_results (title/URL) here for user to check
                "processing_time": time.time() - start_time
            })
        
        logger.info(f"Using {len(sources_for_llm_context)} sources for LLM context generation.")
        context_for_llm = format_context_with_sources(sources_for_llm_context)
        
        if context_for_llm == "No usable content found from search results.":
             logger.error("Context formatting resulted in 'No usable content', though sources_for_llm_context was not empty.")
             return jsonify({
                "answer": "I encountered an internal issue trying to prepare information from search results. Please try again.",
                "sources": [], "processing_time": time.time() - start_time
            })

        history_text = None
        if conversation_history and is_followup:
            history_parts = []
            # Simple history: User: Q1, Assistant: A1, User: Q2 ...
            for i, msg_content in enumerate(conversation_history):
                role = "User" if i % 2 == 0 else "Assistant"
                history_parts.append(f"{role}: {msg_content}")
            history_text = "\n".join(history_parts)
        
        ai_answer = generate_response(original_query, context_for_llm, history_text) # Pass original_query for LLM's understanding of user's intent
        
        # Update conversation memory with the original user query and AI's answer
        current_convo = [original_query, ai_answer]
        CONVERSATION_MEMORY[session_id] = conversation_history + current_convo
        if len(CONVERSATION_MEMORY[session_id]) > 10: # Limit history to last 5 pairs
            CONVERSATION_MEMORY[session_id] = CONVERSATION_MEMORY[session_id][-10:]
        logger.info(f"Updated conversation memory for session {session_id}, new length: {len(CONVERSATION_MEMORY[session_id])}")
        
        # Sources for frontend display (IDs must match those used in context_for_llm)
        display_sources = [{
            "id": r.get('id'),
            "title": r.get('title'),
            "url": r.get('url'),
            "snippet": r.get('snippet', '') 
        } for r in sources_for_llm_context]
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed successfully in {processing_time:.2f} seconds. Returning {len(display_sources)} sources.")
        
        return jsonify({
            "answer": ai_answer,
            "sources": display_sources,
            "processing_time": processing_time
        })
    
    except Exception as e:
        logger.error(f"Unhandled error in /api/chat: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred."}), 500

@app.route('/api/transform_query', methods=['POST'])
def transform_query_endpoint():
    try:
        data = request.json
        current_query = data.get('current_query', '')
        previous_query = data.get('previous_query', '')
        
        if not current_query: # previous_query can be empty for first turn
            logger.warning("Transform query called with empty current_query.")
            return jsonify({"transformed_query": ""}) # Return empty if current is empty
        if not previous_query: # Not a follow-up, or first turn of a follow-up chain
            logger.info("No previous query provided for transformation, returning current query as-is.")
            return jsonify({"transformed_query": current_query})
            
        logger.info(f"Transforming query - Previous: '{previous_query}', Current: '{current_query}'")
        transformed_query = transform_followup_query(current_query, previous_query)
        return jsonify({"transformed_query": transformed_query})
        
    except Exception as e:
        logger.error(f"Error in /api/transform_query: {str(e)}", exc_info=True)
        # Fallback to current_query if transformation fails
        return jsonify({"error": str(e), "transformed_query": request.json.get('current_query', '')}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "search_engines": list(SEARCH_ENGINES.keys()),
            "max_results": MAX_SEARCH_RESULTS,
            "model": "gemini-2.0-flash"
        }
    })

if __name__ == '__main__':
    logger.info("Starting RAG System Backend on port 5000")
    # For production, use a proper WSGI server like Gunicorn or Waitress
    # app.run(host='0.0.0.0', port=5000, debug=False) # debug=False for production
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True for development as in original
