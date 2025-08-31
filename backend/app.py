from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import fitz  # PyMuPDF
from Model import semantic_search  # your semantic search function
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("key.env")

app = Flask(__name__)
CORS(app)  # Allow all origins for deployment

# Initialize Gemini client with better error handling
try:
    # Use your API key directly or from environment
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBys3qxTdGbxJFwXDPLnaP9VUcOO7fvCxU")
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Test the connection
    model = genai.GenerativeModel('gemini-1.5-flash')
    print(f"✅ Gemini client initialized with model: gemini-1.5-flash")
    client = True
except Exception as e:
    print(f"❌ Failed to initialize Gemini client: {e}")
    client = False

# Gemini model configuration
GEMINI_MODEL = "gemini-1.5-flash"
_AVAILABLE_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

def _get_available_models():
    return _AVAILABLE_MODELS

def _gemini_chat(messages):
    if not client:
        raise RuntimeError("Gemini client not initialized - check your API key")
    
    try:
        # Initialize the model
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Convert messages to Gemini format
        if len(messages) == 1 and messages[0].get("role") == "user":
            # Simple user message
            prompt = messages[0]["content"]
        else:
            # Convert conversation format
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
        
        print(f"🔄 Using Gemini model: {GEMINI_MODEL}")
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=1000,
                temperature=0.7,
            )
        )
        
        print(f"✅ Success with Gemini model: {GEMINI_MODEL}")
        
        # Create response object similar to Groq format
        class GeminiResponse:
            def __init__(self, text, model_name):
                self.choices = [type('obj', (object,), {
                    'message': type('obj', (object,), {'content': text})()
                })()]
                self.model = model_name
        
        return GeminiResponse(response.text, GEMINI_MODEL)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Gemini model failed: {error_msg}")
        raise RuntimeError(f"Gemini API failed: {error_msg}")

# Add health check endpoint for Azure App Service
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "BioSeekAI backend is running",
        "gemini_client": "initialized" if client else "failed",
        "model": GEMINI_MODEL,
        "available_models": _get_available_models()
    })

# Endpoint to list available Gemini models
@app.route("/gemini-models", methods=["GET"])
def gemini_models():
    try:
        if not client:
            return jsonify({"error": "Gemini client not initialized"}), 500
        return jsonify({
            "available_models": _get_available_models(),
            "current_model": GEMINI_MODEL
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Test endpoint to check Gemini API
@app.route("/test-gemini", methods=["GET"])
def test_gemini():
    try:
        if not client:
            return jsonify({"error": "Gemini client not initialized"}), 500
        
        response = _gemini_chat([{"role": "user", "content": "Hello, say 'API working'"}])
        reply = response.choices[0].message.content
        
        return jsonify({
            "status": "success", 
            "reply": reply,
            "model_used": response.model
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "error": str(e)
        }), 500

# -------------------- CHAT --------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    
    if not user_input:
        return jsonify({"reply": "Please provide a message"}), 400

    # Check if Gemini client is initialized
    if not client:
        return jsonify({
            "reply": "❌ Error: Gemini API client not initialized. Check your API key."
        }), 500

    try:
        print(f"🔄 Processing chat request: {user_input[:50]}...")
        
        response = _gemini_chat(messages=[{
            "role": "user", 
            "content": user_input
        }])
        
        reply = response.choices[0].message.content
        model_used = getattr(response, 'model', 'unknown')
        
        print(f"✅ Chat response generated using model: {model_used}")
        
        return jsonify({
            "reply": reply,
            "model": model_used
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Chat error: {error_msg}")
        
        # Return more specific error messages
        if "API key" in error_msg or "unauthorized" in error_msg.lower():
            reply = "❌ Error: Invalid or missing Gemini API key"
        elif "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
            reply = "❌ Error: Gemini API rate limit exceeded. Please try again later."
        elif "blocked" in error_msg.lower():
            reply = "❌ Error: Content was blocked by Gemini safety filters"
        else:
            reply = f"❌ Error: {error_msg}"
        
        return jsonify({"reply": reply}), 500

# -------------------- PDF UPLOAD --------------------
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "".join([page.get_text("text") + "\n" for page in doc])

    if not text.strip():
        return jsonify({"summary": "⚠ No text found in the PDF."})

    try:
        response = _gemini_chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes PDF documents."
                },
                {"role": "user", "content": f"Summarize this document:\n\n{text[:4000]}"},
            ]
        )
        summary = response.choices[0].message.content
    except Exception as e:
        summary = f"❌ Error: {str(e)}"

    print(f"[UPLOAD] File processed: {file.filename}")
    return jsonify({"summary": summary})

# -------------------- SEMANTIC SEARCH --------------------
@app.route("/search", methods=["POST"])
def search():
    start_time = time.time()
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Missing query"}), 400

    try:
        results = semantic_search(user_query=query, top_n=10)

        # Shorten abstracts
        for art in results:
            abstract = art.get("abstract", "")
            sentences = abstract.split(". ")
            art["abstract"] = ". ".join(sentences[:3]) + ("..." if len(sentences) > 3 else "")

        reply_text = (
            f"Found {len(results)} articles. Top title: {results[0]['title']}" 
            if results else "No results found."
        )
        response_time = round(time.time() - start_time, 2)

        print(f"[SEARCH] Query: {query} | Results: {len(results)} | Time: {response_time}s")
        return jsonify({
            "query": query,
            "results": results,
            "reply": reply_text,
            "response_time_sec": response_time
        })

    except Exception as e:
        print(f"[SEARCH ERROR] {str(e)}")
        return jsonify({"error": str(e)}), 500

# -------------------- MAIN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting BioSeekAI backend on port {port}")
    print(f"📊 Using Gemini model: {GEMINI_MODEL}")
    app.run(host="0.0.0.0", port=port, debug=False)

