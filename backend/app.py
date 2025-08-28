from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import fitz  # PyMuPDF
from Model import semantic_search  # your semantic search function
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("key.env")

app = Flask(__name__)
CORS(app)  # Allow all origins for deployment

# Initialize Groq client with better error handling
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print(f"✅ Groq client initialized")
except Exception as e:
    print(f"❌ Failed to initialize Groq client: {e}")
    client = None

# Updated model candidates with correct model names
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_MODEL_CANDIDATES = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192", 
    "mixtral-8x7b-32768",
    "llama3-70b-8192"
]

def _groq_chat(messages):
    if not client:
        raise RuntimeError("Groq client not initialized - check your API key")
    
    last_error = None
    
    for model_name in _MODEL_CANDIDATES:
        try:
            print(f"🔄 Trying model: {model_name}")
            
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1000,  # Add max tokens limit
                temperature=0.7   # Add temperature for consistency
            )
            
            print(f"✅ Success with model: {model_name}")
            return resp
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Model {model_name} failed: {error_msg}")
            last_error = e
            
            # If it's a 404 error, the model doesn't exist - try next one
            if "404" in error_msg or "Not Found" in error_msg:
                continue
            # If it's a rate limit or quota issue, try next model
            elif "rate_limit" in error_msg.lower() or "quota" in error_msg.lower():
                continue
            # For other errors, also try next model
            else:
                continue
    
    # If all models failed, raise the last error with more context
    raise RuntimeError(f"All Groq models failed. Last error: {last_error}")

# Add health check endpoint for Azure App Service
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "BioSeekAI backend is running",
        "groq_client": "initialized" if client else "failed",
        "models": _MODEL_CANDIDATES
    })

# Test endpoint to check Groq API
@app.route("/test-groq", methods=["GET"])
def test_groq():
    try:
        if not client:
            return jsonify({"error": "Groq client not initialized"}), 500
        
        response = _groq_chat([{"role": "user", "content": "Hello, say 'API working'"}])
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

    try:
        # Use ONLY the correct current model - no fallbacks
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ Current working model
            messages=[{"role": "user", "content": user_input}],
            max_tokens=1000,
            temperature=0.7
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"❌ Error: {str(e)}"

    print(f"[CHAT] {user_input} -> {reply}")
    return jsonify({"reply": reply})

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
        response = _groq_chat(
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
    print(f"📊 Available Groq models: {_MODEL_CANDIDATES}")
    app.run(host="0.0.0.0", port=port, debug=False)

