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
# app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

CORS(app)  # Allow all origins for deployment

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------- CHAT --------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"reply": "Please provide a message"}), 400

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": user_input}],
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
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes PDF documents."
                },
                {"role": "user", "content": f"Summarize this document:\n\n{text[:4000]}"},
            ],
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
    # Use 0.0.0.0 to allow external access when deployed
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)

