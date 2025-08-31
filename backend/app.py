from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile

# Lazy import biomed_search to avoid heavy load on cold start
biomed_search = None

def get_biomed_search():
    global biomed_search
    if biomed_search is None:
        import biomed_search
    return biomed_search

# ---------------- FLASK APP ----------------
app = Flask(__name__, static_folder="build")  # serve React build
CORS(app)  # allow React frontend to call Flask

# ---------------- SERVE REACT ----------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    """Serve React frontend"""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# ---------------- SEARCH ----------------
@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        if not query:
            return jsonify([])

        search_module = get_biomed_search()
        results = search_module.semantic_search(query)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- CHAT ----------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"reply": "⚠ Empty message."})

        search_module = get_biomed_search()
        client = search_module.get_llama_client()
        system_msg = "You are Clario AI, a biomedical assistant."
        response = client.chat_completion(
            model="meta-llama/3b-instruct",  # lighter 3B model
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": message}
            ],
            max_tokens=300,
            temperature=0.5
        )
        reply = response["choices"][0]["message"]["content"].strip() if "choices" in response else "⚠ No response."
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"⚠ Error: {str(e)}"}), 500

# ---------------- FILE UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"summary": "⚠ No file uploaded."})

        file = request.files["file"]
        if not file.filename:
            return jsonify({"summary": "⚠ Empty filename."})

        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        with open(temp_path, "r", errors="ignore") as f:
            text = f.read()

        search_module = get_biomed_search()
        client = search_module.get_llama_client()
        system_msg = "You are a biomedical assistant. Summarize uploaded research content."
        response = client.chat_completion(
            model="meta-llama/3b-instruct",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text[:3000]}  # truncate long files
            ],
            max_tokens=250,
            temperature=0.5
        )
        summary = response["choices"][0]["message"]["content"].strip() if "choices" in response else "⚠ No summary."
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"summary": f"⚠ Error processing file: {str(e)}"}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
