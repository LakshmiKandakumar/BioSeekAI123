from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import biomed_search as Model

app = Flask(__name__, static_folder="build")
CORS(app)

# Serve React frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# Search endpoint
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify([])
    return jsonify(Model.semantic_search(query))

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"reply": "⚠ Empty message."})
    client = Model.get_client()
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are Clario AI, a biomedical assistant."},
            {"role": "user", "content": message}
        ],
        max_tokens=300,
        temperature=0.5
    )
    reply = response["choices"][0]["message"]["content"].strip() if "choices" in response else "⚠ No response."
    return jsonify({"reply": reply})

# File upload & summarize
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"summary": "⚠ No file uploaded."})
    file = request.files["file"]
    if not file.filename:
        return jsonify({"summary": "⚠ Empty filename."})
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)
    with open(temp_path, "r", errors="ignore") as f:
        text = f.read()
    client = Model.get_client()
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a biomedical assistant. Summarize uploaded research content."},
            {"role": "user", "content": text[:3000]}
        ],
        max_tokens=250,
        temperature=0.5
    )
    summary = response["choices"][0]["message"]["content"].strip() if "choices" in response else "⚠ No summary."
    return jsonify({"summary": summary})

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
