# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import Model     # ✅ import your model.py
import os
import tempfile

# ---------------- FLASK APP ----------------
app = Flask(__name__, static_folder="build")  # serve React build
CORS(app)  # allow React frontend to call Flask

# ---------------- SERVE REACT ----------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    """
    Serve React frontend
    """
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
            return jsonify([])   # return empty list directly

        # semantic_search ALREADY returns a list of dict articles
        sorted_articles = Model.semantic_search(query)
        return jsonify(sorted_articles)   # ✅ return plain list
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

        # use your Llama client to generate a reply
        response = Model.client.chat_completion(
            messages=[
                {"role": "system", "content": "You are Clario AI, a biomedical assistant."},
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

        # save temporarily
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        # simple text read (can extend for PDF, DOCX, etc.)
        with open(temp_path, "r", errors="ignore") as f:
            text = f.read()

        # summarize using Llama client
        response = Model.client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a biomedical assistant. Summarize uploaded research content."},
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
    # Use port from environment variable if deploying
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
