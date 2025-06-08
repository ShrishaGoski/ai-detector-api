from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
detector = pipeline("text-classification", model="roberta-base-openai-detector")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.json["text"]
    result = detector(text[:512])[0]

    return jsonify({
        "ai_generated": result["label"],
        "confidence": round(result["score"] * 100, 2),
        "references": [f"https://www.google.com/search?q={'+'.join(text.split()[:5])}"]
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
