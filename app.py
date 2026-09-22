from flask import Flask, render_template, request, jsonify
import json
import spacy
import math

app = Flask(__name__)

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Load FAQ data
with open("FAQ.json", "r") as f:
    faqs = json.load(f)

faq_questions = [faq["question"] for faq in faqs]
faq_answers = [faq["answer"] for faq in faqs]

# Preprocess
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Vector + cosine similarity
def text_to_vector(text):
    words = text.split()
    return {word: words.count(word) for word in words}

def cosine_sim(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    sum1 = sum(v**2 for v in vec1.values())
    sum2 = sum(v**2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return numerator / denominator if denominator else 0.0

# Build vectors
processed_questions = [preprocess(q) for q in faq_questions]
faq_vectors = [text_to_vector(q) for q in processed_questions]

# Chatbot answer
def get_answer(user_question):
    user_processed = preprocess(user_question)
    user_vec = text_to_vector(user_processed)
    sims = [cosine_sim(user_vec, faq_vec) for faq_vec in faq_vectors]
    best_index = sims.index(max(sims))
    return faq_answers[best_index]

# Routes
@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json["message"]
    answer = get_answer(user_input)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
