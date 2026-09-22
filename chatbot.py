import json
import spacy
import math

# -----------------------------
# 1. Load spaCy model
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# 2. Load FAQ.json
# -----------------------------
with open("FAQ.json", "r") as f:
    faqs = json.load(f)

faq_questions = [faq["question"] for faq in faqs]
faq_answers = [faq["answer"] for faq in faqs]

# -----------------------------
# 3. Preprocessing function
# -----------------------------
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# -----------------------------
# 4. Pure-Python vector + cosine similarity
# -----------------------------
def text_to_vector(text):
    words = text.split()
    return {word: words.count(word) for word in words}

def cosine_sim(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)

    sum1 = sum(v**2 for v in vec1.values())
    sum2 = sum(v**2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return numerator / denominator

# -----------------------------
# 5. Build vectors for FAQ questions
# -----------------------------
processed_questions = [preprocess(q) for q in faq_questions]
faq_vectors = [text_to_vector(q) for q in processed_questions]

# -----------------------------
# 6. Answer matching function
# -----------------------------
def get_answer(user_question):
    user_processed = preprocess(user_question)
    user_vec = text_to_vector(user_processed)

    sims = [cosine_sim(user_vec, faq_vec) for faq_vec in faq_vectors]
    best_index = sims.index(max(sims))
    return faq_answers[best_index]

# -----------------------------
# 7. Chat loop
# -----------------------------
print("AI Chatbot Ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = get_answer(user_input)
    print("Bot:", response)
