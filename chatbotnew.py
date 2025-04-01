from flask import Flask, render_template, request, jsonify
import json
import spacy
import random

app = Flask(__name__)

# Load NLP model (spaCy's medium model for better word embeddings)
nlp = spacy.load("en_core_web_md")

# Load knowledge base from JSON file
def load_knowledge_base(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

knowledge_base = load_knowledge_base('shrimp_knowledge.json')

# Function to find best matching intent using NLP-based similarity
def find_best_intent(user_message):
    user_doc = nlp(user_message.lower())
    best_intent = None
    best_similarity = 0.0
    
    for intent in knowledge_base['intents']:
        for pattern in intent['patterns']:
            pattern_doc = nlp(pattern.lower())
            similarity = user_doc.similarity(pattern_doc)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_intent = intent
    
    return best_intent if best_similarity > 0.6 else None  # Only accept good matches

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.json['message']
    matched_intent = find_best_intent(user_message)
    
    if matched_intent:
        response = random.choice(matched_intent['responses'])
    else:
        response = random.choice(knowledge_base['default_responses'])
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)