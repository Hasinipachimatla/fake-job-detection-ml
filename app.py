from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('fake_job_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# simple keyword-based explanation
fake_keywords = ["easy money", "work from home", "no experience", "quick cash", "earn fast"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    job_desc = request.form['job_desc']
    
    data = vectorizer.transform([job_desc])
    prediction = model.predict(data)[0]

    result = "Real Job ✅"
    reason = "This job description looks legitimate."

    if prediction == 1:
        result = "Fake Job ❌"
        reasons = []

        for word in fake_keywords:
            if word in job_desc.lower():
                reasons.append(f"Contains suspicious phrase: '{word}'")

        if not reasons:
            reasons.append("General pattern matches fake job postings")

        reason = ", ".join(reasons)

    return render_template('index.html', prediction_text=result, reason_text=reason)

if __name__ == "__main__":
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

    