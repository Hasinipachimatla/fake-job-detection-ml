from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open('fake_job_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['job_description']
    vector = vectorizer.transform([input_text])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        result = "Fake Job 🚨"
    else:
        result = "Real Job 💼"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
    