import os
import sqlite3
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import openai

app = Flask(__name__)

DB_NAME = "health.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  sleep INTEGER,
                  steps INTEGER,
                  diet TEXT,
                  exercise TEXT,
                  glucose REAL,
                  blood_pressure REAL,
                  bmi REAL,
                  age INTEGER)''')
    conn.commit()
    conn.close()

init_db()

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # binary prediction
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

X_dummy = np.random.rand(500, 5)
y_dummy = np.random.randint(0, 2, 500)
scaler = StandardScaler()
X_dummy = scaler.fit_transform(X_dummy)

model = build_model(X_dummy.shape[1])
model.fit(X_dummy, y_dummy, epochs=10, batch_size=16, verbose=0)

openai.api_key = os.getenv("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")

def gpt_explanation(user_data, risk_score):
    prompt = f"""
    User health summary:
    Sleep: {user_data['sleep']} hours
    Steps: {user_data['steps']}
    Diet: {user_data['diet']}
    Exercise: {user_data['exercise']}
    Glucose: {user_data['glucose']}
    Blood Pressure: {user_data['blood_pressure']}
    BMI: {user_data['bmi']}
    Age: {user_data['age']}

    Neural network risk score: {risk_score}

    Please explain this to the user in simple language,
    give lifestyle advice (diet, exercise, sleep),
    and motivate them with a positive message.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",   # or "gpt-3.5-turbo"
        messages=[{"role": "system", "content": "You are a friendly health coach."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    sleep = int(request.form["sleep"])
    steps = int(request.form["steps"])
    diet = request.form["diet"]
    exercise = request.form["exercise"]
    glucose = float(request.form["glucose"])
    blood_pressure = float(request.form["blood_pressure"])
    bmi = float(request.form["bmi"])
    age = int(request.form["age"])

  
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO logs (sleep, steps, diet, exercise, glucose, blood_pressure, bmi, age) VALUES (?,?,?,?,?,?,?,?)",
              (sleep, steps, diet, exercise, glucose, blood_pressure, bmi, age))
    conn.commit()
    conn.close()

    X_input = np.array([[glucose, blood_pressure, bmi, age, sleep]])
    X_input = scaler.transform(X_input)
    prediction = model.predict(X_input)[0][0]

    result = "⚠️ High Risk of Disease" if prediction > 0.5 else "✅ Low Risk of Disease"

    user_data = {
        "sleep": sleep,
        "steps": steps,
        "diet": diet,
        "exercise": exercise,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "bmi": bmi,
        "age": age
    }

    advice = gpt_explanation(user_data, prediction)

    return render_template("index.html", prediction=result, advice=advice)

if __name__ == "__main__":
    app.run(debug=True)