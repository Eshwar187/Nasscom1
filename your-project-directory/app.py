from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import mysql.connector
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# MySQL Database Connection Details
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "Si#200805"
DB_NAME = "sensor_data"

# Connect to MySQL database
try:
    db = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    print("Database connection successful.")
except mysql.connector.Error as err:
    print(f"Error connecting to MySQL: {err}")
    exit(1)

# Load Hugging Face model and tokenizer
model_name = "gpt2"  # You can choose a different model from Hugging Face Model Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token if pad_token is not set

# Serve the HTML page when user accesses the root route
@app.route('/')
def index():
    return render_template('index.html')

# Fetch all sensor names from the database
@app.route('/get_sensors', methods=['GET'])
def get_sensors():
    cursor = db.cursor(dictionary=True)
    query = "SELECT DISTINCT sensor_name FROM sensors"
    cursor.execute(query)
    result = cursor.fetchall()

    sensor_names = [row['sensor_name'] for row in result]
    return jsonify(sensor_names)

# Fetch sensor data based on selected sensor
@app.route('/get_sensor_data', methods=['POST'])
def get_sensor_data():
    sensor_name = request.json['sensor_name']

    cursor = db.cursor(dictionary=True)
    query = "SELECT * FROM sensors WHERE sensor_name = %s ORDER BY timestamp DESC LIMIT 1"
    cursor.execute(query, (sensor_name,))
    result = cursor.fetchone()

    if result:
        sensor_value = result['sensor_value']
        ideal_value = result['ideal_value']

        feedback = generate_stabilization_feedback(sensor_name, sensor_value, ideal_value)

        return jsonify({
            "sensor_name": sensor_name,
            "actual_value": sensor_value,
            "ideal_value": ideal_value,
            "feedback": feedback
        })
    else:
        return jsonify({"error": "Sensor not found"}), 404

# Updated function to generate AI-based feedback using Hugging Face models
def generate_stabilization_feedback(sensor_name, sensor_value, ideal_value):
    try:
        # Updated, more focused prompt to guide the model
        prompt = (
            f"Sensor: {sensor_name}\n"
            f"Actual Value: {sensor_value}\n"
            f"Ideal Value: {ideal_value}\n\n"
            "Based on the above sensor data, explain how AI and machine learning could be applied to stabilize the sensor performance. "
            "Discuss techniques like calibration, prediction accuracy, anomaly detection, and error minimization."
        )

        # Tokenize the input and generate feedback using the Hugging Face model
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Model parameters for more focused and deterministic output
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            max_length=150,  # Increase max length if necessary
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=2,  # Prevents repetition of n-grams
            eos_token_id=tokenizer.eos_token_id,  # Stops generation at eos token
            temperature=0.7,  # Less randomness, more focused responses
            top_p=0.85  # Top-p (nucleus sampling) for coherent output
        )
        
        feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return feedback

    except Exception as e:
        print(f"Error generating feedback: {e}")
        return "Error generating feedback. Please try again."

if __name__ == '__main__':
    app.run(debug=True)
