from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('gait_cnn_model.h5')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """Mock door controller API for gait authentication."""
    try:
        data = request.json['sensor_data']  # Expected: (128, 6) array
        data = np.array(data)[np.newaxis, :]  # Add batch dimension
        
        # Normalize orientation
        mag = np.sqrt(np.sum(data**2, axis=2, keepdims=True))
        data = data / (mag + 1e-10)
        
        # Predict employee ID (0-based, mapped back to 1-based)
        pred = model.predict(data)
        employee_id = np.argmax(pred, axis=1)[0] + 1  # Convert to 1-based (1 to 30)
        
        # Mock door control
        response = {"status": "success", "employee_id": int(employee_id), "access": "granted"}
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)