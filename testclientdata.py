from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Endpoint to receive data
@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        # Extract the incoming JSON data
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        # Log received data for debugging
        print("Received data:", data)

        # Simulate a response
        response = {
            "status": "success",
            "message": "Data processed successfully",
            "received_data": data
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
