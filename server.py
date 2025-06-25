# server.py
# priima duomenis iš autonominės stotelės ir juos apdoroja. - iš autonomous_station.py (RPI)

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def receive_data():
  if request.method == 'POST':
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

    # Išskiriame duomenis
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    prediction_text = data.get('prediction_text')
    prediction_value = data.get('prediction_value')

    # Spausdiname tvarkingą pranešimą serveryje
    print(f"Gauta iš stotelės: T={temperature}°C, H={humidity}%. Prognozė: {prediction_text} (Reikšmė: {prediction_value:.4f})")
    
    # Čia galite išsaugoti duomenis į failą, duomenų bazę ar pan.

    return jsonify({'status': 'success', 'message': 'Data received'}), 200
  
  return jsonify({'status': 'error', 'message': 'Only POST method is allowed'}), 405

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=5000)
