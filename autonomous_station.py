# autonomous_station.py
# Skirta Raspberry Pi Pico W su TFLite modeliu, kuris prognozuoja sniego tikimybę pagal temperatūros ir drėgmės duomenis.
# paleistas RPI, siundžia duomenis į Flask serverį (PC), kuris yra sukonfigūruotas priimti duomenis.


import time
import numpy as np
from tflite_runtime.interpreter import Interpreter
import requests
from collections import deque

# --- Konfiguracija ---
# Modelio ir normalizavimo konstantos
TEMP_MEAN = 1.57673
TEMP_STD = 7.20765
HUMI_MEAN = 76.98635
HUMI_STD = 16.93975
TFLITE_MODEL_PATH = "snow_forecast_model.tflite"

# Svarbu: Įrašykite adresą, kurį rodo jūsų Flask serveris!
SERVER_URL = "http://192.168.1.102:5000/data"

# Prognozės darymo intervalas sekundėmis (pvz., 3600 = 1 valanda)
CHECK_INTERVAL_SECONDS = 30 # Testavimui nustatome trumpesnį intervalą

# --- Pagalbinės funkcijos ---

def read_sensor():
    """Simuliuoja temperatūros ir drėgmės jutiklio duomenų gavimą."""
    # Vėliau čia bus realaus jutiklio nuskaitymo kodas.
    # Kol kas grąžiname šiek tiek kintančius fiktyvius duomenis,
    # imituojame vėstantį orą, tinkamą sniegui.
    temp = np.random.uniform(-3.0, 1.0)
    humi = np.random.uniform(80.0, 95.0)
    return round(temp, 2), round(humi, 2)

def send_data(temperature, humidity, prediction_text, prediction_value):
    """Siunčia jutiklio duomenis ir prognozę į serverį."""
    payload = {
        "temperature": temperature,
        "humidity": humidity,
        "prediction_text": prediction_text,
        "prediction_value": float(prediction_value) # Svarbu konvertuoti į standartinį float
    }
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=5) # Pridedame timeout
        response.raise_for_status()
        print(f"Duomenys sėkmingai išsiųsti į {SERVER_URL}")
    except requests.exceptions.RequestException as e:
        print(f"Klaida siunčiant duomenis: {e}")

def scale_value(value, mean, std):
    """Normalizuoja reikšmę naudojant Z-score."""
    return (value - mean) / std

def run_prediction(interpreter, input_details, output_details, temp_m2, humi_m2, temp_m1, humi_m1, temp_now, humi_now):
    """Vykdo prognozę naudojant paruoštą TFLite modelį."""
    scaled_t0 = scale_value(temp_m2, TEMP_MEAN, TEMP_STD)
    scaled_t1 = scale_value(temp_m1, TEMP_MEAN, TEMP_STD)
    scaled_t2 = scale_value(temp_now, TEMP_MEAN, TEMP_STD)
    scaled_h0 = scale_value(humi_m2, HUMI_MEAN, HUMI_STD)
    scaled_h1 = scale_value(humi_m1, HUMI_MEAN, HUMI_STD)
    scaled_h2 = scale_value(humi_now, HUMI_MEAN, HUMI_STD)

    input_data_float = np.array([[scaled_t0, scaled_t1, scaled_t2, scaled_h0, scaled_h1, scaled_h2]], dtype=np.float32)

    if input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details['quantization']
        input_data = (input_data_float / input_scale + input_zero_point).astype(np.int8)
    else:
        input_data = input_data_float

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data_quantized = interpreter.get_tensor(output_details['index'])

    if output_details['dtype'] == np.int8:
        output_scale, output_zero_point = output_details['quantization']
        output_data_float = (output_data_quantized.astype(np.float32) - output_zero_point) * output_scale
    else:
        output_data_float = output_data_quantized.astype(np.float32)

    prediction_value = output_data_float[0][0]
    if prediction_value > 0.5:
        return "Taip (Sniegas prognozuojamas)", prediction_value
    else:
        return "Ne (Sniegas neprognozuojamas)", prediction_value

# --- Pagrindinė programos logika ---
def main():
    """Pagrindinė funkcija, kuri įkelia modelį ir paleidžia amžiną ciklą."""
    print("Autonominė orų stotelė paleidžiama...")

    try:
        interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        print("TFLite modelis sėkmingai įkeltas.")
    except Exception as e:
        print(f"Kritinė klaida įkeliant modelį: {e}")
        return

    measurements = deque(maxlen=3)

    print("Užpildomi pradiniai matavimai...")
    measurements.append(read_sensor())
    time.sleep(1)
    measurements.append(read_sensor())
    print("Pradiniai duomenys paruošti.")

    while True:
        print(f"\n--- {time.ctime()} ---")
        try:
            current_temp, current_humi = read_sensor()
            print(f"Nuskaityti jutiklio duomenys: T={current_temp}°C, H={current_humi}%")
            measurements.append((current_temp, current_humi))
        except Exception as e:
            print(f"Klaida nuskaitant jutiklio duomenis: {e}")
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        if len(measurements) < 3:
            print(f"Nepakanka duomenų prognozei. Reikia 3 matavimų, turima {len(measurements)}. Laukiama...")
        else:
            (temp_m2, humi_m2), (temp_m1, humi_m1), (temp_now, humi_now) = measurements
            print(f"Duomenys prognozei: T-2h=({temp_m2}, {humi_m2}), T-1h=({temp_m1}, {humi_m1}), T-0h=({temp_now}, {humi_now})")
            result, value = run_prediction(interpreter, input_details, output_details, temp_m2, humi_m2, temp_m1, humi_m1, temp_now, humi_now)
            print(f"Prognozė: {result} (Reikšmė: {value:.4f})")
            send_data(temp_now, humi_now, result, value)

        print(f"Laukiama {CHECK_INTERVAL_SECONDS} sek. iki kito tikrinimo...")
        time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma sustabdyta vartotojo.")