import tensorflow as tf
import numpy as np

# --- Konfiguracija ---
# Būtinai atnaujinkite šias reikšmes pagal tai, ką išspausdino jūsų prepare_model.py scenarijus!
# -- atnaujinta! --
TEMP_MEAN = 1.57673
TEMP_STD = 7.20765
HUMI_MEAN = 76.98635
HUMI_STD = 16.93975

TFLITE_MODEL_PATH = "snow_forecast_model.tflite" # Kelias iki jūsų .tflite modelio failo

# Funkcija įvesties duomenų normalizavimui (Z-score scaling)
def scale_value(value, mean, std):
    return (value - mean) / std

def predict_snow(temp_minus_2, humi_minus_2, temp_minus_1, humi_minus_1, temp_now, humi_now):
    """
    Funkcija, kuri įkelia TFLite modelį, paruošia įvesties duomenis,
    atlieka prognozę ir grąžina rezultatą.
    """
    # Įkeliame TFLite modelį ir priskiriame tenzorius
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Gauname įvesties ir išvesties tenzorių informaciją
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Paruošiame įvesties duomenis
    # 1. Normalizuojame reikšmes
    scaled_t0 = scale_value(temp_minus_2, TEMP_MEAN, TEMP_STD)
    scaled_t1 = scale_value(temp_minus_1, TEMP_MEAN, TEMP_STD)
    scaled_t2 = scale_value(temp_now, TEMP_MEAN, TEMP_STD)
    scaled_h0 = scale_value(humi_minus_2, HUMI_MEAN, HUMI_STD)
    scaled_h1 = scale_value(humi_minus_1, HUMI_MEAN, HUMI_STD)
    scaled_h2 = scale_value(humi_now, HUMI_MEAN, HUMI_STD)

    # 2. Suformuojame įvesties masyvą (float32)
    input_data_float = np.array([[scaled_t0, scaled_t1, scaled_t2, scaled_h0, scaled_h1, scaled_h2]], dtype=np.float32)

    # 3. Kvantavimas į INT8 (jei modelis naudoja INT8 įvestį)
    # Patikriname, ar įvesties tipas yra INT8
    if input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details['quantization']
        # Kvantavimo formulė: int_val = float_val / scale + zero_point
        input_data = (input_data_float / input_scale + input_zero_point).astype(np.int8)
    else: # Jei modelis laukia float32 (mažai tikėtina pagal jūsų prepare_model.py)
        input_data = input_data_float

    # Nustatome įvesties tenzoriaus reikšmę
    interpreter.set_tensor(input_details['index'], input_data)

    # Vykdome prognozę
    interpreter.invoke()

    # Gauname išvesties tenzoriaus reikšmę
    output_data_quantized = interpreter.get_tensor(output_details['index'])

    # Dekvantavimas (jei modelis naudoja INT8 išvestį)
    if output_details['dtype'] == np.int8:
        output_scale, output_zero_point = output_details['quantization']
        # Dekvantavimo formulė: float_val = (int_val - zero_point) * scale
        output_data_float = (output_data_quantized.astype(np.float32) - output_zero_point) * output_scale
    else: # Jei modelis grąžina float32
        output_data_float = output_data_quantized.astype(np.float32)

    # Interpretuojame rezultatą (modelis grąžina reikšmę tarp 0 ir 1)
    # Jūsų prepare_model.py scenarijuje riba buvo 0.5
    prediction_value = output_data_float[0][0]
    
    if prediction_value > 0.5:
        return "Taip (Sniegas prognozuojamas)", prediction_value
    else:
        return "Ne (Sniegas neprognozuojamas)", prediction_value

if __name__ == "__main__":
    print("TensorFlow Lite modelio testavimas sniego prognozei.\n")

    # Pavyzdiniai įvesties duomenys (temperatūra °C, drėgmė %)
    # Pakeiskite šias reikšmes norimomis testuoti
    # Reikšmės: (t-2h), (h-2h), (t-1h), (h-1h), (t_dabar), (h_dabar)
    
    # Scenarijus 1: Šalta ir drėgna (tikėtinas sniegas)
    print("Testuojamas scenarijus: Šalta ir drėgna")
    temp_m2_s1, humi_m2_s1 = -2.0, 85.0
    temp_m1_s1, humi_m1_s1 = -1.5, 88.0
    temp_now_s1, humi_now_s1 = -1.0, 90.0
    
    result_s1, value_s1 = predict_snow(temp_m2_s1, humi_m2_s1, temp_m1_s1, humi_m1_s1, temp_now_s1, humi_now_s1)
    print(f"  Įvestis: T(t-2)={temp_m2_s1}°C, H(t-2)={humi_m2_s1}%; T(t-1)={temp_m1_s1}°C, H(t-1)={humi_m1_s1}%; T(t0)={temp_now_s1}°C, H(t0)={humi_now_s1}%")
    print(f"  Prognozė: {result_s1} (Reikšmė: {value_s1:.4f})\n")

    # Scenarijus 2: Šilta ir sausa (mažai tikėtinas sniegas)
    print("Testuojamas scenarijus: Šilta ir sausa")
    temp_m2_s2, humi_m2_s2 = 10.0, 50.0
    temp_m1_s2, humi_m1_s2 = 11.0, 48.0
    temp_now_s2, humi_now_s2 = 12.0, 45.0

    result_s2, value_s2 = predict_snow(temp_m2_s2, humi_m2_s2, temp_m1_s2, humi_m1_s2, temp_now_s2, humi_now_s2)
    print(f"  Įvestis: T(t-2)={temp_m2_s2}°C, H(t-2)={humi_m2_s2}%; T(t-1)={temp_m1_s2}°C, H(t-1)={humi_m1_s2}%; T(t0)={temp_now_s2}°C, H(t0)={humi_now_s2}%")
    print(f"  Prognozė: {result_s2} (Reikšmė: {value_s2:.4f})\n")

    # Scenarijus 3: Temperatūra apie nulį, vidutinė drėgmė
    print("Testuojamas scenarijus: Temperatūra apie nulį, vidutinė drėgmė")
    temp_m2_s3, humi_m2_s3 = 1.0, 70.0
    temp_m1_s3, humi_m1_s3 = 0.5, 75.0
    temp_now_s3, humi_now_s3 = 0.0, 80.0

    result_s3, value_s3 = predict_snow(temp_m2_s3, humi_m2_s3, temp_m1_s3, humi_m1_s3, temp_now_s3, humi_now_s3)
    print(f"  Įvestis: T(t-2)={temp_m2_s3}°C, H(t-2)={humi_m2_s3}%; T(t-1)={temp_m1_s3}°C, H(t-1)={humi_m1_s3}%; T(t0)={temp_now_s3}°C, H(t0)={humi_now_s3}%")
    print(f"  Prognozė: {result_s3} (Reikšmė: {value_s3:.4f})\n")

    # Galite pridėti daugiau savo testavimo scenarijų
    # Pavyzdžiui, galite nuskaityti keletą eilučių iš jūsų originalaus CSV failo
    # ir pateikti jas modeliui.
    print("\nTestavimas baigtas.\n")