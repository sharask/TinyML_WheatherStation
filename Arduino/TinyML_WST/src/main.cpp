#include <Arduino.h>
#include <Arduino_HS300x.h> // Biblioteka, skirta darbui su HS300x temperatūros ir drėgmės jutikliu

// TensorFlow Lite Micro bibliotekos
#include <tensorflow/lite/micro/all_ops_resolver.h>
//#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Jūsų sugeneruotas modelis
#include "model.h" // Įsitikinkite, kad šis failas yra 'src' arba 'include' aplanke

// --- Konfiguracija ---

// Normalizavimo konstantos iš prepare_model.py
const float TEMP_MEAN = 1.57673;
const float TEMP_STD = 7.20765;
const float HUMI_MEAN = 76.98635;
const float HUMI_STD = 16.93975;

// Laiko intervalas tarp matavimų (milisekundėmis)
// Realiomis sąlygomis tai būtų valanda (3600000), bet testavimui naudojame 5 sekundes
const int CHECK_INTERVAL_MS = 5000;

// --- TensorFlow Lite kintamieji ---
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tenorių arena - atminties sritis, kurią naudos TFLite.
// Modelis yra mažas, bet saugumo dėlei priskiriame 8KB.
// Jei trūktų atminties, šią reikšmę reikėtų didinti.
constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// --- Duomenų saugojimas ---
// Struktūra vienam matavimui
struct SensorReading {
  float temperature;
  float humidity;
};

// Ciklinis buferis 3 paskutiniams matavimams saugoti
const int HISTORY_SIZE = 3;
SensorReading history[HISTORY_SIZE];
int history_index = 0;
int measurement_count = 0;


void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("--- Sniego prognozės stotelė ---");

  // 1. Jutiklio inicializavimas
  if (!HS300x.begin()) {
    Serial.println("Klaida: nepavyko inicializuoti HS300x jutiklio!");
    while (1);
  }
  Serial.println("HS300x jutiklis paruoštas.");

  // 2. TFLite modelio paruošimas
  // Sukuriame klaidų pranešėją
  //static tflite::MicroErrorReporter micro_error_reporter;
  //error_reporter = &micro_error_reporter;

  // Įkeliame modelį iš C masyvo (snow_forecast_model.h)
  model = tflite::GetModel(snow_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Modelio versija nesutampa! Tikėtasi %d, gauta %d.",
                           TFLITE_SCHEMA_VERSION, model->version());
    return;
  }

  // Sukuriame operacijų sprendiklį (resolver).
  // AllOpsResolver įtraukia visas palaikomas operacijas, kas naudoja daugiau atminties.
  // Norint optimizuoti, galima sukurti MicroMutableOpResolver ir pridėti tik reikalingas operacijas.
  static tflite::AllOpsResolver resolver;

  // Sukuriame interpretatorių
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Paskirstome atmintį tenzoriams
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("Klaida skirstant atmintį tenzoriams!");
    return;
  }

  // Gauname nuorodas į įvesties ir išvesties tenzorius
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TensorFlow Lite modelis sėkmingai įkeltas.");
  Serial.println("Pradedami matavimai...");
}

void loop() {
  // 1. Nuskaitome jutiklio duomenis
  float temp = HS300x.readTemperature();
  float humi = HS300x.readHumidity();

  if (isnan(temp) || isnan(humi)) {
    Serial.println("Klaida: nepavyko nuskaityti duomenų iš jutiklio!");
    delay(CHECK_INTERVAL_MS);
    return;
  }

  Serial.print("Nuskaityta: T=");
  Serial.print(temp, 2);
  Serial.print("°C, H=");
  Serial.print(humi, 2);
  Serial.println("%");

  // 2. Išsaugome matavimą istorijoje
  history[history_index] = {temp, humi};
  history_index = (history_index + 1) % HISTORY_SIZE;
  if (measurement_count < HISTORY_SIZE) {
    measurement_count++;
  }

  // 3. Jei turime pakankamai duomenų, darome prognozę
  if (measurement_count < HISTORY_SIZE) {
    Serial.print("Renkama istorija... Reikia ");
    Serial.print(HISTORY_SIZE);
    Serial.print(", turima ");
    Serial.print(measurement_count);
    Serial.println(".");
  } else {
    Serial.println("Ruošiami duomenys prognozei...");

    // 4. Duomenų paruošimas (normalizavimas ir kvantavimas)
    float input_scale = input->params.scale;
    int8_t input_zero_point = input->params.zero_point;

    float input_float[6];
    for (int i = 0; i < HISTORY_SIZE; ++i) {
      int current_index = (history_index - HISTORY_SIZE + i + HISTORY_SIZE) % HISTORY_SIZE;
      input_float[i]     = (history[current_index].temperature - TEMP_MEAN) / TEMP_STD;
      input_float[i + 3] = (history[current_index].humidity - HUMI_MEAN) / HUMI_STD;
    }

    for (int i = 0; i < 6; ++i) {
      input->data.int8[i] = static_cast<int8_t>(input_float[i] / input_scale + input_zero_point);
    }
    
    // 5. Vykdome modelį
    if (interpreter->Invoke() != kTfLiteOk) {
      error_reporter->Report("Modelio vykdymo klaida!");
      return;
    }

    // 6. Apdorojame rezultatą (dekvantavimas)
    int8_t output_quantized = output->data.int8[0];
    float output_scale = output->params.scale;
    int8_t output_zero_point = output->params.zero_point;
    float prediction_value = (static_cast<float>(output_quantized) - output_zero_point) * output_scale;

    // 7. Išvedame prognozę
    Serial.println("--- PROGNOZĖ ---");
    Serial.print("Modelio reikšmė: ");
    Serial.println(prediction_value, 4);
    if (prediction_value > 0.5) {
      Serial.println("Rezultatas: TAIP, sniegas prognozuojamas!");
    } else {
      Serial.println("Rezultatas: NE, sniegas neprognozuojamas.");
    }
    Serial.println("------------------");
  }

  // Laukiame iki kito matavimo
  Serial.print("Laukiama ");
  Serial.print(CHECK_INTERVAL_MS / 1000);
  Serial.println(" sek...");
  Serial.println();
  delay(CHECK_INTERVAL_MS);
}