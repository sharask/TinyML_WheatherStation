
//Mano modelis
#include "model.h"

//modelis is github
//#include "model_github.h"

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>


// Set to 1 if you want to see whether your model can forecast the snow
#define DEBUG_SNOW 0


// Set to 1 if you are using the Arduino Nano 33 BLE Sense Rev2
#define ARDUINO_ARDUINO_NANO33BLE_REV2 1


const tflite::Model* tflu_model            = nullptr;
tflite::MicroInterpreter* tflu_interpreter = nullptr;
TfLiteTensor* tflu_i_tensor                = nullptr;
TfLiteTensor* tflu_o_tensor                = nullptr;

float tflu_i_scale = 1.0f;
int32_t tflu_i_zero_point = 0;
float tflu_o_scale = 1.0f;
int32_t tflu_o_zero_point = 0;

constexpr int t_sz = 4096;
uint8_t tensor_arena[t_sz] __attribute__((aligned(16)));

tflite::AllOpsResolver tflu_ops_resolver;

constexpr int num_hours = 3;
int8_t t_vals [num_hours] = {0};
int8_t h_vals [num_hours] = {0};
int cur_idx = 0;

#if defined(ARDUINO_ARDUINO_NANO33BLE)
#if ARDUINO_ARDUINO_NANO33BLE_REV2 == 0
#include <Arduino_HTS221.h>
#define SENSOR HTS
#else
#include <Arduino_HS300x.h>
#define SENSOR HS300x
#endif

void setup() {
  Serial.begin(9600);

  while (!Serial);

  if (!SENSOR.begin()) {
    Serial.println("Failed sensor initialization!");
    while (1);
  }

  Serial.print("Test Temperature = ");
  Serial.print(SENSOR.readTemperature(), 2);
  Serial.println(" °C");
  Serial.print("Test Humidity = ");
  Serial.print(SENSOR.readHumidity(), 2);
  Serial.println(" %");

  tflu_model = tflite::GetModel(snow_model_tflite);

  static tflite::MicroInterpreter static_interpreter(
      tflu_model,
      tflu_ops_resolver,
      tensor_arena,
      t_sz);

  tflu_interpreter = &static_interpreter;

  tflu_interpreter->AllocateTensors();
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);

  const auto* i_quant = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);
  const auto* o_quant = reinterpret_cast<TfLiteAffineQuantization*>(tflu_o_tensor->quantization.params);

  tflu_i_scale      = i_quant->scale->data[0];
  tflu_i_zero_point = i_quant->zero_point->data[0];
  tflu_o_scale      = o_quant->scale->data[0];
  tflu_o_zero_point = o_quant->zero_point->data[0];
}
#endif

#if defined(ARDUINO_RASPBERRY_PI_PICO)
#include <DHT.h>

// Arduino pin number
const int gpio_pin_dht_pin = 10;
DHT dht(gpio_pin_dht_pin, DHT22);

#define SENSOR dht

void setup() {
  Serial.begin(9600);
  while(!Serial);
  SENSOR.begin();
  delay(2000);

  Serial.print("Test Temperature = ");
  Serial.print(SENSOR.readTemperature(), 2);
  Serial.println(" °C");
  Serial.print("Test Humidity = ");
  Serial.print(SENSOR.readHumidity(), 2);
  Serial.println(" %");

  tflu_model = tflite::GetModel(snow_model_tflite);

  static tflite::MicroInterpreter static_interpreter(
      tflu_model,
      tflu_ops_resolver,
      tensor_arena,
      t_sz);

  tflu_interpreter = &static_interpreter;

  tflu_interpreter->AllocateTensors();
  tflu_i_tensor = tflu_interpreter->input(0);
  tflu_o_tensor = tflu_interpreter->output(0);

  const auto* i_quant = reinterpret_cast<TfLiteAffineQuantization*>(tflu_i_tensor->quantization.params);
  const auto* o_quant = reinterpret_cast<TfLiteAffineQuantization*>(tflu_o_tensor->quantization.params);

  tflu_i_scale      = i_quant->scale->data[0];
  tflu_i_zero_point = i_quant->zero_point->data[0];
  tflu_o_scale      = o_quant->scale->data[0];
  tflu_o_zero_point = o_quant->zero_point->data[0];
}
#endif

constexpr int num_reads = 3;

void loop() {
  
  float t = 0.0f;
  float h = 0.0f;

  #if DEBUG_SNOW == 1
    t = -10.0f;
    h = 100.0f;
  #else
  
    for(int i = 0; i < num_reads; ++i) {
      t += SENSOR.readTemperature();
      h += SENSOR.readHumidity();
      delay(3000);
    }
  
    t /= (float)num_reads;
    h /= (float)num_reads;
  #endif

  Serial.print("Temperature = ");
  Serial.print(t, 2);
  Serial.println(" °C");
  Serial.print("Humidity = ");
  Serial.print(h, 2);
  Serial.println(" %");


  // PAVYZDINIS MODELIS
  // Use the mean and standard deviation
  // extracted from your dataset
  //constexpr float t_mean = 2.08993f;
  //constexpr float h_mean = 87.22773f;
  //constexpr float t_std  = 6.82158f;
  //constexpr float h_std  = 14.21543f;

  // MANO MODELIS
  constexpr float t_mean = 1.57673f;
  constexpr float h_mean = 76.98635f;
  constexpr float t_std  = 7.20765f;
  constexpr float h_std  = 16.93975f;

  
  t = (t - t_mean) / t_std;
  h = (h - h_mean) / h_std;

  t = (t / tflu_i_scale);
  t += (float)tflu_i_zero_point;
  h = (h / tflu_i_scale);
  h += (float)tflu_i_zero_point;

  t_vals[cur_idx] = t;
  h_vals[cur_idx] = h;

  cur_idx = (cur_idx + 1) % num_hours;

  int idx0 = cur_idx;
  int idx1 = (cur_idx - 1 + num_hours) % num_hours;
  int idx2 = (cur_idx - 2 + num_hours) % num_hours;
  tflu_i_tensor->data.int8[0] = t_vals[idx2];
  tflu_i_tensor->data.int8[1] = t_vals[idx1];
  tflu_i_tensor->data.int8[2] = t_vals[idx0];
  tflu_i_tensor->data.int8[3] = h_vals[idx2];
  tflu_i_tensor->data.int8[4] = h_vals[idx1];
  tflu_i_tensor->data.int8[5] = h_vals[idx0];

  tflu_interpreter->Invoke();

  float out_int8 = tflu_o_tensor->data.int8[0];
  float out_f = (out_int8 - tflu_o_zero_point);
  out_f *= tflu_o_scale;

  Serial.print("Prediction result: ");
  Serial.println(out_f);

  if (out_f > 0.5) {
    Serial.println("Yes, it snows");
  }
  else {
    Serial.println("No, it does not snow");
  }

  delay(2000);
}
