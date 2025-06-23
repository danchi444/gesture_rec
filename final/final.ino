#include <Arduino_BMI270_BMM150.h>
#include <ArduinoBLE.h>
#include <TensorFlowLite.h>
#include "model_quantized.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define SENSOR_SAMPLE_RATE_HZ 88
#define WINDOW_SIZE ((int)(3 * SENSOR_SAMPLE_RATE_HZ))
#define NUM_FEATURES 6
#define DEBOUNCE_MS 5000 

const char* BLE_DEVICE_NAME = "magic_wand";
const char* BLE_SERVICE_UUID = "0d431e22-b9ed-4938-b8fe-fe47311b469b";
const char* BLE_CHAR_UUID    = "24b077e2-a798-49ea-821f-824bd3998bde";
BLEService gestureService(BLE_SERVICE_UUID);
BLECharacteristic gestureChar(BLE_CHAR_UUID, BLERead | BLENotify, 1);

const float means[6] = {0.157700, -0.303717, 0.655390, 1.477335, -2.782316, -7.167620};
const float stds[6] = {0.314980, 0.295075, 0.531993, 98.251744, 59.885257, 32.191821};

float sensorBuffer[WINDOW_SIZE][NUM_FEATURES]; 
unsigned int bufferIndex = 0;
unsigned long lastSampleTime = 0; 

bool isRecordingWindow = false; 

unsigned long lastGestureTime = 0; 

constexpr int kTensorArenaSize = 75 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

float input_scale;
int input_zero_point;

void setup() {
  //Serial.begin(115200);
  //while (!Serial);

  if (!IMU.begin()) {
    //Serial.println("Failed to initialize IMU!");
    while (1);
  }

  if (!BLE.begin()) {
    //Serial.println("BLE init failed!");
    while (1);
  }

  BLE.setLocalName(BLE_DEVICE_NAME);
  BLE.setAdvertisedService(gestureService);
  gestureService.addCharacteristic(gestureChar);
  BLE.addService(gestureService);
  BLE.advertise();
  //Serial.println("BLE device ready, advertising...");

  const tflite::Model* model = tflite::GetModel(model_tflite);
  static tflite::MicroMutableOpResolver<11> micro_op_resolver;
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddBatchToSpaceNd();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddExpandDims();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddSpaceToBatchNd();
  micro_op_resolver.AddStridedSlice();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  input_scale = input->params.scale;
  input_zero_point = input->params.zero_point;
  output = interpreter->output(0);
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    //Serial.println("Connected to central");
    bufferIndex = 0;
    lastSampleTime = millis();
    lastGestureTime = 0;
    isRecordingWindow = false;

    while (central.connected()) {
      unsigned long now = millis();

      if (now - lastSampleTime >= 1000 / SENSOR_SAMPLE_RATE_HZ) {
        float ax, ay, az, gx, gy, gz;
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
          IMU.readAcceleration(ax, ay, az);
          IMU.readGyroscope(gx, gy, gz);

          float raw_vals[6] = {ax, ay, az, gx, gy, gz};
          float norm_vals[6];
          for (int i = 0; i < 6; i++) {
            norm_vals[i] = (raw_vals[i] - means[i]) / stds[i];
          }

          if (!isRecordingWindow) { 
            float gx_current = norm_vals[3];
            float gy_current = norm_vals[4];
            float gz_current = norm_vals[5];

            if (abs(gx_current) > 1.0 || abs(gy_current) > 1.0 || abs(gz_current) > 1.0) {
              isRecordingWindow = true; 
              bufferIndex = 0;     
              //Serial.println("Gesture trigger detected. Starting window capture.");
            }
          }

          if (isRecordingWindow) {
            if (bufferIndex < WINDOW_SIZE) { 
              for (int i = 0; i < NUM_FEATURES; i++) {
                sensorBuffer[bufferIndex][i] = norm_vals[i];
              }
              bufferIndex++; 
            }
          }
        }
        lastSampleTime = now; 
      }

      if (isRecordingWindow && bufferIndex >= WINDOW_SIZE) { 
        if (now - lastGestureTime > DEBOUNCE_MS) { 
          int idx = 0;
          for (int i = 0; i < WINDOW_SIZE; i++) {
            for (int j = 0; j < NUM_FEATURES; j++) {
              float val = sensorBuffer[i][j];
              int8_t q = static_cast<int8_t>(round(val / input_scale) + input_zero_point);
              input->data.int8[idx++] = q;
            }
          }

          interpreter->Invoke();

          int gestureIdx = 0; 
          int8_t maxVal = output->data.int8[0];
          for (int i = 1; i < output->dims->data[1]; i++) {
            if (output->data.int8[i] > maxVal) {
              maxVal = output->data.int8[i];
              gestureIdx = i;
            }
          }

          const int junkIndex = 3; 
          if (gestureIdx != junkIndex) {
            uint8_t gestureByte = static_cast<uint8_t>(gestureIdx);
            gestureChar.writeValue(&gestureByte, 1);

            //Serial.print("Final gesture index (output): ");
            //Serial.println(gestureIdx);

            lastGestureTime = now;   
          }
        }

        isRecordingWindow = false; 
        bufferIndex = 0;         
        //Serial.println("Window processed. Waiting for next gesture trigger.");

      } 
    } 
    //Serial.println("Central disconnected");
  } 
}