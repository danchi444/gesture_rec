#include <Arduino_BMI270_BMM150.h>
#include <ArduinoBLE.h>
#include "gesture_model.h"

#define SENSOR_SAMPLE_RATE_HZ 104
#define WINDOW_SIZE 1 * SENSOR_SAMPLE_RATE_HZ
#define NUM_FEATURES 6
#define MODEL_INPUT_SIZE (WINDOW_SIZE * NUM_FEATURES)
#define WINDOW_STRIDE_MS 500
#define DEBOUNCE_MS 1000

const char* BLE_DEVICE_NAME = "magic_wand";
const char* BLE_SERVICE_UUID = "0d431e22-b9ed-4938-b8fe-fe47311b469b";
const char* BLE_CHAR_UUID    = "24b077e2-a798-49ea-821f-824bd3998bde";
BLEService gestureService(BLE_SERVICE_UUID); // service je kontejner za karakteristike
BLECharacteristic gestureChar(BLE_CHAR_UUID, BLERead | BLENotify, 20); // karakteristika koja se moze citati (ime geste)
                                                                      // i koja obavijesti pratitelje o promjenama

Eloquent::ML::Port::RandomForest model;

float sensorBuffer[MODEL_INPUT_SIZE];
unsigned int bufferIndex = 0;
unsigned long lastSampleTime = 0;
unsigned long lastWindowTime = 0;

String lastGesture = "";
unsigned long lastGestureTime = 0;
const char* gestureLabels[] = {"4", "8", "alpha", "double", "flick", "junk"};

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  if (!BLE.begin()) {
    Serial.println("BLE init failed!");
    while (1);
  }

  BLE.setLocalName(BLE_DEVICE_NAME);
  BLE.setAdvertisedService(gestureService);
  gestureService.addCharacteristic(gestureChar);
  BLE.addService(gestureService);
  gestureChar.writeValue("ready");
  BLE.advertise();
  Serial.println("BLE device ready, advertising...");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.println("Connected to central");
    bufferIndex = 0;
    lastSampleTime = millis();
    lastWindowTime = millis();

    while (central.connected()) {
      unsigned long now = millis();

      if (now - lastSampleTime >= 1000 / SENSOR_SAMPLE_RATE_HZ) {
        float ax, ay, az, gx, gy, gz;
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
          IMU.readAcceleration(ax, ay, az);
          IMU.readGyroscope(gx, gy, gz);

          if (bufferIndex >= MODEL_INPUT_SIZE) {
            memmove(sensorBuffer, sensorBuffer + NUM_FEATURES, sizeof(float) * (MODEL_INPUT_SIZE - NUM_FEATURES));
            bufferIndex = MODEL_INPUT_SIZE - NUM_FEATURES;
          }

          sensorBuffer[bufferIndex++] = ax;
          sensorBuffer[bufferIndex++] = ay;
          sensorBuffer[bufferIndex++] = az;
          sensorBuffer[bufferIndex++] = gx;
          sensorBuffer[bufferIndex++] = gy;
          sensorBuffer[bufferIndex++] = gz;
        }
        lastSampleTime = now;
      }

      if (bufferIndex >= MODEL_INPUT_SIZE && now - lastWindowTime >= WINDOW_STRIDE_MS) {
        float gyroSum = 0;
        for (int i = 0; i < MODEL_INPUT_SIZE; i += NUM_FEATURES) {
            float gx = sensorBuffer[i+3];
            float gy = sensorBuffer[i+4];
            float gz = sensorBuffer[i+5];
            gyroSum += sqrt(gx*gx + gy*gy + gz*gz);
        }
        float gyroAvg = gyroSum / (MODEL_INPUT_SIZE / NUM_FEATURES);
        if (gyroAvg < 250) {
            continue;
        }
        
        int gestureIdx = model.predict(sensorBuffer);
        String gesture = String(gestureLabels[gestureIdx]);

        Serial.print("Predicted: ");
        Serial.println(gesture);

        if (gesture != "junk") {
          if (gesture != lastGesture || now - lastGestureTime > DEBOUNCE_MS) {
            gestureChar.writeValue(gesture.c_str());
            Serial.print("Sent over BLE: ");
            Serial.println(gesture);

            lastGesture = gesture;
            lastGestureTime = now;
          }
        }
        lastWindowTime = now;
      }
    }
    Serial.println("Central disconnected");
  }
}