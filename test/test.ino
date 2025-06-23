#include <Arduino_BMI270_BMM150.h> 
#include <Arduino.h>     

const unsigned long RECORD_DURATION_MS = 10000; 

unsigned long startTime = 0;
unsigned long lastReadTime = 0;
volatile int sampleCount = 0; 

void setup() {
  Serial.begin(115200); 
  while (!Serial);     

  if (!IMU.begin()) { 
    Serial.println("Failed to initialize IMU!");
    while (1); 
  }

  Serial.println("IMU initialized. Starting sample rate test.");
  Serial.print("Recording for ");
  Serial.print(RECORD_DURATION_MS / 1000);
  Serial.println(" seconds...");

  startTime = millis(); 
  lastReadTime = micros(); 
}

void loop() {
  unsigned long now = millis();

  if (now - startTime >= RECORD_DURATION_MS) {
    Serial.println("\n--- Sample Rate Test Results ---");
    Serial.print("Total samples collected: ");
    Serial.println(sampleCount);
    Serial.print("Recording duration (ms): ");
    Serial.println(RECORD_DURATION_MS);

    if (sampleCount > 0) {
      float averageIntervalMs = (float)RECORD_DURATION_MS / sampleCount;
      float calculatedHz = 1000.0 / averageIntervalMs; 
      Serial.print("Average interval between samples: ");
      Serial.print(averageIntervalMs, 2); 
      Serial.println(" ms");
      Serial.print("Calculated Sample Rate: ");
      Serial.print(calculatedHz, 2); 
      Serial.println(" Hz");
    } else {
      Serial.println("No samples collected during the recording period.");
    }
    
    Serial.println("Test complete. Halting loop.");
    while (1); 
  }

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    float ax, ay, az, gx, gy, gz;
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);
    
    sampleCount++;
  }
}