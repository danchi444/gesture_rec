#include <Arduino_BMI270_BMM150.h>
#include <Arduino.h>            

#define TARGET_SAMPLE_RATE_HZ 88 
#define INTERVAL_US (1000000 / TARGET_SAMPLE_RATE_HZ) 

unsigned long lastSampleMicros = 0; 

void setup() {
  Serial.begin(115200); 
  while (!Serial);     

  if (!IMU.begin()) { 
    Serial.println("Failed to initialize IMU!");
    while (1); 
  }

  Serial.println("timestamp,ax,ay,az,gx,gy,gz");
  Serial.print("IMU initialized. Recording at ~");
  Serial.print(TARGET_SAMPLE_RATE_HZ);
  Serial.println(" Hz...");

  lastSampleMicros = micros(); 
}

void loop() {
  unsigned long currentMicros = micros(); 

  if (currentMicros - lastSampleMicros >= INTERVAL_US) {
    float ax, ay, az, gx, gy, gz;

    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(ax, ay, az); 
      IMU.readGyroscope(gx, gy, gz);    

      char buffer[128];
    snprintf(buffer, sizeof(buffer), "%lu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
              currentMicros, ax, ay, az, gx, gy, gz);
    Serial.println(buffer);

      lastSampleMicros = currentMicros; 
    }
  }
}