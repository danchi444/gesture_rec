#include <Arduino_BMI270_BMM150.h> // bmi270 akc i ziro, bmi150 mag

unsigned long startTime;

void setup() {
  Serial.begin(115200); // serial.begin() ne vraca nis
  // 115200 brzina prijenosa, otp duplo od prosjecne brzine ulaznih podataka sa senzora
    // 104hz * otp 500 bitova po uzorku
  while (!Serial); // za slucaj da mu treba malo, da ne salje podatke prije nego sto se veza uspostavi

  if (!IMU.begin()) { // vraca true/false ako je/nije uspjesno inicijaliziran
    Serial.println("sensor not found");
    while (true);
  }

  startTime = millis();
}

void loop() {
  float ax, ay, az, gx, gy, gz;

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    unsigned long timestamp = millis() - startTime;

    char buffer[128];
    snprintf(buffer, sizeof(buffer), "%lu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
              timestamp, ax, ay, az, gx, gy, gz);
    Serial.println(buffer);
  }
}