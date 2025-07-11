const int voltagePin = A0;
const int groundPin = A5;

const int readingRed = 2;
const int readingGreen = 3;
const int readingBlue = 4;

float redVoltage = 0;
float greenVoltage = 0;
float blueVoltage = 0;

bool redCaptured = false;
bool greenCaptured = false;
bool blueCaptured = false;

void setup() {
  Serial.begin(9600);
  pinMode(readingRed, INPUT);
  pinMode(readingGreen, INPUT);
  pinMode(readingBlue, INPUT);

  Serial.println("Keep LEDs running, NO BALL placed now...");
}

void loop() {
  // Your original logic preserved exactly
  if (digitalRead(readingGreen) == HIGH && !greenCaptured) {
    greenVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
    greenCaptured = true;
    Serial.print("Green LED detected. Initial Green Voltage (mV): ");
    Serial.println(greenVoltage);
  }
  else if (digitalRead(readingBlue) == HIGH && !blueCaptured) {
    blueVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
    blueCaptured = true;
    Serial.print("Blue LED detected. Initial Blue Voltage (mV): ");
    Serial.println(blueVoltage);
  }
  else if (!redCaptured) {
    redVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
    redCaptured = true;
    Serial.print("Red LED detected. Initial Red Voltage (mV): ");
    Serial.println(redVoltage);
  }

  // After capturing all three
  if (redCaptured && greenCaptured && blueCaptured) {
    Serial.println("All initial voltages measured. Outputting CSV:");
    Serial.print(redVoltage);
    Serial.print(",");
    Serial.print(greenVoltage);
    Serial.print(",");
    Serial.println(blueVoltage);

    while (1);  // stop execution
  }
}

float measureVoltage(int pin) {
  delay(100);  // Wait for voltage to stabilize
  int raw = analogRead(pin);
  float voltage_mv = (raw * 5000.0) / 1023.0;
  return voltage_mv;
}
