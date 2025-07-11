const int voltagePin = A0;
const int groundPin = A5;

const int readingRed = 2;
const int readingGreen = 3;
const int readingBlue = 4;

float redVoltageInitial = 0;
float greenVoltageInitial = 0;
float blueVoltageInitial = 0;

bool firstIteration = true;

void setup() {
  Serial.begin(9600);
  pinMode(readingRed, INPUT);
  pinMode(readingGreen, INPUT);
  pinMode(readingBlue, INPUT);
//  Serial.println("Keep LEDs running, NO BALL placed now...");
}

void loop() {
  if (firstIteration) {
    // Store initial readings for all 3 colors

    if (digitalRead(readingGreen) == HIGH) {
      
      greenVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
//      Serial.print("Initial Green: ");
//      Serial.println(greenVoltageInitial);
    }

    else if (digitalRead(readingBlue) == HIGH) {

      blueVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
//      Serial.print("Initial Blue: ");
//      Serial.println(blueVoltageInitial);
    }
    else {

      redVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
//      Serial.print("Initial Red: ");
//      Serial.println(redVoltageInitial);
    }

    // Check if all 3 are captured
    if (redVoltageInitial > 0 && greenVoltageInitial > 0 && blueVoltageInitial > 0) {
//      Serial.println("Initial measurements complete. Place the ball now...");
      delay(5000);  // give user time to place the ball
      firstIteration = false;
        Serial.print(redVoltageInitial);
        Serial.print(",");
        Serial.print(greenVoltageInitial);
        Serial.print(",");
        Serial.println(blueVoltageInitial);
    }
  }

}

float measureVoltage(int pin) {
  delay(100);  // Wait for signal to stabilize
  int raw = analogRead(pin);
  float voltage_mv = (raw * 5000.0) / 1023.0;
  return voltage_mv;
}
