const int voltagePin = A0;
const int groundPin = A5;

float redVoltageInitial = 0;
float greenVoltageInitial = 0;
float blueVoltageInitial = 0;

bool firstIteration = true;

void setup() {
  Serial.begin(9600);


}

void loop() {

      delay(2000);
      redVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      delay(5000);      
      greenVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      delay(5000);      
      blueVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
 

      Serial.print(redVoltageInitial);
      Serial.print(",");
      Serial.print(greenVoltageInitial);
      Serial.print(",");
      Serial.println(blueVoltageInitial);    
}

float measureVoltage(int pin) {
  delay(100);  // Wait for signal to stabilize
  int raw = analogRead(pin);
  float voltage_mv = (raw * 5000.0) / 1023.0;
  return voltage_mv;
}
