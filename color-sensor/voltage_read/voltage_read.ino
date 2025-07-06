const int voltagePin = A0;
const int groundPin = A5;

const int readingRed = 2;
const int readingGreen = 3;
const int readingBlue = 4;

float redVoltageInitial = 0;
float greenVoltageInitial = 0;
float blueVoltageInitial = 0;

float redVoltage = 0;
float greenVoltage = 0;
float blueVoltage = 0;

bool firstIteration = true;

void setup() {
  Serial.begin(9600);
  pinMode(readingRed, INPUT);
  pinMode(readingGreen, INPUT);
  pinMode(readingBlue, INPUT);
  Serial.println("Keep LEDs running, NO BALL placed now...");
}

void loop() {
  if (firstIteration) {
    // Store initial readings for all 3 colors

    if (digitalRead(readingGreen) == HIGH) {
      greenVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Green: ");
      Serial.println(greenVoltageInitial);
    }

    else if (digitalRead(readingBlue) == HIGH) {
      blueVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Blue: ");
      Serial.println(blueVoltageInitial);
    }
    else {
      redVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Red: ");
      Serial.println(redVoltageInitial);
    }

    // Check if all 3 are captured
    if (redVoltageInitial > 0 && greenVoltageInitial > 0 && blueVoltageInitial > 0) {
      Serial.println("Initial measurements complete. Place the ball now...");
      delay(5000);  // give user time to place the ball
      firstIteration = false;
    }
  }
  else {


    if (digitalRead(readingGreen) == HIGH) {
      greenVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Green: ");
      Serial.println(greenVoltage);
    }

    else if (digitalRead(readingBlue) == HIGH) {
      blueVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Blue: ");
      Serial.println(blueVoltage);
    }

    else{
      redVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Red: ");
      Serial.println(redVoltage);
    }

    // Only proceed if all 3 colors were read
    if (redVoltage > 0 && greenVoltage > 0 && blueVoltage > 0) {
      float diffRed = redVoltage - redVoltageInitial;
      float diffGreen = greenVoltage - greenVoltageInitial;
      float diffBlue = blueVoltage - blueVoltageInitial;

      Serial.println("Voltage Differences:");
      Serial.print("Red Diff: "); Serial.println(diffRed);
      Serial.print("Green Diff: "); Serial.println(diffGreen);
      Serial.print("Blue Diff: "); Serial.println(diffBlue);

      if (diffRed > diffGreen && diffRed > diffBlue) {
        Serial.println("Detected Color: RED");
      }
      else if (diffGreen > diffRed && diffGreen > diffBlue) {
        Serial.println("Detected Color: GREEN");
      }
      else if (diffBlue > diffRed && diffBlue > diffGreen) {
        Serial.println("Detected Color: BLUE");
      }
      else {
        Serial.println("Color unclear or mixed.");
      }

      // Stop after one detection
      while (1);
    }
  }
}

float measureVoltage(int pin) {
  delay(100);  // Wait for signal to stabilize
  int raw = analogRead(pin);
  float voltage_mv = (raw * 5000.0) / 1023.0;
  return voltage_mv;
}
const int voltagePin = A0;
const int groundPin = A5;

const int readingRed = 2;
const int readingGreen = 3;
const int readingBlue = 4;

float redVoltageInitial = 0;
float greenVoltageInitial = 0;
float blueVoltageInitial = 0;

float redVoltage = 0;
float greenVoltage = 0;
float blueVoltage = 0;

bool firstIteration = true;

void setup() {
  Serial.begin(9600);
  pinMode(readingRed, INPUT);
  pinMode(readingGreen, INPUT);
  pinMode(readingBlue, INPUT);
  Serial.println("Keep LEDs running, NO BALL placed now...");
}

void loop() {
  if (firstIteration) {
    // Store initial readings for all 3 colors

    if (digitalRead(readingGreen) == HIGH) {
      greenVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Green: ");
      Serial.println(greenVoltageInitial);
    }

    else if (digitalRead(readingBlue) == HIGH) {
      blueVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Blue: ");
      Serial.println(blueVoltageInitial);
    }
    else {
      redVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Red: ");
      Serial.println(redVoltageInitial);
    }

    // Check if all 3 are captured
    if (redVoltageInitial > 0 && greenVoltageInitial > 0 && blueVoltageInitial > 0) {
      Serial.println("Initial measurements complete. Place the ball now...");
      delay(5000);  // give user time to place the ball
      firstIteration = false;
    }
  }
  else {


    if (digitalRead(readingGreen) == HIGH) {
      greenVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Green: ");
      Serial.println(greenVoltage);
    }

    else if (digitalRead(readingBlue) == HIGH) {
      blueVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Blue: ");
      Serial.println(blueVoltage);
    }

    else{
      redVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Red: ");
      Serial.println(redVoltage);
    }

    // Only proceed if all 3 colors were read
    if (redVoltage > 0 && greenVoltage > 0 && blueVoltage > 0) {
      float diffRed = redVoltage - redVoltageInitial;
      float diffGreen = greenVoltage - greenVoltageInitial;
      float diffBlue = blueVoltage - blueVoltageInitial;

      Serial.println("Voltage Differences:");
      Serial.print("Red Diff: "); Serial.println(diffRed);
      Serial.print("Green Diff: "); Serial.println(diffGreen);
      Serial.print("Blue Diff: "); Serial.println(diffBlue);

      if (diffRed > diffGreen && diffRed > diffBlue) {
        Serial.println("Detected Color: RED");
      }
      else if (diffGreen > diffRed && diffGreen > diffBlue) {
        Serial.println("Detected Color: GREEN");
      }
      else if (diffBlue > diffRed && diffBlue > diffGreen) {
        Serial.println("Detected Color: BLUE");
      }
      else {
        Serial.println("Color unclear or mixed.");
      }

      // Stop after one detection
      while (1);
    }
  }
}

float measureVoltage(int pin) {
  delay(100);  // Wait for signal to stabilize
  int raw = analogRead(pin);
  float voltage_mv = (raw * 5000.0) / 1023.0;
  return voltage_mv;
}
const int voltagePin = A0;
const int groundPin = A5;

const int readingRed = 2;
const int readingGreen = 3;
const int readingBlue = 4;

float redVoltageInitial = 0;
float greenVoltageInitial = 0;
float blueVoltageInitial = 0;

float redVoltage = 0;
float greenVoltage = 0;
float blueVoltage = 0;

bool firstIteration = true;

void setup() {
  Serial.begin(9600);
  pinMode(readingRed, INPUT);
  pinMode(readingGreen, INPUT);
  pinMode(readingBlue, INPUT);
  Serial.println("Keep LEDs running, NO BALL placed now...");
}

void loop() {
  if (firstIteration) {
    // Store initial readings for all 3 colors

    if (digitalRead(readingGreen) == HIGH) {
      greenVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Green: ");
      Serial.println(greenVoltageInitial);
    }

    else if (digitalRead(readingBlue) == HIGH) {
      blueVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Blue: ");
      Serial.println(blueVoltageInitial);
    }
    else {
      redVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Red: ");
      Serial.println(redVoltageInitial);
    }

    // Check if all 3 are captured
    if (redVoltageInitial > 0 && greenVoltageInitial > 0 && blueVoltageInitial > 0) {
      Serial.println("Initial measurements complete. Place the ball now...");
      delay(5000);  // give user time to place the ball
      firstIteration = false;
    }
  }
  else {


    if (digitalRead(readingGreen) == HIGH) {
      greenVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Green: ");
      Serial.println(greenVoltage);
    }

    else if (digitalRead(readingBlue) == HIGH) {
      blueVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Blue: ");
      Serial.println(blueVoltage);
    }

    else{
      redVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Red: ");
      Serial.println(redVoltage);
    }

    // Only proceed if all 3 colors were read
    if (redVoltage > 0 && greenVoltage > 0 && blueVoltage > 0) {
      float diffRed = redVoltage - redVoltageInitial;
      float diffGreen = greenVoltage - greenVoltageInitial;
      float diffBlue = blueVoltage - blueVoltageInitial;

      Serial.println("Voltage Differences:");
      Serial.print("Red Diff: "); Serial.println(diffRed);
      Serial.print("Green Diff: "); Serial.println(diffGreen);
      Serial.print("Blue Diff: "); Serial.println(diffBlue);

      if (diffRed > diffGreen && diffRed > diffBlue) {
        Serial.println("Detected Color: RED");
      }
      else if (diffGreen > diffRed && diffGreen > diffBlue) {
        Serial.println("Detected Color: GREEN");
      }
      else if (diffBlue > diffRed && diffBlue > diffGreen) {
        Serial.println("Detected Color: BLUE");
      }
      else {
        Serial.println("Color unclear or mixed.");
      }

      // Stop after one detection
      while (1);
    }
  }
}

float measureVoltage(int pin) {
  delay(100);  // Wait for signal to stabilize
  int raw = analogRead(pin);
  float voltage_mv = (raw * 5000.0) / 1023.0;
  return voltage_mv;
}
const int voltagePin = A0;
const int groundPin = A5;

const int readingRed = 2;
const int readingGreen = 3;
const int readingBlue = 4;

float redVoltageInitial = 0;
float greenVoltageInitial = 0;
float blueVoltageInitial = 0;

float redVoltage = 0;
float greenVoltage = 0;
float blueVoltage = 0;

bool firstIteration = true;

void setup() {
  Serial.begin(9600);
  pinMode(readingRed, INPUT);
  pinMode(readingGreen, INPUT);
  pinMode(readingBlue, INPUT);
  Serial.println("Keep LEDs running, NO BALL placed now...");
}

void loop() {
  if (firstIteration) {
    // Store initial readings for all 3 colors

    if (digitalRead(readingGreen) == HIGH) {
      greenVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Green: ");
      Serial.println(greenVoltageInitial);
    }

    else if (digitalRead(readingBlue) == HIGH) {
      blueVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Blue: ");
      Serial.println(blueVoltageInitial);
    }
    else {
      redVoltageInitial = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Initial Red: ");
      Serial.println(redVoltageInitial);
    }

    // Check if all 3 are captured
    if (redVoltageInitial > 0 && greenVoltageInitial > 0 && blueVoltageInitial > 0) {
      Serial.println("Initial measurements complete. Place the ball now...");
      delay(5000);  // give user time to place the ball
      firstIteration = false;
    }
  }
  else {


    if (digitalRead(readingGreen) == HIGH) {
      greenVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Green: ");
      Serial.println(greenVoltage);
    }

    else if (digitalRead(readingBlue) == HIGH) {
      blueVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Blue: ");
      Serial.println(blueVoltage);
    }

    else{
      redVoltage = measureVoltage(voltagePin) - measureVoltage(groundPin);
      Serial.print("Red: ");
      Serial.println(redVoltage);
    }

    // Only proceed if all 3 colors were read
    if (redVoltage > 0 && greenVoltage > 0 && blueVoltage > 0) {
      float diffRed = redVoltage - redVoltageInitial;
      float diffGreen = greenVoltage - greenVoltageInitial;
      float diffBlue = blueVoltage - blueVoltageInitial;

      Serial.println("Voltage Differences:");
      Serial.print("Red Diff: "); Serial.println(diffRed);
      Serial.print("Green Diff: "); Serial.println(diffGreen);
      Serial.print("Blue Diff: "); Serial.println(diffBlue);

      if (diffRed > diffGreen && diffRed > diffBlue) {
        Serial.println("Detected Color: RED");
      }
      else if (diffGreen > diffRed && diffGreen > diffBlue) {
        Serial.println("Detected Color: GREEN");
      }
      else if (diffBlue > diffRed && diffBlue > diffGreen) {
        Serial.println("Detected Color: BLUE");
      }
      else {
        Serial.println("Color unclear or mixed.");
      }

      // Stop after one detection
      while (1);
    }
  }
}

float measureVoltage(int pin) {
  delay(100);  // Wait for signal to stabilize
  int raw = analogRead(pin);
  float voltage_mv = (raw * 5000.0) / 1023.0;
  return voltage_mv;
}
