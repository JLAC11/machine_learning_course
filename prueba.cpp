#define PIN_INPUT 0
#define PIN_OUTPUT 3
// Se proponen valores iniciales de constantes, solo para correr el código
float Kp = 1;
float Kd = 1;
float last_err, dt, sensorValue, reference, controlAction;

void setup()
{
    reference = 0;
    last_err = 0;
    dt = 100;
}

void loop()
{
    // sensorValue = analogRead(A0);
    controlAction = control(sensorValue, reference);
}

float control(float sensorValue, float reference)
{
    float err = reference - sensorValue;
    float rate_err = (err - last_err) / dt;

    last_err = err;
    return Kp * err + Kd * rate_err;
}