
float sensorValue, reference; // Paramaters for function
float readRef();
float sense();

void setup()
{
    int samplingTime = 5;      // In ms
    float lastError = 0;       // This is just the delayed signal
    float controlAction = 0;   // Response
    float Kp = 3000, Kd = 800; // Parameters
}

void loop() 
{
    delay(samplingTime);
    alarm();
}

void alarm()
{
    float sensorValue = sense();
    float reference = readRef();
    float controlAction = control(sensorValue, reference);
    actuate(controlAction);
}

/* float control(float sensorValue, float reference)
 * Implements a PD controller for use on Arduino. Only this function is implemented.
 * Receives a sensor value and a set point or reference value.
 * Requires as global variables the last error, and the constants for the controller.
 * Params:
 *   @param sensorValue - Value read by the sensor on the Arduino
 *   @param reference - Reference value for the system.
 * Returns:
 *   float controlAction - Action that the controller proposes given its logic (PD)
 */
float control(float sensorValue, float reference)
{
    /* A PD controller is described in the following way:
     *   PD = Kp*E + Kd*E'
     *   Where:
     *   E = sensorValue - reference
     *   The derivative is approximated in the following way:
     *   E' = (E[n]-E[n-1]) / samplingTime
     *   PD is the controller action.
     */
    float error = sensorValue - reference;
    float prop = Kp * error;
    float der = Kd * (error - lastError) / samplingTime;
    lastError = error; // set for next iteration
    return prop + der;
}