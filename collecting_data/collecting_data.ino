#include "Arduino_BHY2.h"
#include "Nicla_System.h"

SensorXYZ accel(SENSOR_ID_ACC);
SensorXYZ gyro(SENSOR_ID_GYRO);


#define SAMPLE_RATE 100
#define DELAY       (1000.0 / (float) SAMPLE_RATE)


uint32_t numSamples = 2 * SAMPLE_RATE;
uint32_t threshold = 800;
uint32_t samplesRead = numSamples;

void setup()
{
  Serial.begin(1000000);
  while(!Serial);

  nicla::begin();
  nicla::leds.begin();

  if (!BHY2.begin(NICLA_I2C))
  {
    Serial.println("Failed to initialize BHY2!");
    while (1);
  }
  
  bool init_success = accel.begin(SAMPLE_RATE);
  init_success &= gyro.begin(SAMPLE_RATE);

  if (!init_success)
  {
    Serial.println("Failed to initialize sensors!");
    while (1);
  }

  Serial.println("acc_X,acc_Y,acc_Z,gyro_X,gyro_Y,gyro_Z");
}

void loop()
{
  nicla::leds.setColor(red);
  
  while (samplesRead == numSamples) 
  {
    BHY2.update();
    
    static auto printTime = millis();
    if (millis() - printTime >= DELAY) 
    {
      printTime = millis();

      int32_t x = gyro.x();
      int32_t y = gyro.y();
      int32_t z = gyro.z();
   
      uint32_t sum = abs(x) + abs(y) + abs(z);
      if (sum >= threshold)
      {
        samplesRead = 0;
      }
    }
  }

  nicla::leds.setColor(blue);

  while (samplesRead < numSamples) 
  {
    BHY2.update();
    
    static auto printTime = millis();
    if (millis() - printTime >= DELAY) 
    {
      printTime = millis();

      int32_t ax = accel.x();
      int32_t ay = accel.y();
      int32_t az = accel.z();
      
      int32_t gx = gyro.x();
      int32_t gy = gyro.y();
      int32_t gz = gyro.z();
   
      samplesRead++;

      Serial.print(ax);
      Serial.print(',');
      Serial.print(ay);
      Serial.print(',');
      Serial.print(az);
      Serial.print(',');
      Serial.print(gx);
      Serial.print(',');
      Serial.print(gy);
      Serial.print(',');
      Serial.print(gz);
      Serial.println();

      if (samplesRead >= numSamples)
      {
        Serial.println();
        Serial.println();
        
        nicla::leds.setColor(off);
        delay(1000);
      }
    }
  }
  
}
