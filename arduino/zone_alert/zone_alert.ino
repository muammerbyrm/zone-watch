// zone_alert.ino
// zone-watch uygulamasindan gelen "ZONE_ENTRY" / "ZONE_CLEAR"
// satirlarini okuyup bir LED ve (istege bagli) buzzer ile
// bildirim veren basit bir ornek sketch.
//
// Baglanti (varsayilan):
//   LED    -> pin 8 (seri direncle)
//   Buzzer -> pin 9 (istege bagli, pasif buzzer)

const int LED_PIN = 8;
const int BUZZER_PIN = 9;
const long BAUD_RATE = 115200;

String incoming = "";

void setup()
{
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.begin(BAUD_RATE);
}

void loop()
{
  while (Serial.available())
  {
    char c = Serial.read();
    if (c == '\n')
    {
      incoming.trim();
      handleEvent(incoming);
      incoming = "";
    }
    else
    {
      incoming += c;
    }
  }
}

void handleEvent(const String& event)
{
  if (event == "ZONE_ENTRY")
  {
    digitalWrite(LED_PIN, HIGH);
    tone(BUZZER_PIN, 1200, 150); // kisa uyari sesi
  }
  else if (event == "ZONE_CLEAR")
  {
    digitalWrite(LED_PIN, LOW);
  }
}
