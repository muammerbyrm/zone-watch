# zone-watch

Webcam üzerinden gerçek zamanlı insan tespiti yapan ve tanımlı bir
**izleme bölgesine** (örn. bir kapı ya da pencere önü) giriş olduğunda
seri port üzerinden bağlı bir **Arduino/ESP32** kartına olay bildirimi
gönderen basit bir güvenlik/embedded sistem örneği.

## Nasıl çalışır?

1. Webcam'den gelen görüntü [YOLOv8n-Pose](https://docs.ultralytics.com/tasks/pose/)
   modeliyle (ONNX Runtime üzerinden) işlenir, kişiler tespit edilir.
2. Her tespit edilen kişinin kutusu, koddaki `Config::ZONE_*` sabitleriyle
   tanımlanan dikdörtgen bölgeyle karşılaştırılır.
3. Bölgeyle yeterli örtüşme oranı olan bir kişi varsa ve bu durum birkaç
   ardışık karede tutarlıysa (titremeyi önlemek için), seri porttan
   `ZONE_ENTRY` / `ZONE_CLEAR` satırları gönderilir.
4. `arduino/zone_alert` altındaki örnek sketch bu satırları okuyup bir
   LED/buzzer ile bildirim verir — dilediğin gibi röle, kapı kilidi,
   alarm modülü vb. ile değiştirebilirsin.

## Gereksinimler

- Windows + Visual Studio (proje `windows.h` ve `DCB`/`CreateFileA` gibi
  Windows seri port API'lerini kullanır)
- [OpenCV](https://opencv.org/) (core, imgproc, highgui, dnn modülleri)
- [ONNX Runtime](https://onnxruntime.ai/) (CPU ya da CUDA execution provider)
- `yolov8n-pose.onnx` modeli (Ultralytics'ten `.pt` -> `.onnx` export edilebilir)
- Arduino/ESP32 + Arduino IDE (donanım tarafı için, opsiyonel)

## Kurulum

```bash
# Ultralytics ile modeli export etmek icin (Python tarafinda, bir kere):
pip install ultralytics
yolo export model=yolov8n-pose.pt format=onnx imgsz=320
```

`yolov8n-pose.onnx` dosyasını derlenen `.exe` ile aynı klasöre koy.

## Yapılandırma

`main.cpp` içindeki `Config` namespace'inden:

- `CAMERA_INDEX` — birden fazla kameran varsa index'i değiştir
- `ZONE_X/Y/W/H` — izlenecek bölgenin piksel koordinatları (uygulamayı
  çalıştırıp görüntüye bakarak kendi sahnene göre ayarla)
- `SERIAL_PORT` — Arduino/ESP32'nin bağlı olduğu COM portu
- `CONF_THRESH`, `ZONE_OVERLAP_RATIO`, `STABLE_FRAMES_REQUIRED` — hassasiyet
  ayarları

## Donanım bağlantısı (örnek)

```
Arduino Uno       Bileşen
-----------       -------
Pin 8       ---->  LED (+ seri direnç)
Pin 9       ---->  Pasif buzzer (opsiyonel)
GND         ---->  Ortak GND
```

`arduino/zone_alert/zone_alert.ino` dosyasını kartına yükle, ardından
Windows tarafında uygulamayı çalıştır.

## Notlar

- Bu proje bir **varlık/giriş tespiti** demosudur; gerçek bir güvenlik
  sisteminde kullanılacaksa aydınlatma koşulları, yanlış pozitif oranı ve
  gizlilik/KVKK gereksinimleri ayrıca değerlendirilmelidir.
- Kişilerin görüntüsünü işleyen her sistemde olduğu gibi, kayıt tutuyorsan
  ilgili mevzuata (KVKK/GDPR) uygunluğu kontrol et.

## Lisans

MIT — bkz. `LICENSE`.
