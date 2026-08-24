// zone-watch
// ------------------------------------------------------------
// Webcam uzerinden gercek zamanli insan tespiti yapar ve
// tanimli bir "izleme bolgesi" (ornegin bir kapi ya da pencere
// onu) icine biri girdiginde seri port uzerinden bagli bir
// Arduino/ESP32 kartina olay bildirimi gonderir.
//
// Kullanim alani: guvenlik/farkindalik amacli varlik-tespit
// sistemleri, embedded sistemlerle entegrasyon ornegi.
//
// Bagimliliklar: OpenCV, ONNX Runtime (CPU veya CUDA saglayici)
// Model: YOLOv8n-Pose (Ultralytics) .onnx formatinda
// ------------------------------------------------------------

#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>   // sadece NMSBoxes icin, inference icin degil
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <array>

// ------------------------------------------------------------
// Yapilandirma - ihtiyaca gore duzenle
// ------------------------------------------------------------
namespace Config
{
    constexpr int CAMERA_INDEX = 0;          // birden fazla kamera varsa index degistir
    constexpr int FRAME_W = 640;
    constexpr int FRAME_H = 480;
    constexpr int MODEL_INPUT = 320;         // YOLOv8n-Pose giris boyutu
    constexpr float CONF_THRESH = 0.35f;
    constexpr float NMS_THRESH = 0.45f;

    // Izlenecek bolge (ornegin kapi/pencere onu), piksel cinsinden
    // dikdortgen olarak tanimlanir. Kamerayi calistirip goruntuye
    // bakarak bu degerleri kendi sahnene gore ayarla.
    constexpr int ZONE_X = 220;
    constexpr int ZONE_Y = 80;
    constexpr int ZONE_W = 200;
    constexpr int ZONE_H = 320;

    // Bir tespitin "bolgede" sayilmasi icin, kisi kutusunun bolgeyle
    // kesisim oraninin en az bu kadar olmasi gerekir (0-1 arasi).
    constexpr float ZONE_OVERLAP_RATIO = 0.25f;

    // Titremeyi (flicker) onlemek icin ardisik kac karede ayni
    // sonuc gorulmesi gerektigi.
    constexpr int STABLE_FRAMES_REQUIRED = 5;

    constexpr const char* SERIAL_PORT = "\\\\.\\COM7";
    constexpr DWORD BAUD_RATE = CBR_115200;

    constexpr const wchar_t* MODEL_PATH = L"yolov8n-pose.onnx";
}

// ------------------------------------------------------------
// Seri porta veri yazan yardimci fonksiyon
// ------------------------------------------------------------
bool SendToSerial(HANDLE hSerial, const std::string& msg)
{
    if (hSerial == INVALID_HANDLE_VALUE) return false;
    DWORD bytesWritten = 0;
    BOOL ok = WriteFile(hSerial, msg.c_str(), static_cast<DWORD>(msg.size()), &bytesWritten, NULL);
    if (!ok || bytesWritten != msg.size())
    {
        std::cerr << "WriteFile error: " << GetLastError() << std::endl;
        return false;
    }
    FlushFileBuffers(hSerial);
    return true;
}

HANDLE OpenSerialPort(const char* portName, DWORD baudRate)
{
    HANDLE hSerial = CreateFileA(
        portName,
        GENERIC_READ | GENERIC_WRITE,
        0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
    );
    if (hSerial == INVALID_HANDLE_VALUE) return hSerial;

    DCB dcbSerialParams = { 0 };
    dcbSerialParams.DCBlength = sizeof(DCB);
    GetCommState(hSerial, &dcbSerialParams);
    dcbSerialParams.BaudRate = baudRate;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;
    dcbSerialParams.fBinary = TRUE;
    dcbSerialParams.fDtrControl = DTR_CONTROL_ENABLE;
    dcbSerialParams.fRtsControl = RTS_CONTROL_ENABLE;
    SetCommState(hSerial, &dcbSerialParams);

    COMMTIMEOUTS timeouts = { 0 };
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 1000;
    timeouts.WriteTotalTimeoutMultiplier = 50;
    SetCommTimeouts(hSerial, &timeouts);

    PurgeComm(hSerial, PURGE_RXCLEAR | PURGE_TXCLEAR);
    EscapeCommFunction(hSerial, SETDTR);
    EscapeCommFunction(hSerial, SETRTS);
    Sleep(50);

    return hSerial;
}

// ------------------------------------------------------------
// COCO 17-keypoint iskelet baglantilari (0-indexed) - sadece
// gorsellestirme icin kullanilir, zon mantigi kutu tabanlidir.
// ------------------------------------------------------------
const std::vector<std::pair<int, int>> SKELETON = {
    {15,13},{13,11},{16,14},{14,12},{11,12},
    {5,11},{6,12},{5,6},{5,7},{6,8},{7,9},{8,10},
    {1,2},{0,1},{0,2},{1,3},{2,4},{3,5},{4,6}
};

struct PoseDetection
{
    cv::Rect box;
    float conf;
    std::vector<cv::Point2f> keypoints;
    std::vector<float> kptConf;
};

// Goruntuyu oran korumadan kare model girisine sigdirir.
cv::Mat StretchResize(const cv::Mat& src, int targetSize, float& scaleX, float& scaleY)
{
    if (src.cols == targetSize && src.rows == targetSize)
    {
        scaleX = 1.0f;
        scaleY = 1.0f;
        return src;
    }
    scaleX = (float)targetSize / src.cols;
    scaleY = (float)targetSize / src.rows;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(targetSize, targetSize));
    return resized;
}

// YOLOv8-Pose ciktisini coz (output shape: [1, 56, 8400])
std::vector<PoseDetection> ParseYoloPose(const cv::Mat& output, float confThresh,
    float scaleX, float scaleY)
{
    std::vector<PoseDetection> results;
    std::vector<cv::Rect> boxesForNMS;
    std::vector<float> scoresForNMS;
    std::vector<PoseDetection> temp;

    cv::Mat dataT;
    cv::transpose(output, dataT); // 8400 x 56

    int numDet = dataT.rows;
    for (int i = 0; i < numDet; i++)
    {
        const float* row = dataT.ptr<float>(i);
        float objConf = row[4];
        if (objConf < confThresh) continue;

        float cx = row[0], cy = row[1], w = row[2], h = row[3];
        float x1 = (cx - w / 2) / scaleX;
        float y1 = (cy - h / 2) / scaleY;
        float x2 = (cx + w / 2) / scaleX;
        float y2 = (cy + h / 2) / scaleY;

        PoseDetection det;
        det.box = cv::Rect(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
        det.conf = objConf;

        for (int k = 0; k < 17; k++)
        {
            float kx = row[5 + k * 3 + 0];
            float ky = row[5 + k * 3 + 1];
            float kc = row[5 + k * 3 + 2];
            det.keypoints.push_back(cv::Point2f(kx / scaleX, ky / scaleY));
            det.kptConf.push_back(kc);
        }

        temp.push_back(det);
        boxesForNMS.push_back(det.box);
        scoresForNMS.push_back(objConf);
    }

    std::vector<int> keepIdx;
    cv::dnn::NMSBoxes(boxesForNMS, scoresForNMS, confThresh, Config::NMS_THRESH, keepIdx);

    for (int idx : keepIdx)
        results.push_back(temp[idx]);

    return results;
}

void DrawSkeleton(cv::Mat& frame, const PoseDetection& det, float kptConfThresh = 0.5f)
{
    cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 1);
    for (int i = 0; i < 17; i++)
        if (det.kptConf[i] > kptConfThresh)
            cv::circle(frame, det.keypoints[i], 2, cv::Scalar(0, 0, 255), -1);

    for (auto& pair : SKELETON)
    {
        int a = pair.first, b = pair.second;
        if (det.kptConf[a] > kptConfThresh && det.kptConf[b] > kptConfThresh)
            cv::line(frame, det.keypoints[a], det.keypoints[b], cv::Scalar(255, 200, 0), 1);
    }
}

// Kisi kutusunun izleme bolgesiyle kesisim oranini hesaplar
// (kesisim alani / kisi kutusu alani).
float ZoneOverlapRatio(const cv::Rect& personBox, const cv::Rect& zone)
{
    cv::Rect inter = personBox & zone;
    if (inter.area() <= 0 || personBox.area() <= 0) return 0.0f;
    return (float)inter.area() / (float)personBox.area();
}

int main()
{
    // ---------------- Webcam baslat ----------------
    cv::VideoCapture cap(Config::CAMERA_INDEX, cv::CAP_DSHOW);
    if (!cap.isOpened())
    {
        std::cout << "Kamera acilamadi (index " << Config::CAMERA_INDEX << ")" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, Config::FRAME_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, Config::FRAME_H);

    // ---------------- Seri port ----------------
    HANDLE hSerial = OpenSerialPort(Config::SERIAL_PORT, Config::BAUD_RATE);
    if (hSerial == INVALID_HANDLE_VALUE)
    {
        std::cout << "COM port acilamadi (seri port olmadan gorsel modda devam edilecek): "
            << GetLastError() << std::endl;
    }

    // ---------------- Model yukleme (ONNX Runtime) ----------------
    cv::setUseOptimized(true);
    cv::setNumThreads(cv::getNumberOfCPUs());

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "zone_watch");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try
    {
        OrtCUDAProviderOptions cudaOptions{};
        cudaOptions.device_id = 0;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        std::cout << "CUDA execution provider eklendi." << std::endl;
    }
    catch (const Ort::Exception& e)
    {
        std::cout << "CUDA provider eklenemedi, CPU'ya dusuluyor: " << e.what() << std::endl;
    }

    Ort::Session session(nullptr);
    try
    {
        session = Ort::Session(env, Config::MODEL_PATH, sessionOptions);
    }
    catch (const Ort::Exception& e)
    {
        std::cout << "Model yuklenemedi: " << e.what() << std::endl;
        if (hSerial != INVALID_HANDLE_VALUE) CloseHandle(hSerial);
        return -1;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNameAlloc = session.GetInputNameAllocated(0, allocator);
    auto outputNameAlloc = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> inputNames = { inputNameAlloc.get() };
    std::vector<const char*> outputNames = { outputNameAlloc.get() };

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const cv::Rect zone(Config::ZONE_X, Config::ZONE_Y, Config::ZONE_W, Config::ZONE_H);

    bool zoneOccupied = false;
    bool candidateState = false;
    int stableCounter = 0;
    int eventCounter = 0;

    auto prevTime = std::chrono::high_resolution_clock::now();

    std::cout << "Baslatildi. Cikmak icin ESC." << std::endl;

    while (true)
    {
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000)
            break;

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        float scaleX, scaleY;
        cv::Mat resizedInput = StretchResize(frame, Config::MODEL_INPUT, scaleX, scaleY);

        cv::Mat blob;
        cv::dnn::blobFromImage(resizedInput, blob, 1.0 / 255.0,
            cv::Size(Config::MODEL_INPUT, Config::MODEL_INPUT), cv::Scalar(0, 0, 0), true, false);

        std::array<int64_t, 4> inputShape = { 1, 3, Config::MODEL_INPUT, Config::MODEL_INPUT };
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, reinterpret_cast<float*>(blob.data), blob.total(),
            inputShape.data(), inputShape.size());

        auto outputTensors = session.Run(Ort::RunOptions{ nullptr },
            inputNames.data(), &inputTensor, 1, outputNames.data(), 1);

        float* outData = outputTensors[0].GetTensorMutableData<float>();
        auto outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int channels = (int)outShape[1];
        int numAnchors = (int)outShape[2];
        cv::Mat output(channels, numAnchors, CV_32F, outData);

        std::vector<PoseDetection> detections = ParseYoloPose(output, Config::CONF_THRESH, scaleX, scaleY);

        // Bolgeyi gorsellestir
        cv::rectangle(frame, zone, cv::Scalar(0, 165, 255), 2);

        bool anyoneInZone = false;
        for (auto& det : detections)
        {
            DrawSkeleton(frame, det);
            float overlap = ZoneOverlapRatio(det.box, zone);
            if (overlap >= Config::ZONE_OVERLAP_RATIO)
            {
                anyoneInZone = true;
                cv::rectangle(frame, det.box, cv::Scalar(0, 0, 255), 3);
            }
        }

        // Histerezis: durumu degistirmeden once N ardisik karede
        // ayni sonucun gorulmesini bekle (titremeyi onler)
        if (anyoneInZone != candidateState)
        {
            candidateState = anyoneInZone;
            stableCounter = 0;
        }
        else
        {
            stableCounter++;
        }

        if (stableCounter >= Config::STABLE_FRAMES_REQUIRED && zoneOccupied != candidateState)
        {
            zoneOccupied = candidateState;
            if (zoneOccupied)
            {
                eventCounter++;
                SendToSerial(hSerial, "ZONE_ENTRY\n");
                std::cout << "[Olay] Bolgeye girildi -> Sayac: " << eventCounter << std::endl;
            }
            else
            {
                SendToSerial(hSerial, "ZONE_CLEAR\n");
                std::cout << "[Olay] Bolge bosaldi" << std::endl;
            }
        }

        auto currTime = std::chrono::high_resolution_clock::now();
        double fps = 1.0 / std::chrono::duration<double>(currTime - prevTime).count();
        prevTime = currTime;

        cv::putText(frame, "FPS: " + std::to_string((int)fps) +
            "  Olay: " + std::to_string(eventCounter) +
            "  Durum: " + (zoneOccupied ? "GIRIS VAR" : "bos"),
            cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
            zoneOccupied ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 1);

        cv::imshow("zone-watch", frame);
        if (cv::waitKey(1) == 27) break;
    }

    if (hSerial != INVALID_HANDLE_VALUE) CloseHandle(hSerial);
    return 0;
}
