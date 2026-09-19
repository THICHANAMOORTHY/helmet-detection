# Helmet & Seatbelt Violation Detection

Implementation of `Helmet_Seatbelt_Violation_Detection_Guide.docx`:
ESP32-CAM → YOLOv8n inference host → rule engine → MQTT alert + local/Firebase log → web dashboard, with a mock E-Challan from number-plate OCR.

```
ESP32-CAM ──MJPEG──▶ vsd (Python) ──▶ rule engine ──▶ MQTT  violations/<deviceId> ──▶ ESP32-CAM buzzer + LED
 firmware/           detector.py        rules.py      alerts.py
                                            └──▶ snapshot + plate OCR + challan ──▶ SQLite (+ Firebase, optional)
                                                                                       └──▶ dashboard (Flask + HTML/JS)
```

## Try it now (no camera, no model)

```bash
pip install -r requirements.txt
python -m pytest                              # 24 tests
python -m vsd --simulate --no-mqtt --no-plate # then open http://localhost:8000
```

`--simulate` drives a synthetic camera and detector (bikes and cars crossing a road, one with a one-frame false detection) through the real rule engine, storage, alerting and dashboard. Everything it produces is badged **simulated**. It proves the plumbing, **not** detection accuracy.

## Real run

1. **Get data** – a public set from Roboflow Universe / Kaggle (both need a free account), plus your own footage:
   ```bash
   # Roboflow: "Download Dataset" -> format "YOLOv8", no augmentation -> unzip, then:
   python scripts/import_roboflow.py --src downloads/<export> --dst datasets/raw   # remaps to rider/helmet/no-helmet
   python scripts/extract_frames.py footage/junction.mp4 --out datasets/raw/images --fps 1
   # label 300–500 of your own frames (YOLO format, ids per data/helmet.yaml) into datasets/raw/{images,labels}
   python scripts/split_dataset.py --src datasets/raw --dst datasets/helmet   # 70/20/10, stratified by lighting
   ```
   The importer prints how many helmet/no-helmet boxes sit inside a rider box. Below ~80% the rule engine will miss violations (the dataset's "rider"/"bike" boxes don't include the head).
2. **Train** (helmet classes first, seatbelt later with `data/full.yaml`):
   ```bash
   python scripts/train.py --data data/helmet.yaml
   ```
   Runs `yolov8n.pt`, 100 epochs, 640 px; reports AP@0.5 per class on the test split, flags helmet/no-helmet below 0.80, and exports `models/helmet_seatbelt.onnx`. A 4 GB GPU handles batch 16; the installed PyTorch here is a CPU build, so install a CUDA build of PyTorch to train on the GPU (or use Colab).
3. **Flash the ESP32-CAM** – edit `firmware/esp32cam/config.h`, open `esp32cam.ino` in Arduino IDE (board *AI Thinker ESP32-CAM*, library *PubSubClient*). The serial monitor prints the stream URL.
4. **Broker** – run Mosquitto on the host (`mosquitto -v`) or use HiveMQ Cloud.
5. **Run**
   ```bash
   cp config.example.yaml config.yaml     # set source (ESP32 URL), MQTT host, location
   python -m vsd --config config.yaml
   ```
   Also works with `--source 0` (webcam) or `--source clip.mp4` (`--loop` to repeat) for the viva demo of a video clip.

Optional extras: `pip install easyocr` (plate OCR / E-Challan), `pip install firebase-admin` + `storage.firebase.enabled: true` (cloud mirror).

## Where each guide section lives

| Guide section | Code |
|---|---|
| Dataset & training | `scripts/`, `data/*.yaml` |
| Inference pipeline (latest-frame reader, thresholds, FPS/latency log) | `vsd/sources.py`, `vsd/detector.py`, `vsd/pipeline.py` |
| Violation rules (rider + no-helmet − helmet, 3-frame debounce, cabin region) | `vsd/rules.py` |
| MQTT event `violations/{deviceId}` + Firebase record | `vsd/alerts.py` |
| ESP32-CAM stream + buzzer/LED | `firmware/esp32cam/` |
| Dashboard: live feed, violation log, analytics | `vsd/dashboard/` |
| E-Challan: plate OCR + mock challan | `vsd/plate.py` |

Rule notes: a no-helmet box counts when it sits inside a rider box and no *more confident* helmet box is on the same head, so a helmeted pillion doesn't hide a bare-headed rider. A violation fires once per incident; `cooldown_s` sets how long it must be gone before the same spot can fire again. Tune `min_consecutive`, `max_gap`, per-class confidence and `cabin_roi` in the config.

## Changes from the guide

- **Buzzer/LED pins:** GPIO13 and GPIO14, not 12/13/14/15 as the guide lists. GPIO12 and GPIO15 are boot-strapping pins on the ESP32, and a pull-up on GPIO12 can prevent boot.
- **Storage:** local SQLite + image files by default, so the demo works without a Firebase project. The dashboard reads that store through a Flask API and Server-Sent Events, not the Firestore SDK. Firebase mirroring is opt-in.
- **Alert before OCR:** the MQTT alert is published the moment a violation is confirmed. OCR, snapshots and DB writes run on a worker thread, so a slow plate read never delays the buzzer.

## Verified vs not

Verified here: rule engine, plate normalisation, storage, MQTT payload/topic (against a fake client), the full simulated pipeline, dashboard in a browser, `extract_frames` / `split_dataset`, and `train.py` end to end (1 epoch on a tiny simulated set → evaluation → ONNX export → loaded by `YoloDetector`).

**Not verified** (no hardware, broker, credentials or dataset available): the ESP32 firmware (never compiled), a live MQTT broker round trip, Firebase, EasyOCR on real plates, and any real-world accuracy. No trained model ships with this repo.
