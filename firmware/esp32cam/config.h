#pragma once

// ---- edit these ----
#define WIFI_SSID      "your-wifi"
#define WIFI_PASSWORD  "your-password"

#define MQTT_HOST      "192.168.1.10"   // the machine running Mosquitto (or your HiveMQ host)
#define MQTT_PORT      1883
#define MQTT_USER      ""               // leave empty for an open local broker
#define MQTT_PASSWORD  ""

#define DEVICE_ID      "cam01"          // must match device_id in the Python config

// ---- alert outputs ----
// GPIO12 and GPIO15 are boot-strapping pins (a pull-up on GPIO12 can stop the board booting),
// so the buzzer/LED use GPIO13 and GPIO14. Do not use GPIO4: it drives the on-board flash LED.
#define BUZZER_PIN     13
#define LED_PIN        14
#define ALERT_MS       3000

// ---- camera ----
#define FRAME_SIZE     FRAMESIZE_VGA    // 640x480. Drop to FRAMESIZE_QVGA if frames arrive slowly.
#define JPEG_QUALITY   12               // 10 (best) .. 63 (smallest). Raise it first if Wi-Fi is the bottleneck.
