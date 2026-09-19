/*
 * ESP32-CAM (AI-Thinker): MJPEG stream out, MQTT violation alert in.
 *
 *   http://<ip>:81/stream   MJPEG stream  -> the Python host reads this (source: in config)
 *   http://<ip>:81/capture  single JPEG
 *   MQTT  violations/<DEVICE_ID>  -> buzzer + LED for ALERT_MS
 *
 * Arduino IDE: board "AI Thinker ESP32-CAM", library "PubSubClient" (Nick O'Leary).
 * Flash through an FTDI adapter: IO0 -> GND while uploading, then remove it and reset.
 * Power it from a solid 5V/2A supply - Wi-Fi bursts brown out weak USB ports.
 *
 * NOTE: not compiled or run in the environment this was written in. Expect to adjust
 * for your board / core version (tested pattern: Arduino-ESP32 2.x/3.x).
 */
#include <WiFi.h>
#include <PubSubClient.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "config.h"

// AI-Thinker pin map
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

static const char *STREAM_CT = "multipart/x-mixed-replace;boundary=frame";
static const char *STREAM_BOUNDARY = "\r\n--frame\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

WiFiClient net;
PubSubClient mqtt(net);
String alertTopic = String("violations/") + DEVICE_ID;
volatile uint32_t alertUntil = 0;
uint32_t lastMqttTry = 0;

static esp_err_t streamHandler(httpd_req_t *req) {
  esp_err_t res = httpd_resp_set_type(req, STREAM_CT);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  char part[64];
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { res = ESP_FAIL; break; }
    size_t n = snprintf(part, sizeof(part), STREAM_PART, (unsigned)fb->len);
    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, part, n);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;  // client went away
  }
  return res;
}

static esp_err_t captureHandler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { httpd_resp_send_500(req); return ESP_FAIL; }
  httpd_resp_set_type(req, "image/jpeg");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

void startServer() {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = 81;
  cfg.ctrl_port += 1;
  httpd_uri_t stream = {"/stream", HTTP_GET, streamHandler, NULL};
  httpd_uri_t capture = {"/capture", HTTP_GET, captureHandler, NULL};
  httpd_handle_t h = NULL;
  if (httpd_start(&h, &cfg) == ESP_OK) {
    httpd_register_uri_handler(h, &stream);
    httpd_register_uri_handler(h, &capture);
  }
}

bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAME_SIZE;
  c.jpeg_quality = JPEG_QUALITY;
  c.fb_count = psramFound() ? 2 : 1;
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;  // always serve the newest frame, never a stale one
  return esp_camera_init(&c) == ESP_OK;
}

void onMessage(char *topic, byte *payload, unsigned int len) {
  // Any message on our topic is a confirmed violation; the host already did the reasoning.
  alertUntil = millis() + ALERT_MS;
  Serial.printf("ALERT %.*s\n", (int)len, (const char *)payload);
}

void mqttLoop() {
  if (mqtt.connected()) { mqtt.loop(); return; }
  if (millis() - lastMqttTry < 3000) return;  // non-blocking retry, streaming keeps running
  lastMqttTry = millis();
  String id = String("esp32cam-") + DEVICE_ID;
  bool ok = strlen(MQTT_USER) ? mqtt.connect(id.c_str(), MQTT_USER, MQTT_PASSWORD) : mqtt.connect(id.c_str());
  if (ok) { mqtt.subscribe(alertTopic.c_str(), 1); Serial.println("MQTT connected"); }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT); pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW); digitalWrite(LED_PIN, LOW);

  if (!initCamera()) { Serial.println("Camera init failed"); delay(2000); ESP.restart(); }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // power-save adds tens of ms of jitter to the stream
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print('.'); }
  Serial.printf("\nStream: http://%s:81/stream\n", WiFi.localIP().toString().c_str());

  startServer();
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMessage);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) { WiFi.reconnect(); delay(1000); return; }
  mqttLoop();
  bool on = (int32_t)(alertUntil - millis()) > 0;
  digitalWrite(LED_PIN, on);
  digitalWrite(BUZZER_PIN, on && ((millis() / 150) % 2 == 0));  // pulsing beep, steady LED
  delay(5);
}
