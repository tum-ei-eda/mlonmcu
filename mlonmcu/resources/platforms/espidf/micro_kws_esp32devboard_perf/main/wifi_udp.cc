#include "esp_log.h"
#include "esp_event.h"
#include "esp_wifi.h"
//#include "esp_eap_client.h" // Use the new header file for WPA2-Enterprise
#include "esp_http_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "nvs_flash.h"
#include "lwip/err.h"
#include "lwip/sys.h"

#include "wifi_udp.h"

#if CONFIG_ENABLE_WIFI

#define EXAMPLE_ESP_WIFI_SSID       CONFIG_ESP_WIFI_SSID
#define EXAMPLE_ESP_WIFI_PASS       CONFIG_ESP_WIFI_PASSWORD

#if CONFIG_ESP_WIFI_AUTH_OPEN
#define EXAMPLE_ESP_WIFI_AUTH_MODE  WIFI_AUTH_OPEN
#elif CONFIG_ESP_WIFI_AUTH_WEP
#define EXAMPLE_ESP_WIFI_AUTH_MODE  WIFI_AUTH_WEP
#elif CONFIG_ESP_WIFI_AUTH_WPA_PSK
#define EXAMPLE_ESP_WIFI_AUTH_MODE  WIFI_AUTH_WPA_PSK
#elif CONFIG_ESP_WIFI_AUTH_WPA2_PSK
#define EXAMPLE_ESP_WIFI_AUTH_MODE  WIFI_AUTH_WPA2_PSK
#elif CONFIG_ESP_WIFI_AUTH_WPA_WPA2_PSK
#define EXAMPLE_ESP_WIFI_AUTH_MODE  WIFI_AUTH_WPA_WPA2_PSK
#elif CONFIG_ESP_WIFI_AUTH_WPA3_PSK
#define EXAMPLE_ESP_WIFI_AUTH_MODE  WIFI_AUTH_WPA3_PSK
#elif CONFIG_ESP_WIFI_AUTH_WPA2_WPA3_PSK
#define EXAMPLE_ESP_WIFI_AUTH_MODE  WIFI_AUTH_WPA2_WPA3_PSK
#elif CONFIG_ESP_WIFI_AUTH_WAPI_PSK
#define EXAMPLE_ESP_WIFI_AUTH_MODE  WIFI_AUTH_WAPI_PSK
#endif

#define EXAMPLE_ESP_MAXIMUM_RETRY   CONFIG_ESP_WIFI_MAXIMUM_RETRY
#define UDP_SERVER_IP               CONFIG_ESP_UDP_SERVER_IP
#define UDP_SERVER_PORT             CONFIG_ESP_UDP_SERVER_PORT

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1
static const char *TAG = "wifi station";
static int s_retry_num = 0;



static void event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < EXAMPLE_ESP_MAXIMUM_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGI(TAG, "retry to connect to the AP");
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
        ESP_LOGI(TAG, "connect to the AP fail");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "got ip:" IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

void wifi_init(void) {
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    
    
    // Configuration for WPA2-Enterprise
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = EXAMPLE_ESP_WIFI_SSID, 
            .password = EXAMPLE_ESP_WIFI_PASS,
            .scan_method = WIFI_ALL_CHANNEL_SCAN,
            .bssid_set = false,
            .channel = 0,
            .listen_interval = 10,
        },
    };

    wifi_config.sta.threshold.authmode = EXAMPLE_ESP_WIFI_AUTH_MODE;


    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
   
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_sta finished.");

    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                           pdFALSE,
                                           pdFALSE,
                                           portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "connected to ap SSID:%s password:%s",
                 EXAMPLE_ESP_WIFI_SSID, EXAMPLE_ESP_WIFI_PASS);
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGI(TAG, "Failed to connect to SSID:%s, password:%s",
                 EXAMPLE_ESP_WIFI_SSID, EXAMPLE_ESP_WIFI_PASS);
    } else {
        ESP_LOGE(TAG, "UNEXPECTED EVENT");
    }
}

void udp_send(const void *message, uint16_t len) {
    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(UDP_SERVER_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(UDP_SERVER_PORT);

    int server_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (server_sock < 0) {
        ESP_LOGE(TAG, "Socket creation failed!");
        return;
    }

    ESP_LOGI(TAG, "Sending to %s:%d", UDP_SERVER_IP, UDP_SERVER_PORT);
    if(-1 == sendto(server_sock, message, len, 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr)))
    {
        ESP_LOGE(TAG, "Sent failed");
    }
    close(server_sock);
} 

void wifi_scan (void)
{
    uint16_t number = 10;
    wifi_ap_record_t ap_info[10];
    uint16_t ap_count = 0;
    wifi_scan_config_t scan_config = {
        .ssid = 0,
        .bssid = 0,
        .channel = 6,
        .show_hidden = true,
        .scan_type = WIFI_SCAN_TYPE_ACTIVE,
    };
    scan_config.scan_time.active.min = 120;
    scan_config.scan_time.active.max = 150;
    esp_wifi_scan_start(&scan_config, true);
    esp_wifi_scan_get_ap_records(&number, ap_info);
    esp_wifi_scan_get_ap_num(&ap_count);
    for (int i = 0; i< ap_count; i++) {
        ESP_LOGI(TAG, "SSID: %s, RSSI: %d, AUTHMODE; %d", ap_info[i].ssid, ap_info[i].rssi, ap_info[i].authmode);
    }
    if (0 == ap_count) { ESP_LOGE(TAG, "No Wifi Networks detected"); }
} 

/* UDP Send Task */
/* void udp_send_task(void *pvParameters) {
    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(UDP_SERVER_IP);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(UDP_SERVER_PORT);

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Socket creation failed!");
        printf("Socket creation failed");
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "Sending to %s:%d", UDP_SERVER_IP, UDP_SERVER_PORT);
    
    
    while (1) {
        char message[] = "Hello from ESP32!";
        sendto(sock, message, strlen(message), 0, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        ESP_LOGI(TAG, "Sent: %s", message);
        printf("Sent message");
        vTaskDelay(pdMS_TO_TICKS(2000));  // Send every 2 seconds
    }

    close(sock);
    vTaskDelete(NULL);
} */

/* This is a dummy function which is not used */
/* This is only used to prevent an exception from happening */
/*esp_err_t _http_event_handler(esp_http_client_event_t *evt) {
    switch(evt->event_id) {
        case HTTP_EVENT_ERROR:
            ESP_LOGD(TAG, "HTTP_EVENT_ERROR");
            break;
        case HTTP_EVENT_ON_CONNECTED:
            ESP_LOGD(TAG, "HTTP_EVENT_ON_CONNECTED");
            break;
        case HTTP_EVENT_HEADER_SENT:
            ESP_LOGD(TAG, "HTTP_EVENT_HEADER_SENT");
            break;
        case HTTP_EVENT_ON_HEADER:
            ESP_LOGD(TAG, "HTTP_EVENT_ON_HEADER, key=%s, value=%s", evt->header_key, evt->header_value);
            break;
        case HTTP_EVENT_ON_DATA:
            ESP_LOGD(TAG, "HTTP_EVENT_ON_DATA, len=%d", evt->data_len);
            if (!esp_http_client_is_chunked_response(evt->client)) {
                printf("%.*s", evt->data_len, (char*)evt->data);
            }
            break;
        case HTTP_EVENT_ON_FINISH:
            ESP_LOGD(TAG, "HTTP_EVENT_ON_FINISH");
            break;
        case HTTP_EVENT_DISCONNECTED:
            ESP_LOGD(TAG, "HTTP_EVENT_DISCONNECTED");
            break;
        case HTTP_EVENT_REDIRECT:
            ESP_LOGD(TAG, "HTTP_EVENT_REDIRECT");
            break;
    }
    return ESP_OK;
} */

/* This is a dummy function which is not used */
/* This is only used to prevent an exception from happening */
/* void send_detection_result(const char* keyword) {
    esp_http_client_config_t config = {
        .url = "http://10.152.31.110:5000/endpoint",
        .event_handler = _http_event_handler
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);

    esp_http_client_set_method(client, HTTP_METHOD_POST);
    esp_http_client_set_post_field(client, keyword, strlen(keyword));

    esp_err_t err = esp_http_client_perform(client);
    if (err == ESP_OK) {
        ESP_LOGD(TAG, "HTTP POST Status = %d, content_length = %lld",
                 esp_http_client_get_status_code(client),
                 esp_http_client_get_content_length(client));
    } else {
        ESP_LOGE(TAG, "HTTP POST request failed: %s", esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);
} */

#endif //CONFIG_ENABLE_WIFI
