#ifndef WIFI_UDP_H
#define WIFI_UDP_H

#if CONFIG_ENABLE_WIFI
void wifi_init(void);
void send_detection_result(const char* keyword);
void udp_send(const void *message, uint16_t len);
void wifi_scan (void);
#endif //CONFIG_ENABLE_WIFI

#endif // WIFI_UDP_H
