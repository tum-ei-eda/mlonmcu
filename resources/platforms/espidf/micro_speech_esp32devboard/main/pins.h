#ifndef MICRO_KWS_PINS_H_
#define MICRO_KWS_PINS_H_

void gpio_init(void);

#if defined(CONFIG_IDF_TARGET_ESP32C3)

#define I2S_SCK_PIN 32
#define I2S_WS_PIN 25
#define I2S_DATA_IN_PIN 33
#define I2S_PORT_ID 0

#define GPIO_LED_RED ((gpio_num_t)3)
#define GPIO_LED_GREEN ((gpio_num_t)4)
#define GPIO_LED_BLUE ((gpio_num_t)5)

#elif defined(CONFIG_IDF_TARGET_ESP32)

#define I2S_SCK_PIN 8
#define I2S_WS_PIN 9
#define I2S_DATA_IN_PIN 10
#define I2S_PORT_ID 0

#else

#error "ESP-IDF target not supported. Please provide information in pin_def.h"

#endif

#endif  // MICRO_KWS_PINS_H_
