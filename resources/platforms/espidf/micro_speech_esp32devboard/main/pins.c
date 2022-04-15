#include "pins.h"

void gpio_init()
{
#if defined(GPIO_LED_RED) && defined(GPIO_LED_GREEN) && defined(GPIO_LED_BLUE)
    gpio_reset_pin(GPIO_LED_RED);
    gpio_set_direction(GPIO_LED_RED, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_LED_GREEN);
    gpio_set_direction(GPIO_LED_GREEN, GPIO_MODE_OUTPUT);
    gpio_reset_pin(GPIO_LED_BLUE);
    gpio_set_direction(GPIO_LED_BLUE, GPIO_MODE_OUTPUT);
#endif
}

