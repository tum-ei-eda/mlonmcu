#include <stdint.h>
#include <stdio.h>
#include <cstring>

#include "posterior.h"

#include "esp_log.h"
#include "esp_timer.h"

/**
 * @brief Default constructor for posterior handler
 *
 * @param history_length Number of past model outputs to consider.
 * @param trigger_threshold_single Threshold value between 0 and 255 for moving average.
 * @param suppression_ms For how many ms a new detection should be ignored.
 * @param category_count Number of used labels.
 */
PosteriorHandler::PosteriorHandler(uint32_t history_length, uint8_t trigger_threshold_single,
                                   uint32_t suppression_ms, uint32_t category_count)
    : posterior_history_length_(history_length),
      posterior_trigger_threshold_(trigger_threshold_single * history_length),
      posterior_suppression_ms_(suppression_ms),
      posterior_category_count_(category_count) {

  /* ------------------------ */
  /* ENTER STUDENT CODE BELOW */
  /* ------------------------ */

  // Allocate memory 
  posterior_history_ = (uint8_t**)malloc(posterior_category_count_ * sizeof(uint8_t*));
  for (uint32_t i = 0; i < posterior_category_count_; i++) {
    posterior_history_[i] = (uint8_t*)calloc(posterior_history_length_, sizeof(uint8_t));
    }

  // Allocate array with moving sums
  posterior_moving_average_ = (uint32_t*)calloc(posterior_category_count_, sizeof(uint32_t));

  // Allocate array with the last trigger timestamps
  last_trigger_time_ = (uint32_t*)calloc(posterior_category_count_, sizeof(uint32_t));

  /* ------------------------ */
  /* ENTER STUDENT CODE ABOVE */
  /* ------------------------ */
}

/**
 * @brief Destructor for posterior handler class
 */
PosteriorHandler::~PosteriorHandler() {

  /* ------------------------ */
  /* ENTER STUDENT CODE BELOW */
  /* ------------------------ */

  // Free all memory
  for (uint32_t i = 0; i < posterior_category_count_; i++) {
    free(posterior_history_[i]);
  }
  free(posterior_history_);

  free(posterior_moving_average_);
  free(last_trigger_time_);

  /* ------------------------ */
  /* ENTER STUDENT CODE ABOVE */
  /* ------------------------ */
}

/**
 * @brief Implementation of the posterior handling algorithm.
 *
 * @param new_posteriors The raw model outputs with unsigned 8-bit values.
 * @param time_ms Timestamp for posterior handling (ms).
 * @param top_category_index The index of the detected category/label returned by pointer.
 * @param trigger Flag which should be raised to true if a new detection is available.
 *
 * @return ESP_OK if no error occurred.
 */
esp_err_t PosteriorHandler::Handle(uint8_t* new_posteriors, uint32_t time_ms,
                                   size_t* top_category_index, bool* trigger) {

  /* ------------------------ */
  /* ENTER STUDENT CODE BELOW */
  /* ------------------------ */

  // Base Case: No new detection
  *trigger = false;

  // Update moving avarage and history
  for (size_t i = 0; i < posterior_category_count_; i++) {
    // Sub old value from the moving average
    posterior_moving_average_[i] -= posterior_history_[i][0];
    // Add new sample
    posterior_moving_average_[i] += new_posteriors[i];

    // Shift history to left
    for (size_t j = 0; j < (posterior_history_length_ - 1); j++) {
          posterior_history_[i][j] = posterior_history_[i][j + 1];
        }

    // Put the new posterior at the end
    posterior_history_[i][posterior_history_length_ - 1] = new_posteriors[i];
  }

  // Compare and find the highest sum
  uint32_t best_sum = 0;
  size_t best_index = 0;
  for (size_t i = 0; i < posterior_category_count_; i++) {
    if (posterior_moving_average_[i] > best_sum) {
      best_sum = posterior_moving_average_[i];
      best_index = i;
    }
  }

  // Check the threashold
  if ((best_sum >= posterior_trigger_threshold_) &&
      ((time_ms - last_trigger_time_[best_index]) >= posterior_suppression_ms_)) {
    // If a new detection update supression time
    *trigger = true;
    *top_category_index = best_index;
    last_trigger_time_[best_index] = time_ms;
  }
  // else: leave *top_category_index as unchanged

  return ESP_OK;

  /* ------------------------ */
  /* ENTER STUDENT CODE ABOVE */
  /* ------------------------ */
}
