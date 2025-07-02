/*
 * Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
 *
 * This file is part of the MicroKWS project.
 * See https://gitlab.lrz.de/de-tum-ei-eda-esl/ESD4ML/micro-kws for further
 * info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "model_settings.h"

const char* category_labels[category_count] = {
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 1
    CONFIG_MICRO_KWS_CLASS_LABEL_0,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 2
    CONFIG_MICRO_KWS_CLASS_LABEL_1,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 3
    CONFIG_MICRO_KWS_CLASS_LABEL_2,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 4
    CONFIG_MICRO_KWS_CLASS_LABEL_3,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 5
    CONFIG_MICRO_KWS_CLASS_LABEL_4,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 6
    CONFIG_MICRO_KWS_CLASS_LABEL_5,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 7
    CONFIG_MICRO_KWS_CLASS_LABEL_6,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 8
    CONFIG_MICRO_KWS_CLASS_LABEL_7,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 9
    CONFIG_MICRO_KWS_CLASS_LABEL_8,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 10
    CONFIG_MICRO_KWS_CLASS_LABEL_9,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 11
    CONFIG_MICRO_KWS_CLASS_LABEL_10,
#endif
#if CONFIG_MICRO_KWS_NUM_CLASSES >= 12
    CONFIG_MICRO_KWS_CLASS_LABEL_11,
#endif
};
