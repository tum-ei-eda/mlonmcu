#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""TODO"""

from tqdm import tqdm


def init_progress(total, msg="Processing..."):
    """Helper function to initialize a progress bar for the session."""
    return tqdm(
        total=total,
        desc=msg,
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}s]",
        leave=None,
        mininterval=0.001,
        # maxinterval=0,
    )


def update_progress(pbar, count=1):
    """Helper function to update the progress bar for the session."""
    pbar.update(count)


def get_pbar_callback(pbar):
    def callback(_):
        update_progress(pbar)
    return callback


def close_progress(pbar):
    """Helper function to close the session progressbar, if available."""
    if pbar:
        pbar.close()
