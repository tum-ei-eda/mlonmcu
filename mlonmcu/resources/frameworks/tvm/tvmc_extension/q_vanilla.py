# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from tvm.driver.tvmc.extensions import TVMCExtension  # noqa: E402
from QVanilla_Accelerator.backend import QVanillaAcceleratorBackend  # noqa: E402


class QVanillaExtension(TVMCExtension):
    def __init__(self):
        self.backend = QVanillaAcceleratorBackend()

    def uma_backends(self):
        return [self.backend]
