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
import io
import csv
import ast


class Metrics:
    def __init__(self):
        self.data = {}
        self.optional_keys = []

    @staticmethod
    def from_csv(text):
        # TODO: how to find out data types? -> pandas?
        lines = text.splitlines()
        assert len(lines) == 2, "Metrics should only have two lines"
        # headers, data = lines[0].split(","), lines[1]
        reader = csv.DictReader(lines)
        data = list(reader)[0]
        ret = Metrics()
        ret.data = data
        return ret

    def add(self, name, value, optional=False):
        assert name not in self.data, "Collumn with the same name already exists in metrics"
        self.data[name] = value
        if optional:
            self.optional_keys.append(name)

    def get_data(self, include_optional=False):
        return {
            key: (ast.literal_eval(value) if len(value) > 0 else None) if isinstance(value, str) else value
            for key, value in self.data.items()
            if key not in self.optional_keys or include_optional
        }

    def to_csv(self, include_optional=False):
        data = self.get_data(include_optional=include_optional)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
        return output.getvalue()
