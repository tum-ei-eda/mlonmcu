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
        self.order = []

    @staticmethod
    def from_csv(text):
        # TODO: how to find out data types? -> pandas?
        lines = text.splitlines()
        assert len(lines) == 2, "Metrics should only have two lines"
        # headers, data = lines[0].split(","), lines[1]
        empty = False
        for line in lines:
            if len(line.strip()) == 0:
                empty = True
        ret = Metrics()
        if not empty:
            reader = csv.DictReader(lines)
            data = list(reader)[0]
            data_ = {}
            for key, value in data.items():
                if len(key) > 2 and key[0] == key[-1] == "_":
                    new_key = key[1:-1]
                    data_[new_key] = value
                    ret.optional_keys.append(new_key)
                else:
                    data_[key] = value
            ret.data = data_
            ret.order = list(data_.keys())
        return ret

    def add(self, name, value, optional=False, overwrite=False, prepend=False):
        if not overwrite:
            assert name not in self.data, "Column with the same name already exists in metrics"
        self.data[name] = value
        if optional:
            self.optional_keys.append(name)
        if prepend:
            self.order.insert(0, name)
        else:
            self.order.append(name)

    def get(self, name):
        value = self.data[name]
        return (ast.literal_eval(value) if len(value) > 0 else None) if isinstance(value, str) else value

    def has(self, name):
        return name in self.data

    def get_data(self, include_optional=False, identify_optional=False):
        return {
            f"_{key}_" if identify_optional and key in self.optional_keys else key: self.get(key)
            for key in self.order
            if key not in self.optional_keys or include_optional
        }

    def to_csv(self, include_optional=False):
        data = self.get_data(include_optional=include_optional, identify_optional=True)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)
        return output.getvalue()
