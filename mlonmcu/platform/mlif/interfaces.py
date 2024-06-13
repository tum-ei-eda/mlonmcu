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
"""MLIF Interfaces"""
from mlonmcu.models.utils import fill_data_source_inputs_only

MAX_BATCH_SIZE = int(1e6)
DEFAULT_BATCH_SIZE = 10


def get_header():
    return """
#include "quantize.h"
#include "printing.h"
#include "exit.h"
// #include "ml_interface.h"
#include <cstring>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include  <stdio.h>

extern "C" {
int mlif_process_inputs(size_t, bool*);
int mlif_process_outputs(size_t);
void *mlif_input_ptr(int);
void *mlif_output_ptr(int);
int mlif_input_sz(int);
int mlif_output_sz(int);
int mlif_num_inputs();
int mlif_num_outputs();
}
"""


def get_top_rom(inputs_data):
    in_bufs = []
    for i, ins_data in enumerate(inputs_data):
        temp = []
        for j, in_data in enumerate(ins_data.values()):
            byte_data = in_data.tobytes()
            temp2 = ", ".join(["0x{:02x}".format(x) for x in byte_data] + [""])
            temp.append(temp2)
        in_bufs.append(temp)

    return fill_data_source_inputs_only(in_bufs)


def get_process_inputs_head():
    return """
int mlif_process_inputs(size_t batch_idx, bool *new_)
{
"""


def get_process_inputs_tail():
    return """
}
"""


def get_process_outputs_head():
    return """
int mlif_process_outputs(size_t batch_idx)
{
"""


def get_process_outputs_tail():
    return """
}
"""


def get_process_inputs_rom():
    return """
    *new_ = true;
    int num_inputs = mlif_num_inputs();
    for (int i = 0; i < num_inputs; i++)
    {
        int idx = num_inputs * batch_idx + i;
        int size = mlif_input_sz(i);
        char* model_input_ptr = (char*)mlif_input_ptr(i);
        if (idx >= num_data_buffers_in)
        {
            *new_ = false;
            break;
        }
        if (size != data_size_in[idx])
        {
            return EXIT_MLIF_INVALID_SIZE;
        }
        memcpy(model_input_ptr, data_buffers_in[idx], size);
    }
    return 0;
"""


def get_process_inputs_stdin_raw():
    return """
    char ch;
    *new_ = true;
    for (int i = 0; i < mlif_num_inputs(); i++)
    {
        int cnt = 0;
        int size = mlif_input_sz(i);
        char* model_input_ptr = (char*)mlif_input_ptr(i);
        while(read(STDIN_FILENO, &ch, 1) > 0) {
            // printf("c=%c / %d\\n", ch, ch);
            model_input_ptr[cnt] = ch;
            cnt++;
            if (cnt == size) {
                break;
            }
        }
        // printf("cnt=%d in_size=%lu\\n", cnt, in_size);
        if (cnt == 0) {
            *new_ = false;
            return 0;
        }
        else if (cnt < size)
        {
            return EXIT_MLIF_INVALID_SIZE;
        }
    }
    return 0;
"""


def get_process_outputs_stdout_raw():
    # TODO: maybe hardcode num_outputs and size here because we know it
    # and get rid of loop?
    return """
    for (int i = 0; i < mlif_num_outputs(); i++)
    {
        int8_t *model_output_ptr = (int8_t*)mlif_output_ptr(i);
        int size = mlif_output_sz(i);
        // TODO: move markers out of loop
        write(1, "-?-", 3);
        write(1, model_output_ptr, size);
        write(1, "-!-\\n" ,4);
    }
    return 0;
"""


class ModelSupport:
    def __init__(self, in_interface, out_interface, model_info, target=None, batch_size=None, inputs_data=None):
        self.model_info = model_info
        self.target = target
        self.inputs_data = inputs_data
        self.in_interface = in_interface
        self.out_interface = out_interface
        self.in_interface, self.batch_size = self.select_set_inputs_interface(in_interface, batch_size)
        self.out_interface, self.batch_size = self.select_get_outputs_interface(out_interface, self.batch_size)

    def select_set_inputs_interface(self, in_interface, batch_size):
        if in_interface == "auto":
            assert self.target is not None
            if self.target.supports_filesystem:
                in_interface = "filesystem"
            elif self.target.supports_stdin:
                in_interface = "stdin_raw"
                # TODO: also allow stdin?
            else:  # Fallback
                in_interface = "rom"
        assert in_interface in ["filesystem", "stdin", "stdin_raw", "rom"]
        if batch_size is None:
            if in_interface == "rom":
                batch_size = MAX_BATCH_SIZE  # all inputs are in already compiled into program
            else:
                batch_size = DEFAULT_BATCH_SIZE
        return in_interface, batch_size

    def select_get_outputs_interface(self, out_interface, batch_size):
        if out_interface == "auto":
            assert self.target is not None
            if self.target.supports_filesystem:
                out_interface = "filesystem"
            elif self.target.supports_stdin:
                out_interface = "stdout_raw"
                # TODO: also allow stdout?
            else:  # Fallback
                out_interface = "ram"
        assert out_interface in ["filesystem", "stdout", "stdout_raw", "ram"]
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        return out_interface, batch_size

    def generate_header(self):
        # TODO: make this configurable
        # TODO: do not require C++?
        return get_header()

    def generate_top(self):
        if self.in_interface == "rom":
            return get_top_rom(self.inputs_data)
        return ""

    def generate_bottom(self):
        return ""

    def generate_process_inputs_body(self):
        if self.in_interface == "rom":
            return get_process_inputs_rom()
        elif self.in_interface == "stdin_raw":
            return get_process_inputs_stdin_raw()
        raise NotImplementedError  # TODO: implement: filesystem (bin+npy), stdout

    def generate_process_outputs_body(self):
        if self.out_interface == "stdout_raw":
            return get_process_outputs_stdout_raw()
        raise NotImplementedError  # TODO: implement: filesystem (bin+npy), ram

    def generate_process_inputs(self):
        code = ""
        code += get_process_inputs_head()
        code += self.generate_process_inputs_body()
        code += get_process_inputs_tail()
        return code

    def generate_process_outputs(self):
        code = ""
        code += get_process_outputs_head()
        code += self.generate_process_outputs_body()
        code += get_process_outputs_tail()
        return code

    def generate(self):
        code = ""
        code += self.generate_header()
        code += self.generate_top()
        code += self.generate_process_inputs()
        code += self.generate_process_outputs()
        code += self.generate_bottom()
        return code
