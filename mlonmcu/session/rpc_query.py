#
# Copyright (c) 2024 TUM Department of Electrical and Computer Engineering.
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
"""Print the state of an MLonMCU RPC tracker: `python -m mlonmcu.session.rpc_query host:port [interval]`."""

import argparse
import socket
import time

import mlonmcu.session.rpc_utils as base


def get_summary(sock):
    base.sendjson(sock, {"action": "summary"})
    value = base.recvjson(sock)
    if value.get("code") != "success":
        raise RuntimeError(f"Invalid return value {value}")
    return value["data"]


def format_summary(data):
    servers = sorted(data["server_info"], key=lambda x: (x["key"], x["addr"]))
    lines = ["Server List", "-" * 40, f"{'server-address':<24}{'key':<10}status", "-" * 40]
    for item in servers:
        lines.append(f"{':'.join(map(str, item['addr'])):<24}{item['key']:<10}{item['status']}")
    lines += ["-" * 40, "", "Queue Status"]
    queue = data["queue_info"]
    width = max([len(k) for k in queue] + [3])
    header = f"{'key':<{width}}   total  free  pending"
    lines += ["-" * len(header), header, "-" * len(header)]
    for key in sorted(queue):
        info = queue[key]
        lines.append(f"{key:<{width}}   {info['total']:<5d}  {info['free']:<4d}  {info['pending']:<7d}")
    lines.append("-" * len(header))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Query an MLonMCU RPC tracker")
    parser.add_argument("tracker", help="host:port")
    parser.add_argument("interval", nargs="?", type=float, default=0, help="refresh period in seconds (0: once)")
    args = parser.parse_args(argv)
    host, port = args.tracker.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=10) as sock:
        try:
            while True:
                print(format_summary(get_summary(sock)), flush=True)
                if not args.interval:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
