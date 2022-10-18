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


# this file contains read lock and write lock classes based on filelock
# the two classes are non-blocking


import filelock
from filelock import FileLock
import random  # this is used to create a identifier for every ReadFileLock and WriteFileLock instance.
import string  # this is used to create a identifier for every ReadFileLock and WriteFileLock instance.
import time  # this is used to track the lock actions
import os
import yaml
from pathlib import Path
import atexit


# data structure of the file which serves as the lock


class ReadFileLock:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.lock = FileLock(self.filepath)
        random.seed(time.time())
        self.id = "".join([random.choice(string.ascii_letters + string.digits) for n in range(32)])

    def acquire(self, timeout=10):
        """:return: True means success. False means fail"""
        # the process is the following:
        # 1. aquire filelock
        # 2. read the lock occupation situation
        # 3. check if the lock is already occupied by another write process
        # 4.1. relase filelock and raise exception if the lock is already occupied by another write process
        # 4.2. otherwise write the updated lock occupation situation back, release filelock and return

        # 1. aquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.filepath.parent / "lock_track"):
            # create a file to tack the lock occupation situation
            with open(self.filepath.parent / "lock_track", "w") as track_file:
                pass

        with open(self.filepath.parent / "lock_track", "r") as stream:
            # read the current lock occupation situation
            try:
                lock_occupy_info = yaml.safe_load(stream) or {}
            except yaml.YAMLError as exc:
                print(exc)

        # 3. check if the lock is already occupied by another write process
        write_occupied = False
        for k, v in lock_occupy_info.items():
            if v["type"] == "write":
                write_occupied = True
                break

        # 4
        if write_occupied:
            lock_acquire_success = False
            self.lock.release()
            raise filelock.Timeout(self.lock)  # mimic filelock TimeOut Error
        else:
            lock_occupy_info[self.id] = {"time": "<time>", "type": "read"}
            with open(self.filepath.parent / "lock_track", "w") as track_file:
                yaml.dump(lock_occupy_info, track_file, default_flow_style=False)
            lock_acquire_success = True

        self.lock.release()
        atexit.register(self.release())
        return lock_acquire_success

    def release(self):
        # the process is the following:
        # 1. aquire filelock
        # 2. read the lock occupation situation
        # 3. delete the record of self.id
        # 4. write the updated lock occupation situation back, release filelock and return

        # 1. aquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.filepath.parent / "lock_track"):
            # raise error if file does not exist
            raise RuntimeError("track file for read write log is missing.")

        with open(self.filepath.parent / "lock_track", "r") as stream:
            # read the current lock occupation situation
            try:
                lock_occupy_info = yaml.safe_load(stream) or {}
            except yaml.YAMLError as exc:
                print(exc)

        # 3. delete the record of self.id
        lock_occupy_info.pop(self.id)

        # 4. write the updated lock occupation situation back, release filelock and return
        with open(self.filepath.parent / "lock_track", "w") as track_file:
            if lock_occupy_info:
                yaml.dump(lock_occupy_info, track_file, default_flow_style=False)

        self.lock.release()

    @property
    def is_locked(self):
        """:return: True means success. False means fail"""
        # the process is the following:
        # 1. aquire filelock
        # 2. read the lock occupation situation
        # 3. check if the lock is already occupied by another write process

        # 1. aquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.filepath.parent / "lock_track"):
            # create a file to tack the lock occupation situation
            with open(self.filepath.parent / "lock_track", "w") as track_file:
                pass

        with open(self.filepath.parent / "lock_track", "r") as stream:
            # read the current lock occupation situation
            try:
                lock_occupy_info = yaml.safe_load(stream) or {}
            except yaml.YAMLError as exc:
                print(exc)

        # 3. check if the lock is already occupied by another write process
        write_occupied = False
        for k, v in lock_occupy_info.items():
            if v["type"] == "write":
                write_occupied = True
                break

        self.lock.release()
        return not write_occupied


class WriteFileLock:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.lock = FileLock(self.filepath)
        random.seed(time.time())
        self.id = "".join([random.choice(string.ascii_letters + string.digits) for n in range(32)])

    def acquire(self):
        """:return: True means success. False means fail"""
        # the process is the following:
        # 1. aquire filelock
        # 2. read the lock occupation situation
        # 3. check if the lock is already occupied by another process
        # 4.1. relase filelock and raise exception if the lock is already occupied by another write process
        # 4.2. otherwise write the updated lock occupation situation back, release filelock and return

        # 1. aquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.filepath.parent / "lock_track"):
            # create a file to tack the lock occupation situation
            with open(self.filepath.parent / "lock_track", "w") as track_file:
                pass

        with open(self.filepath.parent / "lock_track", "r") as stream:
            # read the current lock occupation situation
            try:
                lock_occupy_info = yaml.safe_load(stream) or {}
            except yaml.YAMLError as exc:
                print(exc)

        # 3. check if the lock is already occupied by another process
        occupied = False
        for k, v in lock_occupy_info.items():
            if v["type"] == "write" or "read":
                occupied = True
                break

        # 4
        if occupied:
            lock_acquire_success = False
            self.lock.release()
            raise filelock.Timeout(self.lock)  # mimic filelock TimeOut Error
        else:
            lock_occupy_info[self.id] = {"time": "<time>", "type": "write"}  # add new occupation info
            with open(self.filepath.parent / "lock_track", "w") as track_file:
                yaml.dump(lock_occupy_info, track_file, default_flow_style=False)
            lock_acquire_success = True

        self.lock.release()
        atexit.register(self.release())
        return lock_acquire_success

    def release(self):
        # the process is the following:
        # 1. aquire filelock
        # 2. read the lock occupation situation
        # 3. delete the record of self.id
        # 4. write the updated lock occupation situation back, release filelock and return

        # 1. aquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.filepath.parent / "lock_track"):
            # raise error if file does not exist
            raise RuntimeError("track file for read write log is missing.")

        with open(self.filepath.parent / "lock_track", "r") as stream:
            # read the current lock occupation situation
            try:
                lock_occupy_info = yaml.safe_load(stream) or {}
                if lock_occupy_info == None:
                    lock_occupy_info = {}
            except yaml.YAMLError as exc:
                print(exc)

        # 3. delete the record of self.id
        lock_occupy_info.pop(self.id)

        # 4. write the updated lock occupation situation back, release filelock and return
        with open(self.filepath.parent / "lock_track", "w") as track_file:
            if lock_occupy_info:
                yaml.dump(lock_occupy_info, track_file, default_flow_style=False)

        self.lock.release()

    def __del__(self):
        self.release()

    @property
    def is_locked(self):
        """:return: True means success. False means fail"""
        # the process is the following:
        # 1. aquire filelock
        # 2. read the lock occupation situation
        # 3. check if the lock is already occupied by another write process

        # 1. aquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.filepath.parent / "lock_track"):
            # create a file to tack the lock occupation situation
            with open(self.filepath.parent / "lock_track", "w") as track_file:
                pass

        with open(self.filepath.parent / "lock_track", "r") as stream:
            # read the current lock occupation situation
            try:
                lock_occupy_info = yaml.safe_load(stream) or {}
            except yaml.YAMLError as exc:
                print(exc)

        # 3. check if the lock is already occupied by another write process
        occupied = False
        for k, v in lock_occupy_info.items():
            if v["type"] == "write" or "read":
                occupied = True
                break

        self.lock.release()
        return occupied


if __name__ == "__main__":
    filepath = "hello"
    readlock1 = ReadFileLock(filepath)
    print("read1: " + readlock1.id)
    writelock1 = WriteFileLock(filepath)
    print("write1: " + writelock1.id)
    assert readlock1.acquire()
    assert not writelock1.acquire()
    time.sleep(2)
    readlock1.release()
    #
    # time.sleep(2)
    #
    # readlock1 = ReadFileLock(filepath)
    # print("read1: " + readlock1.id)
    # readlock2 = ReadFileLock(filepath)
    # print("read2: " + readlock2.id)
    # assert readlock1.acquire()
    # assert readlock2.acquire()
    # time.sleep(2)
    # readlock1.release()
    # readlock2.release()
    #
    # time.sleep(2)
    #
    # readlock1 = ReadFileLock(filepath)
    # print("read1: " + readlock1.id)
    # writelock1 = WriteFileLock(filepath)
    # print("write1: " + writelock1.id)
    # assert writelock1.acquire()
    # assert not readlock1.acquire()
    # time.sleep(2)
    # writelock1.release()

    # filepath = "hello"
    # readlock1 = ReadFileLock(filepath)
    # print("read1: " + readlock1.id)
    # print(readlock1.is_locked)

    pass
