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
import datetime
import os
import yaml
from pathlib import Path
import atexit


# data structure of the file which serves as the lock


class ReadFileLock:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.trackfilepath = self.filepath.parent / (str(self.filepath.stem) + "_track")
        self.lock = FileLock(self.filepath)
        random.seed(time.time())
        self.id = "".join([random.choice(string.ascii_letters + string.digits) for n in range(32)])

    def acquire(self, raise_exception=True):
        """
        This function tried to acquire a ReadFileLock.
        The process is the following:
        1. acquire filelock
        2. read the lock occupation situation
        3. check if the lock is already occupied by another write process
        4.1. release filelock and raise exception(or return 0) if the lock is already occupied by another write process
        4.2. otherwise write the updated lock occupation situation back, release filelock and return

            Parameters:
                raise_exception (bool): whether an exception should be raised when failed (default: True)

            Returns:
                success (bool): whether succeeded or not.
                    True means succeeded, False means failed (if the param raiseException is set to false).
                    A filelock.Timeout exception will be raised if failed
        """

        # 1. aquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.trackfilepath):
            # create a file to tack the lock occupation situation
            with open(self.trackfilepath, "w") as track_file:
                pass

        with open(self.trackfilepath, "r") as stream:
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
            if raise_exception:
                raise filelock.Timeout(self.lock)  # mimic filelock TimeOut Error
        else:
            time_info = datetime.datetime.now().replace(microsecond=0).isoformat()
            lock_occupy_info[self.id] = {"time": f"{time_info}", "type": "read"}
            with open(self.trackfilepath, "w") as track_file:
                yaml.dump(lock_occupy_info, track_file, default_flow_style=False)
            lock_acquire_success = True

        self.lock.release()
        atexit.register(self.release)
        return lock_acquire_success

    def release(self):
        """
        This function releases a ReadFileLock.
        the process is the following:
        1. acquire filelock
        2. read the lock occupation situation
        3. delete the record of self.id (no exception will be raised if self.id is not found in the record)
        4. write the updated lock occupation situation back, release filelock and return
        """

        # 1. aquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.trackfilepath):
            # raise error if file does not exist
            raise RuntimeError("track file for read write log is missing.")

        with open(self.trackfilepath, "r") as stream:
            # read the current lock occupation situation
            try:
                lock_occupy_info = yaml.safe_load(stream) or {}
            except yaml.YAMLError as exc:
                print(exc)

        # 3. delete the record of self.id
        if self.id in lock_occupy_info.keys():
            lock_occupy_info.pop(self.id)

        # 4. write the updated lock occupation situation back, release filelock and return
        with open(self.trackfilepath, "w") as track_file:
            if lock_occupy_info:
                yaml.dump(lock_occupy_info, track_file, default_flow_style=False)

        self.lock.release()

    @property
    def is_locked(self):
        """
        This property returns if a lock is occupied(locked) by other processes.
        the process is the following:
        1. acquire filelock
        2. read the lock occupation situation
        3. check if the lock is already occupied(locked) by another write process

            Returns:
                is_locked (bool): whether the lock is already occupied(locked) by another write process
        """

        # 1. acquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.trackfilepath):
            # create a file to tack the lock occupation situation
            with open(self.trackfilepath, "w"):
                pass

        with open(self.trackfilepath, "r") as stream:
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
        return write_occupied


class WriteFileLock:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.trackfilepath = self.filepath.parent / (str(self.filepath.stem) + "_track")
        self.lock = FileLock(self.filepath)
        random.seed(time.time())
        self.id = "".join([random.choice(string.ascii_letters + string.digits) for n in range(32)])

    def acquire(self, raise_exception=True):
        """
        This function tried to acquire a WriteFileLock.
        The process is the following:
        1. acquire filelock
        2. read the lock occupation situation
        3. check if the lock is already occupied by another write process
        4.1. release filelock and raise exception(or return 0) if the lock is already occupied by another write process
        4.2. otherwise write the updated lock occupation situation back, release filelock and return

            Parameters:
                raise_exception (bool): whether raises an exception when failed (default: True)

            Returns:
                success (bool): whether succeeded or not.
                    True means succeeded, False means failed (if the param raiseException is set to false).
                    A filelock.Timeout exception will be raised if failed
        """

        # 1. acquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.trackfilepath):
            # create a file to tack the lock occupation situation
            with open(self.trackfilepath, "w") as track_file:
                pass

        with open(self.trackfilepath, "r") as stream:
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
            if raise_exception:
                raise filelock.Timeout(self.lock)  # mimic filelock TimeOut Error
        else:
            time_info = datetime.datetime.now().isoformat()
            lock_occupy_info[self.id] = {"time": f"{time_info}", "type": "read"}
            with open(self.trackfilepath, "w") as track_file:
                yaml.dump(lock_occupy_info, track_file, default_flow_style=False)
            lock_acquire_success = True

        self.lock.release()
        atexit.register(self.release)
        return lock_acquire_success

    def release(self):
        """
        This function releases a WriteFileLock.
        the process is the following:
        1. acquire filelock
        2. read the lock occupation situation
        3. delete the record of self.id (no exception will be raised if self.id is not found in the record)
        4. write the updated lock occupation situation back, release filelock and return
        """

        # 1. acquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.trackfilepath):
            # raise error if file does not exist
            raise RuntimeError("track file for read write log is missing.")

        with open(self.trackfilepath, "r") as stream:
            # read the current lock occupation situation
            try:
                lock_occupy_info = yaml.safe_load(stream) or {}
                if lock_occupy_info is None:
                    lock_occupy_info = {}
            except yaml.YAMLError as exc:
                print(exc)

        # 3. delete the record of self.id
        if self.id in lock_occupy_info.keys():
            lock_occupy_info.pop(self.id)

        # 4. write the updated lock occupation situation back, release filelock and return
        with open(self.trackfilepath, "w") as track_file:
            if lock_occupy_info:
                yaml.dump(lock_occupy_info, track_file, default_flow_style=False)

        self.lock.release()

    @property
    def is_locked(self):
        """
        This property returns if a lock is occupied(locked) by other processes.
        the process is the following:
        1. acquire filelock
        2. read the lock occupation situation
        3. check if the lock is already occupied(locked) by another write process

            Returns:
                is_locked (bool): whether the lock is already occupied(locked) by another write process
        """

        # 1. acquire filelock
        self.lock.acquire(timeout=2)

        # 2. read the lock occupation situation
        if not os.path.exists(self.trackfilepath):
            # create a file to tack the lock occupation situation
            with open(self.trackfilepath, "w"):
                pass

        with open(self.trackfilepath, "r") as stream:
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

    filepath = "/tmp/read_write_file_lock/hello"
    if not os.path.exists(Path(filepath).parent):
        os.mkdir(Path(filepath).parent)
    readlock1 = ReadFileLock(filepath)
    print("read1: " + readlock1.id)
    writelock1 = WriteFileLock(filepath)
    print("write1: " + writelock1.id)
    assert readlock1.acquire(raise_exception=False)
    assert not writelock1.acquire(raise_exception=False)
    time.sleep(20)
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
