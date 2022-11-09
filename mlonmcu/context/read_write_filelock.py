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
"""
This file contains read lock and write lock classes based on filelock
The locks are non-blocking.
"""

from filelock import FileLock
import uuid  # this is used to create a identifier for every ReadFileLock and WriteFileLock instance.
import datetime  # this is used to track the lock actions
import os
import yaml
from pathlib import Path
import atexit


class RWLockTimeout(TimeoutError):
    """Raised when the lock could not be acquired."""

    def __init__(self, lock) -> None:
        #: The Read or Write lock instance.
        self.lock = lock

    def __str__(self) -> str:
        return f"The lock with id '{self.lock.id}' in env '{self.lock.filepath.parent}' could not be acquired."


class ReadFileLock:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.trackfilepath = self.filepath.parent / (str(self.filepath.stem) + "_track")
        self.lock = FileLock(self.filepath)
        self.id = str(uuid.uuid4())

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
                    True means succeeded, False means failed (if the param raiseException is set to False).
                    A RWLockTimeout exception will be raised if failed (if the param raiseException is set to True).
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
                raise RWLockTimeout(self)
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
        self.id = str(uuid.uuid4())

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
                    True means succeeded, False means failed (if the param raiseException is set to False).
                    A RWLockTimeout exception will be raised if failed (if the param raiseException is set to True).
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
                raise RWLockTimeout(self)
        else:
            time_info = datetime.datetime.now().replace(microsecond=0).isoformat()
            lock_occupy_info[self.id] = {"time": f"{time_info}", "type": "write"}
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
