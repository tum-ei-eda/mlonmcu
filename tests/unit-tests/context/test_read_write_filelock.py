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
from pathlib import Path
import pytest
from mlonmcu.context.read_write_filelock import ReadFileLock, WriteFileLock, RWLockTimeout


@pytest.mark.parametrize("test_input_raise_exception", [False, True])
def test_read_after_read(monkeypatch, fake_environment_directory: Path, test_input_raise_exception):
    monkeypatch.chdir(fake_environment_directory)
    filepath = fake_environment_directory / ".lock"

    readlock1 = ReadFileLock(filepath)
    print("read1: " + readlock1.id)
    readlock2 = ReadFileLock(filepath)
    print("read2: " + readlock2.id)

    assert readlock1.acquire(raise_exception=test_input_raise_exception)
    assert readlock2.acquire(raise_exception=test_input_raise_exception)

    readlock1.release()
    readlock2.release()


def test_read_after_write_exception_disabled(monkeypatch, fake_environment_directory: Path):
    monkeypatch.chdir(fake_environment_directory)
    filepath = fake_environment_directory / ".lock"

    writelock1 = WriteFileLock(filepath)
    print("write1: " + writelock1.id)
    readlock1 = ReadFileLock(filepath)
    print("read1: " + readlock1.id)
    assert writelock1.acquire(raise_exception=False)
    assert not readlock1.acquire(raise_exception=False)

    writelock1.release()


def test_read_after_write_exception_enabled(monkeypatch, fake_environment_directory: Path):
    monkeypatch.chdir(fake_environment_directory)
    filepath = fake_environment_directory / ".lock"

    writelock1 = WriteFileLock(filepath)
    print("write1: " + writelock1.id)
    readlock1 = ReadFileLock(filepath)
    print("read1: " + readlock1.id)

    assert writelock1.acquire()
    with pytest.raises(RWLockTimeout, match=r".*The lock.*could\ not\ be\ acquired\..*"):
        readlock1.acquire()

    writelock1.release()


def test_write_after_read_exception_disabled(monkeypatch, fake_environment_directory: Path):
    monkeypatch.chdir(fake_environment_directory)
    filepath = fake_environment_directory / ".lock"

    readlock1 = ReadFileLock(filepath)
    print("read1: " + readlock1.id)
    writelock1 = WriteFileLock(filepath)
    print("write1: " + writelock1.id)
    assert readlock1.acquire(raise_exception=False)
    assert not writelock1.acquire(raise_exception=False)

    readlock1.release()


def test_write_after_read_exception_enabled(monkeypatch, fake_environment_directory: Path):
    monkeypatch.chdir(fake_environment_directory)
    filepath = fake_environment_directory / ".lock"

    readlock1 = ReadFileLock(filepath)
    print("read1: " + readlock1.id)
    writelock1 = WriteFileLock(filepath)
    print("write1: " + writelock1.id)

    assert readlock1.acquire()
    with pytest.raises(RWLockTimeout, match=r".*The lock.*could\ not\ be\ acquired\..*"):
        writelock1.acquire()

    readlock1.release()


def test_write_after_write_exception_disabled(monkeypatch, fake_environment_directory: Path):
    monkeypatch.chdir(fake_environment_directory)
    filepath = fake_environment_directory / ".lock"

    writelock1 = WriteFileLock(filepath)
    print("write1: " + writelock1.id)
    writelock2 = WriteFileLock(filepath)
    print("write1: " + writelock1.id)
    assert writelock1.acquire(raise_exception=False)
    assert not writelock2.acquire(raise_exception=False)

    writelock1.release()


def test_write_after_write_exception_enabled(monkeypatch, fake_environment_directory: Path):
    monkeypatch.chdir(fake_environment_directory)
    filepath = fake_environment_directory / ".lock"

    writelock1 = WriteFileLock(filepath)
    print("write1: " + writelock1.id)
    writelock2 = WriteFileLock(filepath)
    print("write1: " + writelock2.id)

    assert writelock1.acquire()
    with pytest.raises(RWLockTimeout, match=r".*The lock.*could\ not\ be\ acquired\..*"):
        writelock2.acquire()

    writelock1.release()
