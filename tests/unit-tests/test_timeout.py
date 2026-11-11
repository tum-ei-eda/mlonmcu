import time

import pytest

from mlonmcu.timeout import exec_timeout


def add(left, right=0):
    return left + right


def fail():
    raise ValueError("failure from child")


def wait_forever():
    time.sleep(5)


def test_exec_timeout_returns_child_result():
    assert exec_timeout(1, add, 2, right=3) == 5


def test_exec_timeout_reraises_child_exception():
    with pytest.raises(ValueError, match="failure from child"):
        exec_timeout(1, fail)


def test_exec_timeout_terminates_slow_child():
    with pytest.raises(TimeoutError, match="did not complete"):
        exec_timeout(0.01, wait_forever)
