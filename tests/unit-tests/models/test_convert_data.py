import pytest

from mlonmcu.models.convert_data import convert


def test_convert_float():
    # valid
    assert convert("float", "1.0,2.0,3.0,4.0") == b"\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40\x00\x00\x80\x40"

    # invalid
    with pytest.raises(ValueError):
        convert("float", "\x12\x34\x56\x78")


def test_convert_hexstr():
    # valid
    assert convert("hexstr", "\\x12\\x34\\x56\\x78") == b"\x12\x34\x56\x78"


def test_convert_int8():
    # valid
    assert convert("int8", "1,2,3,4") == b"\x01\x02\x03\x04"

    # invalid
    with pytest.raises(ValueError):
        convert("int8", "\x12\x34\x56\x78")
    with pytest.raises(ValueError):
        convert("int8", "1.0,2.0,3.0,4.0")


def test_convert_invalid_mode():
    with pytest.raises(AssertionError):
        convert("invalid", "\x12\x34\x56\x78")
