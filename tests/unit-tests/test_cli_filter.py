from mlonmcu.cli.helper.filter import filter_arg

def test_cli_filter():
    assert filter_arg("x,y") == ["x", "y"]
    assert filter_arg("") == []
    assert filter_arg(None) == []
