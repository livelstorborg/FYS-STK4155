import importlib.util


def test_package_is_installable():
    assert importlib.util.find_spec("project1") is not None
