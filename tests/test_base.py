from dfttoolkit.base_parser import BaseParser
import pytest


@pytest.fixture
def elsi_csc(cwd):
    yield f"{cwd}/fixtures/elsi_files/D_spin_01_kpt_000001.csc"


@pytest.fixture(autouse=True)
def base_parser(default_calc_dir, elsi_csc):
    bp = BaseParser(
        ["aims_out", "control_in", "elsi_out"],
        aims_out=f"{default_calc_dir}/aims.out",
        control_in=f"{default_calc_dir}/control.in",
        elsi_out=elsi_csc,
    )
    yield bp


def test_check_supported_file(default_calc_dir):

    with pytest.raises(ValueError):
        BaseParser(
            ["aims_out", "elsi_in"],
            aims_out=f"{default_calc_dir}/aims.out",
            control_in=f"{default_calc_dir}/control.in",
        )

    with pytest.raises(ValueError):
        BaseParser(
            ["aims.out", "control.in"],
            aims_out=f"{default_calc_dir}/aims.out",
            control_in=f"{default_calc_dir}/control.in",
        )


def test_check_file_exists(default_calc_dir):
    with pytest.raises(FileNotFoundError):
        BaseParser(
            ["aims_out", "control_in"],
            aims_out=f"{default_calc_dir}/aims.out",
            control_in="./control.in",
        )


def test_store_file_paths(base_parser, default_calc_dir, cwd):
    assert base_parser.file_paths == {
        "aims_out": f"{default_calc_dir}/aims.out",
        "control_in": f"{default_calc_dir}/control.in",
        "elsi_out": f"{cwd}/fixtures/elsi_files/D_spin_01_kpt_000001.csc",
    }


def test_read_binary_content(base_parser):
    assert isinstance(base_parser.file_contents["elsi_out"], bytes) is True


def test_read_txt_content(elsi_csc, base_parser, default_calc_dir):
    with open(elsi_csc, "rb") as f:
        assert base_parser.file_contents["elsi_out"] == f.read()

    with open(f"{default_calc_dir}/aims.out", "r") as f:
        assert base_parser.file_contents["aims_out"] == f.readlines()

    with open(f"{default_calc_dir}/control.in", "r") as f:
        assert base_parser.file_contents["control_in"] == f.readlines()


def test__str__(base_parser, default_calc_dir):
    base_parser.lines = base_parser.file_contents["aims_out"]
    with open(f"{default_calc_dir}/aims.out", "r") as f:
        ao_lines = f.readlines()

    assert str(base_parser) == "".join(ao_lines)

    base_parser.lines = base_parser.file_contents["control_in"]
    with open(f"{default_calc_dir}/control.in", "r") as f:
        ci_lines = f.readlines()

    assert str(base_parser) == "".join(ci_lines)


def test__str__error(base_parser):
    # TODO maybe change to read the file as None or "" rather than just setting the attr
    base_parser.lines = None
    with pytest.raises(ValueError):
        str(base_parser)

    base_parser.lines = ""
    with pytest.raises(ValueError):
        str(base_parser)


def test_properties():
    # TODO
    pass
