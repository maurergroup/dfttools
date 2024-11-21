import os
import shutil

import pytest
import yaml
from dfttoolkit.parameters import AimsControl


class TestAimsControl:
    @property
    def aims_fixture_no(self) -> int:
        return int(self.ac.path.split("/")[-2])

    @pytest.fixture(params=range(1, 11), autouse=True)
    def aims_control(self, cwd, request, aims_calc_dir):
        self.ac = AimsControl(
            control_in=f"{cwd}/fixtures/{aims_calc_dir}/{str(request.param)}"
            "/control.in"
        )

    @pytest.fixture
    def ref_files(self, cwd):
        with open(
            f"{cwd}/fixtures/manipulated_aims_files/{self.aims_fixture_no}/control.in",
            "r",
        ) as f:
            yield f.readlines()

    def test_initialise_parameters(self, cwd):
        with open(
            f"{cwd}/fixtures/default_aims_calcs/{self.aims_fixture_no}/control.in",
            "r",
        ) as f:
            assert self.ac.lines == f.readlines()

        assert self.ac.lines == self.ac.file_contents["control_in"]
        assert self.ac.supported_files == ["control_in"]
        assert self.ac.supported_files == self.ac._supported_files
        assert self.ac.path.endswith("control.in")

    def test_remove_keywords_overwrite(self, tmp_dir, ref_files):
        control_path = tmp_dir / "control.in"
        shutil.copy(self.ac.path, control_path)
        ac = AimsControl(control_in=str(control_path))
        ac.remove_keywords("xc", "relax_geometry", "k_grid", output="overwrite")

        assert "".join(ref_files) == control_path.read_text()

    def test_remove_keywords_print(self, capsys, ref_files):
        self.ac.remove_keywords("xc", "relax_geometry", "k_grid", output="print")
        out, err = capsys.readouterr()

        assert out == "".join(ref_files) + "\n"
        assert err == ""

    def test_remove_keywords_return(self, ref_files):
        control = self.ac.remove_keywords("xc", "relax_geometry", "k_grid")
        assert control == ref_files

    def test_get_keywords(self, ref_data):
        keywords = self.ac.get_keywords()
        assert keywords == ref_data["keywords"][self.aims_fixture_no - 1]
