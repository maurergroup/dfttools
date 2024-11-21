import os
import shutil
import pytest
from dfttoolkit.parameters import AimsControl


class TestAimsControl:

    @pytest.fixture(params=range(1, 11), autouse=True)
    def aims_control(self, request, aims_calc_dir):
        self.cwd = os.path.dirname(os.path.realpath(__file__))

        self.ac = AimsControl(
            control_in=f"{self.cwd}/fixtures/{aims_calc_dir}/{str(request.param)}"
            "/control.in"
        )

        # Read reference
        with open(
            f"{self.cwd}/fixtures/manipulated_aims_files/{str(request.param)}"
            "/control.in",
            "r",
        ) as f:
            self.removed_keywords_control_ref = f.readlines()

    @property
    def _aims_fixture_no(self) -> int:
        return int(self.ac.path.split("/")[-2])

    def test_remove_keywords_overwrite(self, tmp_dir):
        control_path = tmp_dir / "control.in"
        shutil.copy(self.ac.path, control_path)
        ac = AimsControl(control_in=str(control_path))
        ac.remove_keywords("xc", "relax_geometry", "k_grid", output="overwrite")

        assert "".join(self.removed_keywords_control_ref) == control_path.read_text()

    def test_remove_keywords_print(self, capfd):
        self.ac.remove_keywords("xc", "relax_geometry", "k_grid", output="print")

        out, err = capfd.readouterr()

        # print(out)
        # print("".join(self.removed_keywords_control_ref))

        assert out == print("".join(self.removed_keywords_control_ref))
        assert err == ""
