import os
import pytest
from dfttoolkit.parameters import AimsControl


# TODO Create generic test class with functions for all tests to inherit


class TestAimsControl:

    @pytest.fixture(params=range(1, 11), autouse=True)
    def aims_control(self, request, aims_calc_dir):

        cwd = os.path.dirname(os.path.realpath(__file__))

        self.ac = AimsControl(
            control_in=f"{cwd}/fixtures/{aims_calc_dir}/{str(request.param)}/control.in"
        )
