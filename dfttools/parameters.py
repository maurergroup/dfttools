from typing import List

import dfttools.utils.file_utils as fu
from dfttools.base_parser import BaseParser


class Parameters(BaseParser):
    """
    Handle files that control parameters for electronic structure calculations.

    Parameters
    ----------
    _supported_files : list
        List of supported file types.
    file_paths : dict
        The paths to the files to be parsed.
    file_contents : dict
        The contents of the files to be parsed.
    """

    # FHI-aims, ...
    _supported_files = ["control_in"]

    def __init__(self, **kwargs):
        super().__init__(self._supported_files, **kwargs)

        for val in kwargs.keys():
            fu.check_required_files(self._supported_files, val)

    @property
    def supported_files(self):
        return self._supported_files

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, lines: List[str]):
        self._lines = lines

    @property
    def path(self):
        return self._file_path

    @path.setter
    def path(self, file_path: str):
        self._file_path = file_path


class AimsControl(Parameters):
    """
    FHI-aims control file parser.

    Attributes
    ----------
    lines
    _supported_files : list
        List of supported file types.
    file_paths : dict
        The paths to the files to be parsed.
    file_contents : dict
        The contents of the files to be parsed.
    """

    def __init__(self, control_in: str = "control.in") -> None:
        super().__init__(control_in=control_in)
        self.lines = self.file_contents["control_in"]
        self.path = self.file_paths["control_in"]

    def add_control_keywords(self, **kwargs: dict) -> None:
        """
        Add keywords to the control.in file.

        Parameters
        ----------
        **kwargs : dict
            Keywords to be added to the control.in file.
        """

        for keyword in kwargs:
            self.lines.append(keyword + "\n")

        # TODO finish this
        raise NotImplementedError

    def remove_control_keywords(self, *args: str) -> None:
        """
        Remove keywords from the control.in file.

        Parameters
        ----------
        *args : str
            Keywords to be removed from the control.in file.
        """

        for keyword in args:
            for i, line in enumerate(self.lines):
                if keyword in line:
                    self.lines.pop(i)

        with open(self.path, "w") as f:
            f.writelines(self.lines)
