from typing import List

import dfttools.utils.file_utils as fu
from dfttools.base_parser import BaseParser


class Parameters(BaseParser):
    """Handle files that control parameters for electronic structure calculations.

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


class AimsControl(Parameters):
    """FHI-aims control file parser.

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

    def __init__(self, control_in="control.in") -> None:
        super().__init__(control_in=control_in)
        self.lines = self._file_contents["control_in"]
        self.path = self._file_paths["control_in"]
        # Check if the control.in file was provided
        fu.check_required_files(self._supported_files, "control_in")

    @property
    def lines(self) -> list:
        return self._lines

    @lines.setter
    def lines(self, lines: List[str]) -> None:
        self._lines = lines

    @property
    def path(self) -> str:
        return self._file_path

    @path.setter
    def path(self, file_path: str) -> None:
        self._file_path = file_path

    def add_control_keywords(self, **kwargs: dict) -> None:
        """Add keywords to the control.in file.

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
        """Remove keywords from the control.in file.

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
