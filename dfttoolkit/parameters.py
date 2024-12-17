from typing import List

import dfttoolkit.utils.file_utils as fu
from dfttoolkit.base_parser import BaseParser


class Parameters(BaseParser):
    """
    Handle files that control parameters for electronic structure calculations.

    If contributing a new parser, please subclass this class, add the new supported file
    type to _supported_files, call the super().__init__ method, include the new file
    type as a kwarg in the super().__init__ call. Optionally include the self.lines line
    in examples.

    ...

    Attributes
    ----------
    _supported_files : list
        List of supported file types.
    """

    def __init__(self, **kwargs: str):
        # FHI-aims, ...
        self._supported_files = ["control_in"]

        # Check that only supported files were provided
        for val in kwargs.keys():
            fu.check_required_files(self._supported_files, val)

        super().__init__(self._supported_files, **kwargs)

    @property
    def supported_files(self) -> List[str]:
        return self._supported_files


class AimsControl(Parameters):
    """
    FHI-aims control file parser.

    ...

    Attributes
    ----------
    lines : List[str]
        The contents of the control.in file.
    path : str
        The path to the control.in file.
    """

    def __init__(
        self, control_in: str = "control.in", parse_file: bool = True
    ):
        if parse_file:
            super().__init__(control_in=control_in)
            self.lines = self.file_contents["control_in"]
            self.path = self.file_paths["control_in"]

            # Check if the control.in file was provided
            fu.check_required_files(self._supported_files, "control_in")

    def add_keywords(self, **kwargs: dict) -> None:
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

    def remove_keywords(self, *args: str) -> None:
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

    def get_keywords(self) -> dict:
        """
        Get the keywords from the control.in file.

        Returns
        -------
        dict
            A dictionary of the keywords in the control.in file.
        """

        keywords = {}

        for line in self.lines:
            spl = line.split()

            if "#" * 80 in line:
                break

            if len(spl) > 0 and spl[0] != "#":
                if len(spl) == 1:
                    keywords[spl[0]] = None
                else:
                    keywords[spl[0]] = " ".join(spl[1:])

        return keywords
