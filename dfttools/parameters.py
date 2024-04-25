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
    _supported_files : list
        List of supported file types.
    file_paths : dict
        The paths to the files to be parsed.
    file_contents : dict
        The contents of the files to be parsed.
    """

    def __init__(self, control_in="control.in") -> None:
        super().__init__(control_in=control_in)
        # Check if the control.in file was provided
        fu.check_required_files(self._supported_files, "control_in")

    def add_control_keywords(self, **kwargs: dict) -> None:
        """Add keywords to the control.in file.

        Parameters
        ----------
        **kwargs : dict
            Keywords to be added to the control.in file.
        """

        for keyword in kwargs:
            self.file_contents["control_in"].append(keyword + "\n")

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
            for i, line in enumerate(self.file_contents["control_in"]):
                if keyword in line:
                    self.file_contents["control_in"].pop(i)

        with open(self.file_paths["control_in"], "w") as f:
            f.writelines(self.file_contents["control_in"])
