import os.path


class BaseParser:
    """
    Generic routines for all parsers.

    ...

    Attributes
    ----------
    file_paths : dict
        the paths to the parsed files
    file_contents : dict
        the contents of parsed files
    """

    def __init__(self, supported_files, **kwargs):
        self.file_paths = {}
        self.file_contents = {}

        for kwarg in kwargs:
            # Check if the file type is supported
            if kwarg not in supported_files:
                raise ValueError(f"{kwarg} is not a supported file.")

            # Check if the file path exists
            if not os.path.isfile(kwargs[kwarg]):
                raise ValueError(f"{kwargs[kwarg]} does not exist.")

            # Store the file paths
            self.file_paths[kwarg] = kwargs[kwarg]

            # Get the contents of the files
            if kwargs[kwarg].endswith(".csc"):
                with open(kwargs[kwarg], "rb") as f:
                    self.file_contents[kwarg] = f.read()

            else:
                with open(kwargs[kwarg], "r") as f:
                    self.file_contents[kwarg] = f.readlines()

    def __str__(self) -> str:
        if self.lines is None or self.lines == "":
            raise ValueError("Could not find file contents.")
        else:
            return "".join(self.lines)

    @property
    def lines(self) -> List[str]:
        return self._lines

    @lines.setter
    def lines(self, value: List[str]):
        self._lines = value

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, value: str):
        self._path = value
