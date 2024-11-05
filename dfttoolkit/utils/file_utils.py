from pathlib import Path
from typing import Union

from click import edit


def aims_bin_path_prompt(change_bin: Union[bool, str], save_dir) -> str:
    """
    Prompt the user to enter the path to the FHI-aims binary, if not already found in
    .aims_bin_loc.txt

    If it is found in .aims_bin_loc.txt, the path will be read from there, unless
    change_bin is True, in which case the user will be prompted to enter the path again.

    Parameters
    ----------
    change_bin : Union[bool, str]
        whether the user wants to change the binary path. If str == "change_bin", the
        user will be prompted to enter the path to the binary again.
    save_dir : str
        the directory to save or look for the .aims_bin_loc.txt file

    Returns
    -------
    binary : str
        path to the location of the FHI-aims binary
    """

    marker = (
        "\n# Enter the path to the FHI-aims binary above this line\n"
        "# Ensure that the full absolute path is provided"
    )

    def write_bin():
        binary = edit(marker)
        binary = str(binary).split()[0]
        if binary is not None:
            if Path(binary).is_file():
                with open(f"{save_dir}/.aims_bin_loc.txt", "w+") as f:
                    f.write(binary)

            else:
                raise FileNotFoundError(
                    "the path to the FHI-aims binary does not exist"
                )

        else:
            raise FileNotFoundError(
                "the path to the FHI-aims binary could not be found"
            )

        return binary

    if (
        not Path(f"{save_dir}/.aims_bin_loc.txt").is_file()
        or change_bin == "change_bin"
    ):
        binary = write_bin()

    else:
        # Parse the binary path from .aims_bin_loc.txt
        with open(f"{save_dir}/.aims_bin_loc.txt", "r") as f:
            binary = f.readlines()[0]

        # Check if the binary path exists and is a file
        if not Path(binary).is_file():
            binary = write_bin()

    return binary


def check_required_files(files: list, *args: str, any=False) -> None:
    """
    Raise an error if a necessary file was not given.

    Parameters
    ----------
    files : list
        supported files to reference provided files against
    *args : str
        the files that are required to be provided
    any : bool
        whether at least one of the files is required or all of them

    Raises
    -------
    ValueError
        if a necessary file was not given
    """

    if any:
        for arg in args:
            if arg in files:
                return

        raise ValueError(f"At least one of the following files is required:\n{args}")

    else:
        for arg in args:
            if arg not in files:
                raise ValueError(f"{arg} was not provided in the constructor.")
