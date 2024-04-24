import os.path
import warnings
from functools import wraps


def no_repeat(func):
    """
    Don't repeat the calculation if aims.out exists in the calculation directory.

    A kwarg must be given to the decorated function called `calc_dir` which is the
    directory where the calculation is to be performed.

    Raises
    -------
    ValueError
        if the `calc_dir` kwarg is not a directory
    """

    @wraps(func)
    def wrapper_no_repeat(*args, **kwargs):
        if "calc_dir" in kwargs and "force" in kwargs:
            if not os.path.isdir(kwargs["calc_dir"]):
                raise ValueError(f"{kwargs.get('calc_dir')} is not a directory.")

            if kwargs["force"]:
                return func(*args, **kwargs)
            if not os.path.isfile(f"{kwargs.get('calc_dir')}/aims.out"):
                return func(*args, **kwargs)
            else:
                print(
                    f"aims.out already exists in {kwargs.get('calc_dir')}. Skipping "
                    "calculation."
                )

        else:
            warnings.warn(
                "'calc_dir' and/or 'force' kwarg not provided: ignoring decorator"
            )

    return wrapper_no_repeat


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
