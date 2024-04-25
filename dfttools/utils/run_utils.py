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
