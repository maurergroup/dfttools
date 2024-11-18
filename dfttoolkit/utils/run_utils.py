import os.path
from functools import wraps


def no_repeat(calc_dir="./", force=False):
    """
    Don't repeat the calculation if aims.out exists in the calculation directory.

    Parameters
    ----------
    calc_dir : str, default="./"
        The directory where the calculation is performed
    force : bool, default=False
        If True, the calculation is performed even if aims.out exists in the calculation
        directory.

    Raises
    -------
    ValueError
        if the `calc_dir` kwarg is not a directory
    """

    def _no_repeat(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.isdir(calc_dir):
                raise ValueError(f"{calc_dir} is not a directory.")
            if force:
                return func(*args, **kwargs)
            if not os.path.isfile(f"{calc_dir}/aims.out"):
                return func(*args, **kwargs)
            else:
                print(f"aims.out already exists in {calc_dir}. Skipping calculation.")

        return wrapper

    return _no_repeat
