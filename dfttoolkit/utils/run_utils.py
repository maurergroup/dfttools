import os.path
from functools import wraps


def no_repeat(
    original_func=None,
    *,
    output_file: str = "aims.out",
    calc_dir: str = "./",
    force: bool = False,
):
    """
    Don't repeat the calculation if aims.out exists in the calculation directory.

    Parameters
    ----------
    output_file : str, default='aims.out'
        The name of the output file to check for.
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
            # Override calc_dir in decorator call if given in func
            if "calc_dir" in kwargs:
                check_dir = kwargs["calc_dir"]
            else:
                check_dir = calc_dir

            if not os.path.isdir(check_dir):
                raise ValueError(f"{check_dir} is not a directory.")
            if force:
                return func(*args, **kwargs)
            if not os.path.isfile(f"{check_dir}/{output_file}"):
                return func(*args, **kwargs)
            else:
                print(f"aims.out already exists in {check_dir}. Skipping calculation.")

        return wrapper

    if original_func:
        return _no_repeat(original_func)

    return _no_repeat
