

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



            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
