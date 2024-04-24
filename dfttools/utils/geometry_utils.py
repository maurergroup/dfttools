from dfttools.geometry import XYZGeometry

def read_xyz_animation(filename) -> list:
    """
    Reads an XYZ animation file, i.e. an XYZ file containing serveral
    geometries.

    Parameters
    ----------
    filename : str
        path to xyz animation file

    Returns
    -------
    list
        list of XYZGeometries

    """
    text_list = []
    geometry_text = ''
    
    with open(filename) as f:
        text = f.readlines()
        
        for ind, line in enumerate(text):
            line_split = line.split()
            
            n_entries = len(line_split)
            
            if not ind == 0 and n_entries == 1:
                text_list.append(geometry_text)
                geometry_text = ''
            
            geometry_text += line
    
    geometry_list = []
    for text in text_list:
        geometry = XYZGeometry()
        geometry.parse(text)
        geometry_list.append(geometry)
    
    return geometry_list

