def get_data(relative_path):
    """
    Get the file path to some data in the qcsubmit package.

    Parameters:
        relative_path: The relative path to the data
    """

    from pkg_resources import resource_filename
    import os

    fn = resource_filename('qcsubmit', os.path.join(
        'data', relative_path))

    if not os.path.exists(fn):
        raise ValueError(f"Sorry! {fn} does not exist. If you just added it, you'll have to re-install")

    return fn
