from typing import List


def get_data(relative_path):
    """
    Get the file path to some data in the qcsubmit package.

    Parameters:
        relative_path: The relative path to the data
    """

    import os

    from pkg_resources import resource_filename

    fn = resource_filename("qcsubmit", os.path.join("data", relative_path))

    if not os.path.exists(fn):
        raise ValueError(
            f"Sorry! {fn} does not exist. If you just added it, you'll have to re-install"
        )

    return fn


def clean_strings(string_list: List[str]) -> List[str]:
    """
    Clean up a list of strings ready to be cast to numbers.
    """
    clean_string = []
    for string in string_list:
        new_string = string.strip()
        clean_string.append(new_string.strip(","))
    return clean_string
