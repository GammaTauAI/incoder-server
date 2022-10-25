from typing import List


def input_to_infill_format(s: str, placeholder: str = '_hole_') -> List[str]:
    return s.split(placeholder)
