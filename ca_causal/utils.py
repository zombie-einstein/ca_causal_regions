import typing

import numpy as np


def number_to_base(n: int, *, base: int, width: int) -> np.array:
    """
    Convert a number into it's representation in argument weight and
    fixed width
    Args:
        n (int): Number to convert
        base (int): Base to represent number in
        width (int): Width of presentation (padding with 0s)
    Returns:
        np.array: Array of digits
    """
    if n > (base**width) - 1:
        raise ValueError(
            (
                f"{n} is outside the allotted width {width}"
                " of the representation in base {base}"
            )
        )
    ret = np.zeros(width).astype("int")
    idx = 0
    while n:
        ret[idx] = int(n % base)
        n //= base
        idx += 1
    return ret


def base_to_number(n: typing.Union[typing.List, np.array], *, base: int):
    """Convert number in base array back to an integer value"""
    return np.sum(n * (base ** np.arange(len(n))))


def left_right_shift(n, base=2):
    """
    Get an integer converted into its base representation and then shifted
    left and right with permutations of the states appended as prefix/suffix

    Args:
        n (int): Number to shift left/right
        base (int): Base to use as representation

    Returns:
        np.array: 2d array containing the shifted values
    """
    b = number_to_base(n, base=base, width=3)
    bases = base ** np.arange(3)
    return np.array(
        [
            np.sum(b[:-1] * bases[1:]) + np.arange(base),
            np.sum(b[1:] * bases[:-1]) + np.arange(base) * (base**2),
        ]
    )


def adjacency_rules(base=2):
    """
    Collects the left and right shifted value for all integers in the
    range appropriate

    Args:
        base (int): Base to use as representation

    Returns:
        dict: Dictionary with shifted values indexed by central value
    """
    return {i: left_right_shift(i, base=base) for i in range(base**3)}


def causal_dependency(rule_num, base=2):
    """
    Derives how a triple state depends on its neighbours given an update rule

    Args:
        rule_num (int): Update rule number
        base (int): Number of possible states

    Returns:
        np.array: Array representing the dependencies. On the left the
            dependency is indicated by the 0th index taking the value {0,-1}
            and conversely on the right in the 1st index taking the value {0,1}
    """
    rule = number_to_base(rule_num, base=2, width=base**3)
    adjacency = adjacency_rules(base=base)

    ret = np.zeros((base**3, 2)).astype(np.int8)

    for k, v in adjacency.items():
        p = [(x, y) for x in v[0] for y in v[1]]
        triple_mappings = {
            i: rule[i[0]] + (rule[k] * base) + rule[i[1]] * (base**2) for i in p
        }
        ll = triple_mappings[(v[0][0], v[1][0])] != triple_mappings[(v[0][1], v[1][0])]
        rr = triple_mappings[(v[0][0], v[1][0])] != triple_mappings[(v[0][0], v[1][1])]
        ret[k] = [ll, rr]

    return ret
