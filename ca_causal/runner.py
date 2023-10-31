import numpy as np

from ca_causal import utils


def run_ca(
    rule_num: int, steps: int, width: int, base=2, random=True, seed=0, decay=0.85
):
    """
    Run the model for a given rule, generating the basic rule output and
    the corresponding causal region representation

    Args:
        rule_num (int): Rule number
        steps (int): NUmber of update steps
        width (int): Width of state array
        base (int): Number of possible states
        random (bool): If `True` the initial state will be randomly generated
            if `False` then start from a single live cell
        seed (int): Random state generation seed
        decay (float): Decay applied to the summation at each step of the
            causal region calculation

    Returns:
        tuple(np.array, np.array): Basic CA phase space, and causal region
            representation
    """
    rule = utils.number_to_base(rule_num, base=base, width=base**3)
    causal_map = utils.causal_dependency(rule_num, base=base) * np.array([-1, 1])

    actual = np.zeros((steps + 1, width)).astype(np.int8)
    triples = np.zeros((steps, width)).astype(np.int8)
    causal = np.zeros((steps, width, 2)).astype(np.float64)

    left_shift = np.arange(-1, width - 1)
    right_shift = np.arange(1, width + 1)

    if random:
        actual[0] = np.random.RandomState(seed).randint(0, base, width)
    else:
        actual[0][int(width / 2)] = 1

    for i in range(1, steps + 1):
        centre = actual[i - 1]
        right = centre.take(right_shift, mode="wrap")
        left = centre.take(left_shift, mode="wrap")
        idx = left + centre * base + right * (base**2)

        actual[i] = rule[idx]
        triples[i - 1] = idx
        causal[i - 1] = causal_map[idx]

    actual = actual[:-1]

    for i in range(1, steps):
        l_idx = np.arange(width) + causal[i, :, 0].astype(np.int8)
        r_idx = np.arange(width) + causal[i, :, 1].astype(np.int8)
        causal[i, :, 0] = causal[i, :, 0] + decay * causal[i - 1, :, 0].take(
            l_idx, mode="wrap"
        )
        causal[i, :, 1] = causal[i, :, 1] + decay * causal[i - 1, :, 1].take(
            r_idx, mode="wrap"
        )

    return actual, causal
