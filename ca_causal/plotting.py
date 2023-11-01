import matplotlib.pyplot as plt
import numpy as np

from ca_causal.runner import run_ca


def comparison_plot(
    rule: int,
    steps: int,
    width: int,
    base=2,
    random=True,
    seed=0,
    decay=0.9,
    colormap: str = "coolwarm",
    overlay: bool = False,
):
    """
    Convenience function for plotting a comparison plot of the regular
    ca rule, and dervied causal region array

    Args:
        rule (int): Rule number
        steps (int): NUmber of update steps
        width (int): Width of state array
        base (int): Number of possible states
        random (bool): If `True` the initial state will be randomly generated
            if `False` then start from a single live cell
        seed (int): Random state generation seed
        decay (float): Decay applied to the summation at each step of the
            causal region calculation
        colormap (str): Colormap to use for causal region plot
        overlay (bool): If `True` the binary ca and causal regions will be
            overlaid
    """

    # Run the model and get the phase space arrays
    actual, causal = run_ca(
        rule, steps, width, base=base, random=random, seed=seed, decay=decay
    )

    # Converts left distance to magnitude
    causal[:, :, 0] = -causal[:, :, 0]

    # These lines compress the 3rd dimension of the causal array
    num = causal[:, :, 0] - causal[:, :, 1]
    den = causal[:, :, 0] + causal[:, :, 1]
    arr = np.divide(num, den, out=np.zeros_like(den), where=den != 0)

    if overlay:
        f, ax = plt.subplots(figsize=(9, 10 * (steps / width)))
        ax.matshow(actual[10:], cmap=plt.get_cmap("binary"))
        ax.matshow(arr[10:], cmap=plt.get_cmap(colormap), alpha=0.75)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # Plot the arrays side-by-side
        f, ax = plt.subplots(1, 2, figsize=(18, 10 * (steps / width)))

        ax[0].matshow(actual[10:], cmap=plt.get_cmap("binary"))
        ax[1].matshow(arr[10:], cmap=plt.get_cmap(colormap))

        for i in ax:
            i.set_xticks([])
            i.set_yticks([])

        f.tight_layout()
