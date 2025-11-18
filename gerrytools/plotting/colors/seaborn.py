import seaborn as sns


def redbluecmap(n: int) -> list[tuple[float, float, float]]:
    """
    Generates a red/white/blue color palette in `n` colors, using the
    `coolwarm` diverging colormap from seaborn.

    Args:
        n (int): The number of colors to generate.

    Returns:
        List of RGB triples (each in [0, 1]).
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    colors = list(reversed(sns.color_palette("coolwarm", n_colors=n)))

    # Make the grey color in the middle more white.
    if n % 2 == 1:
        mid = n // 2
        colors = list(colors)
        colors[mid] = (240 / 255, 240 / 255, 240 / 255)

    return colors


def flare(n: int) -> list[tuple[float, float, float]]:
    """
    Generates a red-to-purple color palette in `n` colors, using the
    `flare` colormap from seaborn.

    Args:
        n (int): Number of colors to generate.

    Returns:
        List of RGB triples (each in [0, 1]).
    """
    return list(sns.color_palette("flare", as_cmap=False, n_colors=n))


def purples(n: int) -> list[tuple[float, float, float]]:
    """
    Generates a list of `n` shades of purple basesd on the `Purples`
    Matplotlib/seaborn colormap.

    Args:
        n (int): Number of colors to generate.

    Returns:
        List of RGB triples (each in [0, 1]).
    """
    return list(sns.color_palette("Purples", as_cmap=False, n_colors=n))


def greens(n: int) -> list[tuple[float, float, float]]:
    """
    Generates a list of `n` shades of purple basesd on the `Greens`
    Matplotlib/seaborn colormap.

    Args:
        n (int): Number of colors to generate.

    Returns:
        List of RGB triples (each in [0, 1]).
    """
    return list(sns.color_palette("Greens", as_cmap=False, n_colors=n))
