import seaborn as sns


def _seaborn_palette(
    name: str, n: int, *, reverse: bool = False
) -> list[tuple[float, float, float]]:
    """One validated core for the seaborn palette wrappers below."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    colors = list(sns.color_palette(name, n_colors=n))
    return list(reversed(colors)) if reverse else colors


def redbluecmap(n: int) -> list[tuple[float, float, float]]:
    """
    Generates a red/white/blue color palette in `n` colors, using the `bwr` diverging colormap
    (reversed, so red comes first) from seaborn.

    Args:
        n (int): The number of colors to generate.

    Returns:
        List of RGB triples (each in [0, 1]).
    """
    return _seaborn_palette("bwr", n, reverse=True)


def greenpurplecmap(n: int) -> list[tuple[float, float, float]]:
    """
    Generates a green/white/purple color palette in `n` colors, using the `PRGn` diverging colormap
    (reversed, so green comes first) from seaborn.

    Args:
        n (int): The number of colors to generate.

    Returns:
        List of RGB triples (each in [0, 1]).
    """
    colors = _seaborn_palette("PRGn", n, reverse=True)

    # Use a consistent light neutral midpoint.
    if n % 2 == 1:
        colors[n // 2] = (240 / 255, 240 / 255, 240 / 255)

    return colors


def flare(n: int) -> list[tuple[float, float, float]]:
    """
    Generates a red-to-purple color palette in `n` colors, using the `flare` colormap from seaborn.

    Args:
        n (int): Number of colors to generate.

    Returns:
        List of RGB triples (each in [0, 1]).
    """
    return _seaborn_palette("flare", n)


def purples(n: int) -> list[tuple[float, float, float]]:
    """
    Generates a list of `n` shades of purple based on the `Purples` Matplotlib/seaborn colormap.

    Args:
        n (int): Number of colors to generate.

    Returns:
        List of RGB triples (each in [0, 1]).
    """
    return _seaborn_palette("Purples", n)


def greens(n: int) -> list[tuple[float, float, float]]:
    """
    Generates a list of `n` shades of green based on the `Greens` Matplotlib/seaborn colormap.

    Args:
        n (int): Number of colors to generate.

    Returns:
        List of RGB triples (each in [0, 1]).
    """
    return _seaborn_palette("Greens", n)
