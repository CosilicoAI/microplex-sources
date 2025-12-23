"""
Microdata loading for calibration.

Loads raw microdata (CPS, FRS) with original survey weights.
"""

from typing import Optional

import pandas as pd


def load_microdata(
    source: str,
    year: int,
    variables: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load microdata for calibration.

    Args:
        source: Data source ("cps" for US, "frs" for UK)
        year: Year of microdata
        variables: Optional list of variables to load

    Returns:
        DataFrame with microdata and original weights in 'weight' column

    Note:
        This is a placeholder. Actual implementation will integrate
        with cosilico-microdata or PolicyEngine data loading.
    """
    raise NotImplementedError(
        f"Microdata loading for {source} {year} not yet implemented. "
        "This will integrate with cosilico-microdata."
    )
