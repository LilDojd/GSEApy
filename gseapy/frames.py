"""Convert polars inputs and results without pyarrow."""

import math
import sys
from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype
from pandas.api.types import is_bool_dtype, is_numeric_dtype

if TYPE_CHECKING:
    import polars as pl

__all__ = ["PolarsResults", "is_polars", "to_pandas", "to_polars"]


def is_polars(obj: Any) -> bool:
    """Whether ``obj`` is a polars DataFrame, LazyFrame or Series."""
    pl = sys.modules.get("polars")
    return pl is not None and isinstance(obj, (pl.DataFrame, pl.LazyFrame, pl.Series))


def to_pandas(obj: Any) -> Any:
    """Return a polars object as pandas; anything else unchanged."""
    if not is_polars(obj):
        return obj
    import polars as pl

    if isinstance(obj, pl.LazyFrame):
        obj = obj.collect()
    if isinstance(obj, pl.Series):
        return pd.Series(obj.to_numpy(), name=obj.name)
    return pd.DataFrame({name: obj.get_column(name).to_numpy() for name in obj.columns})


def to_polars(obj: Union[pd.DataFrame, pd.Series]) -> Union["pl.DataFrame", "pl.Series"]:
    """Return a pandas DataFrame or Series as polars, with any index as columns."""
    try:
        import polars as pl
    except ImportError as error:
        raise ImportError("polars output requires polars: pip install gseapy[polars]") from error

    if isinstance(obj, pd.Series):
        return pl.Series(str(obj.name) if obj.name is not None else "", _values(obj))
    if not isinstance(obj.index, pd.RangeIndex) or obj.index.name is not None:
        obj = obj.reset_index()
    return pl.DataFrame({str(name): _values(obj[name]) for name in obj.columns})


def _values(column: pd.Series) -> Union[np.ndarray, List[Any]]:
    dtype = column.dtype
    if (is_numeric_dtype(dtype) or is_bool_dtype(dtype)) and not isinstance(dtype, ExtensionDtype):
        return column.to_numpy()
    return [None if _missing(value) else value for value in column.tolist()]


def _missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


class PolarsResults:
    """Adds ``to_polars()`` for ``res2d``."""

    res2d: pd.DataFrame

    def to_polars(self) -> "pl.DataFrame":
        """``res2d`` as a polars DataFrame."""
        if getattr(self, "res2d", None) is None:
            raise ValueError("No results yet: call run() first")
        return to_polars(self.res2d)
