from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass(frozen=True)
class DividendSchedule:
    # List of (ex_date_fractional_years, cash_amount)
    dividends: List[Tuple[float, float]]

@dataclass(frozen=True)
class EuropeanOption:
    spot: float
    strike: float
    maturity: float      # T in years
    rate: float          # risk-free rate r
    div_yield: float     # continuous dividend yield q
    option_type: str     # "call" or "put"
    dividend_schedule: Optional[DividendSchedule] = None
