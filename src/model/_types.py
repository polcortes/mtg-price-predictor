from typing import Dict, Annotated

type DatePricesMap = Annotated[Dict[str, float], "A dictionary that maps dates to prices. Example: {'2022-01-01': 10.0, '2022-01-02': 11.0, '2022-01-03': 12.0}"]