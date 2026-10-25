import pandas as pd

from src.retail_pipeline.loaders import RetailDataLoader
from src.retail_pipeline.preprocessing import RetailDataPreprocessor

data = RetailDataLoader("data/online_retail_II.xlsx").load()
cleaned = RetailDataPreprocessor().process(data)

dates = pd.to_datetime(cleaned["InvoiceDate"]).dt.normalize()
calendar = pd.date_range(dates.min(), dates.max(), freq="D")
observed = pd.DatetimeIndex(dates.unique()).sort_values()
missing = calendar.difference(observed)

print("First date:", calendar.min().date())
print("Last date:", calendar.max().date())
print("Total calendar days:", len(calendar))
print("Days with positive-sales transactions:", len(observed))
print("Days without positive-sales transactions:", len(missing))
print("Missing days by weekday:")
# print(pd.Series(missing.day_name()).value_counts().to_string())
print(
    pd.Series(missing.day_name())
    .value_counts()
    .reindex(
        [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ],
        fill_value=0
    )
    .to_string()
)
print("First 20 missing dates:")
print(missing[:20].strftime("%Y-%m-%d").tolist())