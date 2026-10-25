import pandas as pd

from storage_benchmark import (
    prepare_dataset,
    read_dataset,
    OUTPUT_DIR
)


# --------------------------------------------------
# Load the original cleaned dataset
# --------------------------------------------------

original = prepare_dataset().reset_index(drop=True)


# --------------------------------------------------
# Read the previously saved CSV dataset
# --------------------------------------------------

restored = read_dataset(
    OUTPUT_DIR / "retail_plain.csv",
    "csv"
).reset_index(drop=True)


# --------------------------------------------------
# Check dataset dimensions
# --------------------------------------------------

print("Original shape:", original.shape)
print("Restored shape:", restored.shape)


# --------------------------------------------------
# Normalize identifier columns for comparison
# --------------------------------------------------

identifier_columns = ["Invoice", "StockCode"]

for column in identifier_columns:
    original[column] = original[column].astype("string")
    restored[column] = restored[column].astype("string")


# --------------------------------------------------
# Verify all nine columns
# --------------------------------------------------

pd.testing.assert_frame_equal(
    original,
    restored,
    check_dtype=False,
    check_exact=False,
    rtol=1e-10,
    atol=1e-8
)


print("All nine columns passed the data-integrity check!")