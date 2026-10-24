
from src.retail_pipeline.loaders import RetailDataLoader

# Create an instance of our data loader.
loader = RetailDataLoader("data/online_retail_II.xlsx")

# Read both worksheets.
data = loader.load_excel(
    ("Year 2009-2010", "Year 2010-2011")
)

# Display the results.
print("Dataset shape:", data.shape)
print("Columns:", data.columns.tolist())
