
# Import the pandas library for working with datasets.
import pandas as pd

# Location of the Excel workbook inside the project.
file_path = "data/online_retail_II.xlsx"

# Open the workbook and read its worksheet names.
excel_file = pd.ExcelFile(file_path)

# Display the worksheet names.
print("Excel worksheets:")
print(excel_file.sheet_names)


# Read the first 10 rows from the first worksheet.
# nrows=10 avoids loading the entire dataset for this first inspection.
df = pd.read_excel(
    file_path,
    sheet_name="Year 2009-2010",
    nrows=10
)

# Show the first five records.
print("\nFirst five rows:")
print(df.head())

# Show the column names.
print("\nColumn names:")
print(df.columns.tolist())

# Show the data type of each column.
print("\nData types:")
print(df.dtypes)

# Show the number of rows and columns currently loaded.
print("\nSample shape (rows, columns):")
print(df.shape)



# Read all records from both worksheets.
df_2009 = pd.read_excel(
    file_path,
    sheet_name="Year 2009-2010"
)

df_2010 = pd.read_excel(
    file_path,
    sheet_name="Year 2010-2011"
)

# Display the number of rows and columns in each worksheet.
print("\n2009-2010 worksheet shape:")
print(df_2009.shape)

print("\n2010-2011 worksheet shape:")
print(df_2010.shape)

# Combine both worksheets into one DataFrame.
# ignore_index=True creates a continuous row index.
df_full = pd.concat(
    [df_2009, df_2010],
    ignore_index=True
)

# Display the total number of rows and columns.
print("\nCombined dataset shape:")
print(df_full.shape)

# Count missing values in each column.
print("\nMissing values by column:")
print(df_full.isna().sum())



# Count rows that are exact duplicates across all columns.
duplicate_count = df_full.duplicated().sum()

print("\nStep 10 - Duplicate records:")
print("Number of duplicate rows:", duplicate_count)


# Convert invoice numbers to text so we can inspect their prefixes.
invoice_text = df_full["Invoice"].astype(str)

# Identify rows whose invoice number starts with C.
cancelled_rows = invoice_text.str.startswith("C")

# Count the affected rows.
print("\nStep 11 - Cancelled transactions:")
print("Rows with cancelled invoice numbers:", cancelled_rows.sum())

# Count distinct cancelled invoice numbers.
print(
    "Unique cancelled invoices:",
    df_full.loc[cancelled_rows, "Invoice"].nunique()
)


# Step 12: Inspect unusual quantity and price values.

# Count rows with negative quantities.
negative_quantity = (df_full["Quantity"] < 0).sum()

# Count rows with zero quantities.
zero_quantity = (df_full["Quantity"] == 0).sum()

# Count rows with negative prices.
negative_price = (df_full["Price"] < 0).sum()

# Count rows with zero prices.
zero_price = (df_full["Price"] == 0).sum()

print("\nStep 12 - Quantity and Price Checks:")
print("Negative quantities:", negative_quantity)
print("Zero quantities:", zero_quantity)
print("Negative prices:", negative_price)
print("Zero prices:", zero_price)



# Step 13: Compare cancellations with negative quantities.

# True when the invoice begins with C.
is_cancelled = df_full["Invoice"].astype(str).str.startswith("C")

# True when the quantity is negative.
is_negative_quantity = df_full["Quantity"] < 0

# Count each combination of these two conditions.
comparison = pd.crosstab(
    is_cancelled,
    is_negative_quantity,
    rownames=["Cancelled invoice"],
    colnames=["Negative quantity"]
)

print("\nStep 13 - Cancellation vs Negative Quantity:")
print(comparison)



# Step 14: Inspect negative quantities without C-prefixed invoices.

unusual_rows = df_full[
    (~is_cancelled) & (is_negative_quantity)
]

print("\nStep 14 - Negative quantities without cancellation prefix:")
print("Total unusual rows:", len(unusual_rows))

# Display only the first 10 records for inspection.
print(unusual_rows.head(10).to_string(index=False))



# Step 15: Investigate negative quantities without C-prefixed invoices.

print("\nStep 15 - Unusual Record Investigation:")

# Count records with a zero price.
print(
    "Rows with zero price:",
    (unusual_rows["Price"] == 0).sum()
)

# Count records with a missing Customer ID.
print(
    "Rows with missing Customer ID:",
    unusual_rows["Customer ID"].isna().sum()
)

# Count records with a missing description.
print(
    "Rows with missing Description:",
    unusual_rows["Description"].isna().sum()
)

# Show the most frequently occurring descriptions.
print("\nTop 10 descriptions:")
print(
    unusual_rows["Description"]
    .value_counts()
    .head(10)
)


# Step 16: Inspect zero-price records.

# Select records with a price of zero.
zero_price_rows = df_full[df_full["Price"] == 0]

print("\nStep 16 - Zero Price Investigation:")

print("Total zero-price rows:", len(zero_price_rows))

# Count zero-price rows with positive and negative quantities.
print(
    "Positive quantities:",
    (zero_price_rows["Quantity"] > 0).sum()
)

print(
    "Negative quantities:",
    (zero_price_rows["Quantity"] < 0).sum()
)

# Count missing customer identifiers.
print(
    "Missing Customer ID:",
    zero_price_rows["Customer ID"].isna().sum()
)

# Show frequently occurring descriptions.
print("\nTop 10 descriptions:")
print(
    zero_price_rows["Description"]
    .value_counts()
    .head(10)
)



# Step 17: Inspect records containing negative prices.

negative_price_rows = df_full[df_full["Price"] < 0]

print("\nStep 17 - Negative Price Investigation:")
print("Total negative-price records:", len(negative_price_rows))

# Display the matching records.
print(negative_price_rows.to_string(index=False))


# Step 18: Check how many calendar days the dataset covers.

# Find the earliest and latest transaction dates.
first_date = df_full["InvoiceDate"].min()
last_date = df_full["InvoiceDate"].max()

# Count how many different dates contain transactions.
unique_days = df_full["InvoiceDate"].dt.date.nunique()

# Count all calendar days between the first and last date.
calendar_days = (last_date.normalize() - first_date.normalize()).days + 1

# Display the results.
print("\nStep 18 - Daily Data Coverage:")
print("First transaction:", first_date)
print("Last transaction:", last_date)
print("Calendar days:", calendar_days)
print("Days with transactions:", unique_days)
print("Days without transactions:", calendar_days - unique_days)

# Step 19: Investigate calendar days without transactions.

# Create a complete calendar covering the dataset.
all_dates = pd.date_range(
    start=df_full["InvoiceDate"].min().normalize(),
    end=df_full["InvoiceDate"].max().normalize(),
    freq="D"
)

# Identify the dates where transactions were recorded.
transaction_dates = pd.DatetimeIndex(
    df_full["InvoiceDate"].dt.normalize().unique()
)

# Find dates with no recorded transactions.
missing_dates = all_dates.difference(transaction_dates)

# Count missing dates by weekday.
missing_weekdays = pd.Series(
    missing_dates.day_name()
).value_counts()

print("\nStep 19 - Missing Days by Weekday:")
print(missing_weekdays)

print("\nFirst 15 dates without transactions:")
print(missing_dates[:15])

# Step 20: Count transaction records by weekday.

# Extract the weekday name from each transaction date.
weekdays = df_full["InvoiceDate"].dt.day_name()

# Count records for each weekday.
weekday_counts = weekdays.value_counts()

# Display the results in calendar order.
weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

print("\nStep 20 - Transaction Records by Weekday:")
print(weekday_counts.reindex(weekday_order, fill_value=0))