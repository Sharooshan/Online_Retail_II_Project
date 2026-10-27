# Online Retail II – Next-Day Revenue Forecasting

## Project Overview

This project was developed for the **NIB7143CEM – Programming for Data Science** module of the MSc Data Science programme at NIBM, awarded by Coventry University.

The objective is to predict next-day positive-sales revenue using historical retail transaction data from the Online Retail II dataset.

The coursework covers five areas:

1. Object-oriented programming, data validation and automated testing.
2. Data storage formats and performance benchmarking.
3. Exploratory data analysis and supervised machine learning.
4. PySpark implementation and scalability evaluation.
5. Professional practice, data protection, version control and reproducibility.

## Dataset

**Dataset:** Online Retail II  
**Source:** UCI Machine Learning Repository  
**DOI:** https://doi.org/10.24432/C5CG6D  
**Download:** https://archive.ics.uci.edu/dataset/502/online+retail+ii

The original Excel workbook contains two worksheets:

- Year 2009-2010
- Year 2010-2011

The combined dataset contains **1,067,371 records and 8 original attributes**.

The original attributes are `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID` and `Country`.

After preprocessing, the cleaned dataset contains **1,007,913 records and 9 columns**, including the derived `Revenue` field.

The forecasting workflow uses daily positive-sales revenue derived from the transaction records.

### Data Availability

The original dataset is not included in the Git repository.

Download the Excel workbook from the UCI Machine Learning Repository and save it using this exact filename:

`data/online_retail_II.xlsx`

The filename is important because the data preparation script uses this path.

Raw data and customer-level analytical files should remain outside the public repository.

## Project Structure

```text
Online_Retail_II_Project/
|
|-- data/
|   |-- online_retail_II.xlsx
|
|-- outputs/
|   |-- cleaned_retail.parquet
|   |-- cleaned_retail_minimised.parquet
|
|-- results/
|   |-- task3/
|   |-- task4/
|
|-- src/
|   |-- retail_pipeline/
|       |-- __init__.py
|       |-- base.py
|       |-- exceptions.py
|       |-- loaders.py
|       |-- validation.py
|       |-- preprocessing.py
|       |-- prepare_daily_revenue.py
|       |-- task3_eda.py
|       |-- task3_feature_engineering.py
|       |-- task3_modelling.py
|       |-- task3_error_analysis.py
|       |-- task4_scalability.py
|       |-- task4_benchmark.py
|
|-- tests/
|   |-- test_exceptions.py
|   |-- test_loaders.py
|   |-- test_polymorphism.py
|   |-- test_preprocessing.py
|   |-- test_validation.py
|
|-- requirements.txt
|-- README.md
|-- .gitignore
```

The `data`, `outputs` and `results` directories may contain locally generated files that are excluded from version control.

## Software Requirements

The project was developed using the following software environment:

| Component | Version |
|---|---|
| Python | 3.12.10 |
| pandas | 3.1.0 |
| NumPy | 2.5.4 |
| openpyxl | 3.1.5 |
| PyArrow | 26.0.0 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PySpark | 3.5.7 |
| pytest | 9.1.1 |
| Java | 17 |

The project was developed on Windows using Visual Studio Code and PowerShell.

### Hardware Environment

The recorded local benchmark environment used:

- Processor: Intel Core i5-12500H
- Logical processors: 16
- Available memory reported during benchmarking: approximately 7.67 GB

Execution times may vary on other hardware.

## Installation and Environment Setup

### Step 1: Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/Sharooshan/Online_Retail_II_Project.git
cd Online_Retail_II_Project
```

### Step 2: Create a Virtual Environment

```powershell
py -3.12 -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

The `requirements.txt` file contains pinned dependency versions to support reproducibility.

### Step 4: Verify the Environment

```powershell
python --version
python -m pip check
java -version
```

The documented environment uses Python 3.12.10 and Java 17.

The dependency compatibility check previously returned:

```text
No broken requirements found.
```

### Step 5: Download the Dataset

Download the original Online Retail II workbook from:

https://archive.ics.uci.edu/dataset/502/online+retail+ii

Place the workbook at:

```text
data/online_retail_II.xlsx
```

The original workbook is required to reproduce the cleaning process from source data.

## Task 1 – Object-Oriented Data Processing

Task 1 implements an object-oriented Python package for loading, validating and preprocessing retail transaction data.

### Main Components

- `base.py`: Abstract processing interfaces.
- `loaders.py`: Excel data loading.
- `validation.py`: Dataset schema and data validation.
- `preprocessing.py`: Transaction cleaning and revenue calculation.
- `exceptions.py`: Custom exception handling.

The implementation demonstrates abstraction, inheritance, polymorphism and exception handling.

### Automated Testing

Automated tests are implemented using pytest.

Run the tests from the project root:

```powershell
python -m pytest -v
```

The previously recorded test execution passed **19 tests**.

The tests cover data loading, validation, preprocessing, exception handling and polymorphism.

## Task 2 – Storage Format Evaluation

Task 2 compares alternative storage formats for the cleaned retail dataset.

The evaluated formats include:

- CSV
- Pickle
- Parquet
- Parquet with Snappy compression

The comparison considers storage size, write time, full-dataset read time and selected-column read time.

Parquet with Snappy compression was selected as the preferred format based on the experimental findings.

The cleaned dataset is stored locally at:

```text
outputs/cleaned_retail.parquet
```

## Task 3 – Data Science Pipeline

Task 3 follows the **OSEMN framework**:

1. **Obtain:** Load and inspect the retail dataset.
2. **Scrub:** Clean transactions and prepare daily revenue.
3. **Explore:** Examine revenue trends and distributions.
4. **Model:** Construct predictive features and train regression models.
5. **iNterpret:** Evaluate forecasting accuracy and prediction errors.

### Main Scripts

```text
src/retail_pipeline/prepare_daily_revenue.py
src/retail_pipeline/task3_eda.py
src/retail_pipeline/task3_feature_engineering.py
src/retail_pipeline/task3_modelling.py
src/retail_pipeline/task3_error_analysis.py
```

### Data Preparation

The forecasting workflow constructs a continuous daily revenue series covering **739 calendar dates**, from 1 December 2009 to 9 December 2011.

The recorded validation results were:

| Validation measure | Result |
|---|---:|
| Calendar days | 739 |
| Days with transactions | 604 |
| Days without transactions | 135 |
| Missing next-day targets | 1 |
| Daily revenue validation | Passed |

Dates without qualifying positive-sales transactions are assigned zero revenue under the documented preprocessing convention.

The final modelling dataset contains 709 observations and 12 predictors.

### Predictive Modelling

Regression models were evaluated using chronological training, validation and test partitions.

Random Forest was selected based on validation performance.

The recorded test results were:

| Metric | Result |
|---|---:|
| MAE | 11,708.20 |
| RMSE | 20,635.27 |
| R² | 0.513 |
| Persistence baseline MAE | 24,424.04 |

The Random Forest model reduced test MAE by approximately **52.1%** compared with the persistence baseline.

These results describe the recorded experiment and do not guarantee future forecasting performance.

## Task 4 – PySpark Scalability Evaluation

Task 4 implements equivalent daily revenue aggregation operations using pandas and Apache Spark.

The implementation files are:

```text
src/retail_pipeline/task4_scalability.py
src/retail_pipeline/task4_benchmark.py
```

### Spark Configuration

The local Spark environment used:

- PySpark 3.5.7
- Java 17
- Two local worker threads: `local[2]`
- Four SQL shuffle partitions

### Benchmark Methodology

The experiment compares pandas and PySpark using three dataset sizes:

- 25% of the cleaned dataset
- 50% of the cleaned dataset
- 100% of the cleaned dataset

Three measured repetitions were performed for each workload.

The implementations were checked for equivalent aggregation results.

### Reported Benchmark Results

The original recorded benchmark results were:

| Input records | pandas median (seconds) | PySpark median (seconds) |
|---:|---:|---:|
| 251,978 | 0.048557 | 0.306518 |
| 503,956 | 0.075091 | 0.172752 |
| 1,007,913 | 0.151640 | 0.189634 |

Pandas was faster for all three tested dataset sizes. However, the performance difference decreased as the input size increased.

### Repeated Execution Verification

A subsequent execution on 27 October 2026 also completed successfully.

| Input records | pandas median (seconds) | PySpark median (seconds) |
|---:|---:|---:|
| 251,978 | 0.044249 | 0.274750 |
| 503,956 | 0.077765 | 0.167098 |
| 1,007,913 | 0.165553 | 0.216632 |

All nine aggregation comparisons passed validation during the repeated execution.

The two benchmark runs differ slightly because execution time depends on system conditions.

### Recorded Output Files

```text
results/task4/initial_comparison.csv
results/task4/benchmark_results.csv
results/task4/benchmark_chart.png
```

The latest benchmark execution may overwrite previously generated result files. The original reported measurements remain documented above.

## Reproducing the Experimental Results

Run all commands from the project root after activating the virtual environment and installing the pinned dependencies.

### Step 1: Prepare the Dataset

Download the original workbook and save it as:

```text
data/online_retail_II.xlsx
```

Then execute:

```powershell
python -m src.retail_pipeline.prepare_daily_revenue
```

This command was successfully executed in the documented environment.

The script checks whether `outputs/cleaned_retail.parquet` already exists. If it does, the existing cleaned dataset is reused.

For reproduction from the original Excel workbook, begin without the previously generated cleaned Parquet file.

The recorded execution produced:

- 1,007,913 cleaned transactions.
- 739 calendar dates.
- 604 dates with qualifying transactions.
- 135 dates without qualifying transactions.

The daily revenue outputs are written to:

```text
results/task3/daily_revenue.csv
results/task3/daily_revenue.parquet
```

### Step 2: Exploratory Data Analysis

```powershell
python -m src.retail_pipeline.task3_eda
```

### Step 3: Feature Engineering

```powershell
python -m src.retail_pipeline.task3_feature_engineering
```

### Step 4: Predictive Modelling

```powershell
python -m src.retail_pipeline.task3_modelling
```

### Step 5: Prediction Error Analysis

```powershell
python -m src.retail_pipeline.task3_error_analysis
```

The Task 3 scripts generate analytical summaries, modelling results and visualisations under `results/task3/`.

The modelling outputs include:

```text
results/task3/validation_results.csv
results/task3/test_results.csv
results/task3/test_predictions.csv
results/task3/experiment_summary.json
```

The Task 3 commands above follow the scripts' documented entry points. A complete sequential clean-checkout run should be performed to verify all input and output dependencies.

### Step 6: PySpark Scalability Evaluation

Ensure Java 17 is available.

Run the scalability comparison:

```powershell
python -m src.retail_pipeline.task4_scalability
```

Run the repeated benchmark:

```powershell
python -m src.retail_pipeline.task4_benchmark
```

The repeated benchmark command was successfully executed in the documented environment.

It generates:

```text
results/task4/benchmark_results.csv
results/task4/benchmark_chart.png
```

The benchmark compares pandas and PySpark at 25%, 50% and 100% of the cleaned dataset using three measured repetitions.

### Step 7: Automated Testing

```powershell
python -m pytest -v
```

The previously recorded test execution passed 19 tests.

### Reproducibility Notes

- Execute the scripts in the documented order.
- Download the original Excel workbook separately.
- Use the pinned dependency versions.
- Retain the documented data cleaning and chronological modelling procedures.
- Benchmark timings may vary with hardware, memory, operating-system conditions and software configuration.
- Aggregation results should be checked for numerical equivalence rather than identical execution times.
- Generated transaction-level data and modelling datasets should be assessed for disclosure risks before publication.

## Task 5 – Professional and Legal Practice

### Version Control

Git and GitHub were used throughout project development.

The repository contains incremental commits covering data loading, validation, preprocessing, testing, storage evaluation, predictive modelling and scalability analysis.

Feature branches were used to isolate development work before integration into the main branch.

For example, Task 4 was implemented on:

```text
feature/task4-spark
```

The implementation commit was:

```text
d8cbfc9 Evaluate Spark aggregation scalability against pandas with validated benchmarks
```

The feature branch was merged into `main` using merge commit:

```text
385cbfb Merge branch 'feature/task4-spark'
```

The merge was successfully pushed to the GitHub repository.

This provides evidence of incremental development, meaningful commit messages and feature-branch integration.

### Data Protection

The Online Retail II dataset contains a `Customer ID` attribute.

Although this field does not directly reveal customer names, it may constitute pseudonymous personal data if individuals can be identified using additional information.

The project considers GDPR Article 5 principles, including:

- Lawfulness, fairness and transparency.
- Purpose limitation.
- Data minimisation.
- Accuracy.
- Storage limitation.
- Integrity and confidentiality.
- Accountability.

The project uses transaction data for retail revenue analysis and forecasting.

Individual customer identifiers are not necessary for daily revenue aggregation.

To demonstrate data minimisation, a separate analytical dataset was created containing only:

```text
InvoiceDate
Revenue
```

The derived dataset contains **1,007,913 records** and is stored locally as:

```text
outputs/cleaned_retail_minimised.parquet
```

The `Customer ID` field is absent from this derived dataset.

This reduces unnecessary processing of customer identifiers but does not automatically establish complete anonymisation.

The original cleaned dataset remains available locally for reproducibility.

The `.gitignore` configuration excludes the raw data directory, generated outputs, Excel workbooks, Parquet files and other potentially sensitive artefacts.

The dataset's publication licence does not automatically establish a lawful basis for processing personal data. The appropriate lawful basis, retention period, access restrictions and institutional obligations require separate consideration.

Raw and customer-level data should not be committed to a public repository without an appropriate data-protection assessment.

### Reproducibility

The project supports reproducibility through:

- Version-controlled Python source code.
- Pinned dependency versions in `requirements.txt`.
- A documented Python and Java environment.
- Operating-system and hardware information.
- Public dataset source information.
- Automated pytest tests.
- Documented preprocessing and modelling procedures.
- Recorded experimental results.
- Repeated benchmark validation.
- Explicit discussion of hardware-dependent execution times.

## Limitations

- The dataset represents historical retail transactions and may not reflect current purchasing behaviour.
- Forecast accuracy depends on the quality and representativeness of historical data.
- Dates without qualifying transactions are assigned zero revenue according to the preprocessing convention.
- Positive-sales revenue excludes cancellations and returns.
- The forecasting model may perform differently on future data.
- Spark was evaluated on a single computer rather than a distributed cluster.
- Execution timings depend on hardware, memory and system load.
- Removing customer identifiers does not guarantee that all re-identification risks have been eliminated.
- The complete clean-checkout execution sequence should be tested before final submission.

## References

Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

Information Commissioner's Office. *A guide to the data protection principles*. https://ico.org.uk/

Apache Software Foundation. *Apache Spark Documentation*. https://spark.apache.org/docs/

pandas development team. *pandas documentation*. https://pandas.pydata.org/docs/

scikit-learn developers. *scikit-learn documentation*. https://scikit-learn.org/stable/
