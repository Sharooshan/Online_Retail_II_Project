import pandas as pd

from src.retail_pipeline.base import RetailPipelineStage
from src.retail_pipeline.validation import RetailDataValidator, REQUIRED_COLUMNS
from src.retail_pipeline.preprocessing import RetailDataPreprocessor


def test_pipeline_stage_inheritance():
    assert issubclass(RetailDataValidator, RetailPipelineStage)
    assert issubclass(RetailDataPreprocessor, RetailPipelineStage)


def test_validation_stage_process():
    sample = pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    stage = RetailDataValidator(REQUIRED_COLUMNS)
    result = stage.process(sample)

    assert result is sample


def test_preprocessing_stage_process():
    sample = pd.DataFrame({
        "Invoice": ["10001"],
        "StockCode": ["A"],
        "Description": ["Item A"],
        "Quantity": [2],
        "InvoiceDate": ["2010-01-01"],
        "Price": [5.0],
        "Customer ID": [None],
        "Country": ["United Kingdom"],
    })

    stage = RetailDataPreprocessor()
    result = stage.process(sample)

    assert isinstance(stage, RetailPipelineStage)
    assert result["Revenue"].iloc[0] == 10.0