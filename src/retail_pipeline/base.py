
from abc import ABC, abstractmethod

import pandas as pd


class RetailPipelineStage(ABC):
    """Define the common interface for retail processing stages."""

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process a DataFrame and return the resulting DataFrame."""
        pass
