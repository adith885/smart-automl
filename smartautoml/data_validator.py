import pandas as pd
import numpy as np

class DataValidator:

    def validate(self, df, target):
        df = df.copy()

        # Remove empty rows
        df = df.dropna(how="all")

        # Check target exists
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))

        return df