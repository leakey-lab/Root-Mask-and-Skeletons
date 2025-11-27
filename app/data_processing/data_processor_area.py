import pandas as pd


class DataProcessorArea:
    """Handles data loading and preprocessing for root area data."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """Load and preprocess data from CSV."""
        try:
            df = pd.read_csv(self.csv_path, encoding="latin-1")
            # Convert Date to datetime
            df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d", errors="coerce")

            # Convert numeric columns
            df["Tube"] = pd.to_numeric(df["Tube"], downcast="integer", errors="coerce")
            df["Position"] = pd.to_numeric(
                df["Position"], downcast="integer", errors="coerce"
            )
            df["Area (mm²)"] = pd.to_numeric(
                df["Area (mm²)"], downcast="float", errors="coerce"
            )

            # Drop rows with any NaN values
            df.dropna(inplace=True)

            # Pre-compute identifiers
            df["tube_date"] = df.apply(
                lambda x: f"Tube {int(x['Tube'])} ({x['Date'].strftime('%Y-%m-%d')})",
                axis=1,
            )
            df["tube_position"] = df.apply(
                lambda x: f"Tube {int(x['Tube'])}_L{int(x['Position'])}", axis=1
            )
            return df
        except Exception:
            return pd.DataFrame()

    def get_unique_tubes(self):
        """Return sorted unique tubes."""
        return sorted(self.df["Tube"].unique())

    def get_unique_dates(self):
        """Return sorted unique dates."""
        return sorted(self.df["Date"].unique())

    def get_unique_positions(self):
        """Return sorted unique positions."""
        return sorted(self.df["Position"].unique())
