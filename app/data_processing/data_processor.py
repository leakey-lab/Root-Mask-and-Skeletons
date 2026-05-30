import pandas as pd


class MetricDataProcessor:
    """
    Loads and prepares a root-metric CSV (length or area) for visualization.

    The length and area pipelines previously had two near-identical processor
    classes; they now share this base, parameterized only by the metric column.
    """

    #: name of the numeric metric column in the CSV (e.g. "Length (mm)")
    value_column = "Length (mm)"

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        try:
            df = pd.read_csv(self.csv_path, encoding="latin-1")
            df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d", errors="coerce")
            df["Tube"] = pd.to_numeric(df["Tube"], downcast="integer", errors="coerce")
            df["Position"] = pd.to_numeric(df["Position"], downcast="integer", errors="coerce")
            df[self.value_column] = pd.to_numeric(
                df[self.value_column], downcast="float", errors="coerce"
            )
            df.dropna(inplace=True)
            df["tube_date"] = df.apply(
                lambda x: f"Tube {int(x['Tube'])} ({x['Date'].strftime('%Y-%m-%d')})", axis=1
            )
            df["tube_position"] = df.apply(
                lambda x: f"Tube {int(x['Tube'])}_L{int(x['Position'])}", axis=1
            )
            return df
        except Exception:
            return pd.DataFrame()

    def get_unique_tubes(self):
        return sorted(self.df["Tube"].unique())

    def get_unique_dates(self):
        return sorted(self.df["Date"].unique())

    def get_unique_positions(self):
        return sorted(self.df["Position"].unique())

    def get_unique_treatments(self):
        if "Treatment" in self.df.columns:
            return sorted(self.df["Treatment"].astype(str).unique())
        return []

    def get_unique_genotypes(self):
        if "Genotype" in self.df.columns:
            return sorted(self.df["Genotype"].astype(str).unique())
        return []


class DataProcessor(MetricDataProcessor):
    """Root-length data processor."""

    value_column = "Length (mm)"
