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
        self._compute_unique_caches()

    def _compute_unique_caches(self):
        """Compute the sorted unique lookups once instead of on every accessor call."""
        self._unique_tubes = sorted(self.df["Tube"].unique()) if "Tube" in self.df.columns else []
        self._unique_dates = sorted(self.df["Date"].unique()) if "Date" in self.df.columns else []
        self._unique_positions = (
            sorted(self.df["Position"].unique()) if "Position" in self.df.columns else []
        )
        self._unique_treatments = (
            sorted(self.df["Treatment"].astype(str).unique())
            if "Treatment" in self.df.columns
            else []
        )
        self._unique_genotypes = (
            sorted(self.df["Genotype"].astype(str).unique())
            if "Genotype" in self.df.columns
            else []
        )

    def _load_and_prepare_data(self):
        try:
            df = pd.read_csv(self.csv_path, encoding="latin-1")
            df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d", errors="coerce")
            df["Tube"] = pd.to_numeric(df["Tube"], downcast="integer", errors="coerce")
            df["Position"] = pd.to_numeric(df["Position"], downcast="integer", errors="coerce")
            df[self.value_column] = pd.to_numeric(
                df[self.value_column], errors="coerce"
            ).astype("float64")
            df.dropna(subset=["Date", "Tube", "Position", self.value_column], inplace=True)
            date_str = df["Date"].dt.strftime("%Y-%m-%d")
            tube = df["Tube"].astype("int64").astype(str)
            df["tube_date"] = "Tube " + tube + " (" + date_str + ")"
            df["tube_position"] = (
                "Tube " + tube + "_L" + df["Position"].astype("int64").astype(str)
            )
            return df
        except Exception:
            return pd.DataFrame()

    def get_unique_tubes(self):
        return self._unique_tubes

    def get_unique_dates(self):
        return self._unique_dates

    def get_unique_positions(self):
        return self._unique_positions

    def get_unique_treatments(self):
        return self._unique_treatments

    def get_unique_genotypes(self):
        return self._unique_genotypes


class DataProcessor(MetricDataProcessor):
    """Root-length data processor."""

    value_column = "Length (mm)"
