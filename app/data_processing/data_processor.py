import pandas as pd
from .field_map_handler import FieldMapHandler

class DataProcessor:
    """Handles data loading and preprocessing."""

    def __init__(self, csv_path, field_map_path=None, tube_ids_path=None):
        self.csv_path = csv_path
        self.field_map_handler = None
        if field_map_path and tube_ids_path:
            self.field_map_handler = FieldMapHandler(field_map_path, tube_ids_path)
            self.field_map_handler.load_data()
            
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
            df["Length (mm)"] = pd.to_numeric(
                df["Length (mm)"], downcast="float", errors="coerce"
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
            
            # Merge Experimental Data if handler is available
            if self.field_map_handler:
                def get_meta(tube, field):
                    return self.field_map_handler.get_tube_metadata(tube).get(field, "Unknown")

                df["Treatment"] = df["Tube"].apply(lambda t: get_meta(t, "Treatment"))
                df["Genotype"] = df["Tube"].apply(lambda t: get_meta(t, "Genotype"))
                df["Rng"] = df["Tube"].apply(lambda t: get_meta(t, "Rng"))
                df["Col"] = df["Tube"].apply(lambda t: get_meta(t, "Col"))
                
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

    def get_unique_treatments(self):
        if "Treatment" in self.df.columns:
            return sorted(self.df["Treatment"].astype(str).unique())
        return []

    def get_unique_genotypes(self):
        if "Genotype" in self.df.columns:
            return sorted(self.df["Genotype"].astype(str).unique())
        return []
