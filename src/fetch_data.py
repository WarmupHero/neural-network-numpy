import os
import pandas as pd

from src.utils import DATASETS_DIR

class Fetch:
    def __init__(self):
        # Ensure the datasets directory exists
        os.makedirs(DATASETS_DIR, exist_ok=True)

        # Define dataset file paths
        self.banknote_path = os.path.join(DATASETS_DIR, "banknote_auth.csv")
        self.energy_path = os.path.join(DATASETS_DIR, "energy_efficiency.csv")

    def download_all(self):
        """Download datasets if they do not already exist locally."""

        # Banknote Authentication (Classification)
        if not os.path.exists(self.banknote_path):
            # In case of module error, pip install ucimlrepo
            from ucimlrepo import fetch_ucirepo
            print("Downloading Banknote Authentication dataset...")
            banknote = fetch_ucirepo(id=267)
            # Re-attach X and y for EDA
            df_bank = pd.concat([banknote.data.features, banknote.data.targets], axis=1)
            df_bank.to_csv(self.banknote_path, index=False)
            print("Downloaded classification dataset.")
        else:
            print("Banknote Authentication dataset found locally. Skipping download.")

        # Energy Efficiency (Regression)
        if not os.path.exists(self.energy_path):
            from ucimlrepo import fetch_ucirepo
            print("Downloading Energy Efficiency dataset...")
            energy = fetch_ucirepo(id=242)
            # Re-attach X and y for EDA
            df_energy = pd.concat([energy.data.features, energy.data.targets], axis=1)

            # Rename columns using the official dataset variable descriptions
            df_energy.columns = [
                "Relative_Compactness",       # X1
                "Surface_Area",               # X2
                "Wall_Area",                  # X3
                "Roof_Area",                  # X4
                "Overall_Height",             # X5
                "Orientation",                # X6
                "Glazing_Area",               # X7
                "Glazing_Area_Distribution",  # X8
                "Heating_Load",               # Y1
                "Cooling_Load"                # Y2
            ]

            df_energy.to_csv(self.energy_path, index=False)
            print("Downloaded regression dataset.")
        else:
            print("Energy Efficiency dataset found locally. Skipping download.")

        print("Data downloaded successfully!")


if __name__ == "__main__":
    fetch = Fetch()
    fetch.download_all()