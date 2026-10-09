"""
Download the two UCI datasets used by the project.

Input: the UCI Machine Learning Repository, reached through the optional
``ucimlrepo`` package (Banknote Authentication, id 267, and Energy
Efficiency, id 242).

Output: two CSV files in ``data/raw/`` (``banknote_auth.csv`` and
``energy_efficiency.csv``), each holding the feature columns followed by
the target column(s). A file that already exists is not downloaded again.
"""

import os

import pandas as pd

from nn_numpy.config import DATASETS_DIR


class Fetch:
    """
    Download the project's datasets into the datasets folder.

    The class only remembers where each CSV file should live; the actual
    download happens in ``download_all``.

    Attributes
    ----------
    banknote_path : str
        Full path of the Banknote Authentication CSV
        (``data/raw/banknote_auth.csv``), used for binary classification.
    energy_path : str
        Full path of the Energy Efficiency CSV
        (``data/raw/energy_efficiency.csv``), used for regression.
    """

    def __init__(self) -> None:
        """
        Create the datasets folder and store the two CSV file paths.

        Parameters
        ----------
        None
            Uses the module-level constant ``DATASETS_DIR`` from nn_numpy.config.

        Returns
        -------
        None
            Creates ``DATASETS_DIR`` on disk if it does not exist and sets
            ``self.banknote_path`` and ``self.energy_path``.

        Notes
        -----
        Processing:
        1. Create ``DATASETS_DIR`` (no error if it already exists).
        2. Join ``DATASETS_DIR`` with each dataset's file name and store
           the two paths as attributes.
        """
        # Ensure the datasets directory exists
        os.makedirs(DATASETS_DIR, exist_ok=True)

        # Define dataset file paths
        self.banknote_path = os.path.join(DATASETS_DIR, "banknote_auth.csv")
        self.energy_path = os.path.join(DATASETS_DIR, "energy_efficiency.csv")

    def download_all(self) -> None:
        """
        Download each dataset from the UCI repository unless its CSV already exists.

        Parameters
        ----------
        None
            Uses ``self.banknote_path`` and ``self.energy_path``.

        Returns
        -------
        None
            Writes ``banknote_auth.csv`` and/or ``energy_efficiency.csv`` to
            the datasets folder (only the ones that are missing) and prints
            progress messages.

        Raises
        ------
        ModuleNotFoundError
            If a download is needed and the ``ucimlrepo`` package is not
            installed (install it with ``pip install ucimlrepo``).

        Notes
        -----
        Processing, for each of the two datasets:
        1. If the CSV file already exists, print a message and skip it.
        2. Otherwise import ``fetch_ucirepo`` and download the dataset by
           its UCI id (267 for Banknote Authentication, 242 for Energy
           Efficiency).
        3. Concatenate the feature DataFrame and the target DataFrame side
           by side, so the CSV holds features and targets together for EDA.
        4. For Energy Efficiency only, replace the generic column names
           (X1..X8, Y1, Y2) with the descriptive names from the official
           dataset documentation.
        5. Save the combined DataFrame to CSV without the row index.

        ``ucimlrepo`` is imported inside the function, so it is only needed
        when a file actually has to be downloaded.
        """

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
                "Relative_Compactness",  # X1
                "Surface_Area",  # X2
                "Wall_Area",  # X3
                "Roof_Area",  # X4
                "Overall_Height",  # X5
                "Orientation",  # X6
                "Glazing_Area",  # X7
                "Glazing_Area_Distribution",  # X8
                "Heating_Load",  # Y1
                "Cooling_Load",  # Y2
            ]

            df_energy.to_csv(self.energy_path, index=False)
            print("Downloaded regression dataset.")
        else:
            print("Energy Efficiency dataset found locally. Skipping download.")

        print("Data downloaded successfully!")


# Running this file directly downloads any missing dataset CSVs into data/raw/.
if __name__ == "__main__":
    fetch = Fetch()
    fetch.download_all()
