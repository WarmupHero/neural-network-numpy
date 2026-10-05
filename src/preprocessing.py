import os
import numpy as np
import pandas as pd

# Import shared project settings:
# - DATASETS_DIR: folder where the CSV files are stored
# - RANDOM_SEED: fixed seed for reproducibility
from src.utils import DATASETS_DIR, RANDOM_SEED

# Import our custom standard scaler
from src.scalers import StandardScaler

# Import plotting utilities for EDA
from src.visualizations import Visualizer


def train_val_test_split(df, val_size=0.2, test_size=0.2, stratify_col=None, random_seed=RANDOM_SEED):
    """
    Randomly split a dataframe into train / validation / test sets using NumPy only.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to split.
    val_size : float, default=0.2
        Fraction of rows assigned to the validation set.
    test_size : float, default=0.2
        Fraction of rows assigned to the test set.
        The remaining rows form the training set.
    stratify_col : str or None, default=None
        If given, the split is done separately inside each value of this
        column, so every split keeps (almost exactly) the same class
        proportions as the full dataset.
    random_seed : int, default=RANDOM_SEED
        Seed for the shuffling, so the split is reproducible.

    Returns
    -------
    df_train, df_val, df_test : pandas.DataFrame
        The three splits, each in shuffled row order.

    Explanation
    -----------
    For each group (one group per class when stratifying, or the whole
    dataframe otherwise):
        1. shuffle the row positions
        2. take the first val_size share for validation
        3. take the next test_size share for test
        4. keep the rest for training
    """
    rng = np.random.RandomState(random_seed)

    # One group per class when stratifying, otherwise a single group of all rows.
    if stratify_col is None:
        groups = [np.arange(len(df))]
    else:
        labels = df[stratify_col].to_numpy()
        groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]

    train_idx, val_idx, test_idx = [], [], []

    for positions in groups:
        positions = rng.permutation(positions)

        n_val = int(round(len(positions) * val_size))
        n_test = int(round(len(positions) * test_size))

        val_idx.append(positions[:n_val])
        test_idx.append(positions[n_val:n_val + n_test])
        train_idx.append(positions[n_val + n_test:])

    # Merge the groups and shuffle again, so rows are not ordered by class.
    splits = []
    for idx in (train_idx, val_idx, test_idx):
        merged = rng.permutation(np.concatenate(idx))
        splits.append(df.iloc[merged])

    df_train, df_val, df_test = splits
    return df_train, df_val, df_test


# =========================================================
# 1. CLASSIFICATION PREPROCESSING: Banknote Authentication
# =========================================================
class PreprocessBanknote:
    """
    Handles the full preprocessing pipeline for the Banknote dataset.

    Responsibilities:
    - load the dataset from disk
    - optionally perform basic sanity checks
    - optionally remove duplicate rows
    - split into train / validation / test sets
    - optionally generate EDA plots
    - optionally scale the feature values using training-set statistics only

    Leakage note
    ------------
    To avoid even EDA-related leakage concerns, this version performs the split
    before any scaling-based visualization.

    That means:
    - training uses scaler statistics from X_train only
    - EDA scaling comparisons are also based on X_train only
    """

    def __init__(self):
        """
        Store the path to the Banknote dataset CSV file.
        """
        self.banknote_path = os.path.join(DATASETS_DIR, "banknote_auth.csv")

    def load_and_clean(self):
        """
        Load the Banknote dataset, run sanity checks, and remove duplicates.

        Returns
        -------
        df : pandas.DataFrame
            Cleaned dataset.
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.banknote_path)

        print("\n--- Banknote: Data Cleaning & Sanity Checks ---")

        # Record the shape before duplicate removal
        shape_before = df.shape
        print(f"Shape BEFORE dropping duplicates: {shape_before}")

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Record the shape after duplicate removal
        shape_after = df.shape
        print(f"Shape AFTER dropping duplicates: {shape_after}")

        # Compute how many rows were removed
        rows_removed = shape_before[0] - shape_after[0]
        print(f"Total duplicate rows removed: {rows_removed}")

        # Print missing-value counts
        print("\n--- Banknote: Null Check ---")
        null_counts = df.isnull().sum()
        print(null_counts)
        print(f"Total missing values: {null_counts.sum()}")

        # Print dataset structure
        print("\n--- Banknote: DataFrame Info ---")
        df.info()

        # Print summary statistics
        print("\n--- Banknote: Descriptive Statistics ---")
        print(df.describe())

        return df

    def split_dataframe(self, df):
        """
        Split the Banknote dataframe into train / validation / test.

        Returns
        -------
        df_train, df_val, df_test : pandas.DataFrame
            Dataframe splits in 60/20/20 proportions.

        Notes
        -----
        Stratification is used because this is a classification problem.
        That helps preserve class balance across the splits.
        """
        print("\n--- Banknote: Splitting Data ---")

        # 60% train, 20% validation, 20% test, stratified by class
        return train_val_test_split(
            df,
            val_size=0.2,
            test_size=0.2,
            stratify_col="class",
            random_seed=RANDOM_SEED
        )

    def perform_eda(self, df_train, scale_features=True):
        """
        Generate EDA plots using the training split only.

        Parameters
        ----------
        df_train : pandas.DataFrame
            Training subset of the cleaned Banknote dataset.
        scale_features : bool, default=True
            Whether to include the scaling-comparison plot.
        ----------
        """
        print("\n--- Banknote: Generating EDA Plots (Training Split Only) ---")

        # Separate input features from target using only the training subset
        X_train_unscaled = df_train.drop(columns=["class"])

        # Only show scaling comparison when feature scaling is enabled
        if scale_features:
            # Fit scaler on training features only for the scaling comparison plot
            scaler = StandardScaler()
            X_train_scaled_arr = scaler.fit_transform(X_train_unscaled.values)

            # Compare feature distributions before and after scaling.
            # Save this specifically as the classification scaling figure.
            Visualizer.plot_scaling_comparison(
                X_train_unscaled,
                X_train_scaled_arr,
                filename="classification_scaling_comparison.png",
                label="Classification"
            )
        else:
            print("--- Banknote: Skipping scaling comparison plot because scaling is disabled ---")

        # Show classification-specific EDA on training data only.
        # These plots will now be saved under the preprocessing-graphs folder.
        Visualizer.plot_classification_eda(df_train, target_col="class")

    def get_data(self, show_eda=False, preprocessing_enabled=True, scale_features=True):
        """
        Full data-preparation pipeline for the Banknote dataset.

        Parameters
        ----------
        show_eda : bool, default=False
            Whether to display EDA plots.
        preprocessing_enabled : bool, default=True
            Master switch for preprocessing behavior.
            If False, the dataset is loaded and split, but cleaning, EDA,
            and scaling are skipped.
        scale_features : bool, default=True
            Whether to scale input features using StandardScaler fit on the
            training split only.

        Returns
        -------
        X_train, y_train, X_val, y_val, X_test, y_test : numpy.ndarray
            Dataset splits ready for modeling.
        """
        # ---------------------------------------------------------
        # 1. Load dataset
        # ---------------------------------------------------------
        if preprocessing_enabled:
            # Full current behavior: load, clean, sanity-check, remove duplicates
            df = self.load_and_clean()
        else:
            # Minimal behavior: load raw CSV only
            print("\n--- Banknote: Preprocessing disabled ---")
            print("Loading raw dataset without cleaning, EDA, or scaling.")
            df = pd.read_csv(self.banknote_path)

        # ---------------------------------------------------------
        # 2. Split dataset (always required for the training pipeline)
        # ---------------------------------------------------------
        df_train, df_val, df_test = self.split_dataframe(df)

        # ---------------------------------------------------------
        # 3. EDA display control + EDA generation
        # ---------------------------------------------------------
        Visualizer.SHOW_EDA = show_eda

        if preprocessing_enabled:
            self.perform_eda(df_train, scale_features=scale_features)
        elif show_eda:
            print("\n--- Banknote: EDA skipped because preprocessing is disabled ---")

        # ---------------------------------------------------------
        # 4. Separate features and targets
        # ---------------------------------------------------------
        X_train = df_train.drop(columns=["class"]).values
        y_train = df_train["class"].values.reshape(-1, 1)

        X_val = df_val.drop(columns=["class"]).values
        y_val = df_val["class"].values.reshape(-1, 1)

        X_test = df_test.drop(columns=["class"]).values
        y_test = df_test["class"].values.reshape(-1, 1)

        # ---------------------------------------------------------
        # 5. Optional feature scaling
        # ---------------------------------------------------------
        if preprocessing_enabled and scale_features:
            print("\n--- Banknote: Scaling Data ---")

            # Fit scaler only on training features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            # Apply the same training statistics to validation and test
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        else:
            print("\n--- Banknote: Feature scaling skipped ---")

        # ---------------------------------------------------------
        # 6. Final shape summary
        # ---------------------------------------------------------
        print("\n--- Banknote: Final Dataset Split ---")
        print(f"Train set -> X: {X_train.shape}, y: {y_train.shape}")
        print(f"Validation set -> X: {X_val.shape}, y: {y_val.shape}")
        print(f"Test set -> X: {X_test.shape}, y: {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test


# ====================================================
# 2. REGRESSION PREPROCESSING: Energy Efficiency
# ====================================================
class PreprocessEnergy:
    """
    Handles the full preprocessing pipeline for the Energy dataset.

    Responsibilities:
    - load the dataset from disk
    - optionally perform basic sanity checks
    - optionally remove duplicate rows
    - split into train / validation / test sets
    - optionally generate EDA plots
    - optionally scale the feature values using training-set statistics only

    Notes
    -----
    We use Heating_Load as the regression target.
    Cooling_Load is excluded from the input features.
    """

    def __init__(self):
        """
        Store the path to the Energy dataset CSV file.
        """
        self.energy_path = os.path.join(DATASETS_DIR, "energy_efficiency.csv")

    def load_and_clean(self):
        """
        Load the Energy dataset, run sanity checks, and remove duplicates.

        Returns
        -------
        df : pandas.DataFrame
            Cleaned dataset.
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.energy_path)

        print("\n--- Energy: Data Cleaning & Sanity Checks ---")

        # Record the shape before duplicate removal
        shape_before = df.shape
        print(f"Shape BEFORE dropping duplicates: {shape_before}")

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Record the shape after duplicate removal
        shape_after = df.shape
        print(f"Shape AFTER dropping duplicates: {shape_after}")

        # Compute how many rows were removed
        rows_removed = shape_before[0] - shape_after[0]
        print(f"Total duplicate rows removed: {rows_removed}")

        # Print missing-value counts
        print("\n--- Energy: Null Check ---")
        null_counts = df.isnull().sum()
        print(null_counts)
        print(f"Total missing values: {null_counts.sum()}")

        # Print dataset structure
        print("\n--- Energy: DataFrame Info ---")
        df.info()

        # Print summary statistics
        print("\n--- Energy: Descriptive Statistics ---")
        print(df.describe())

        return df

    def split_dataframe(self, df):
        """
        Split the Energy dataframe into train / validation / test.

        Returns
        -------
        df_train, df_val, df_test : pandas.DataFrame
            Dataframe splits in 60/20/20 proportions.

        Notes
        -----
        This is a regression task, so no stratification is used here.
        """
        print("\n--- Energy: Splitting Data ---")

        # 60% train, 20% validation, 20% test (no stratification for regression)
        return train_val_test_split(
            df,
            val_size=0.2,
            test_size=0.2,
            random_seed=RANDOM_SEED
        )

    def perform_eda(self, df_train, scale_features=True):
        """
        Generate EDA plots using the training split only.

        Parameters
        ----------
        df_train : pandas.DataFrame
            Training subset of the cleaned Energy dataset.
        scale_features : bool, default=True
            Whether to include the scaling-comparison plot.

        Why this version is safer
        -------------------------
        Scaling for the comparison plot is fit only on training features,
        so even the EDA path now avoids mixing future validation/test
        information into scaling statistics.
        """
        print("\n--- Energy: Generating EDA Plots (Training Split Only) ---")

        # Remove both targets so only input features remain
        X_train_unscaled = df_train.drop(columns=["Heating_Load", "Cooling_Load"])

        # Only show scaling comparison when feature scaling is enabled
        if scale_features:
            # Fit scaler only on training features for the scaling comparison plot
            scaler = StandardScaler()
            X_train_scaled_arr = scaler.fit_transform(X_train_unscaled.values)

            # Show feature distributions before and after scaling.
            # Save this specifically as the regression scaling figure.
            Visualizer.plot_scaling_comparison(
                X_train_unscaled,
                X_train_scaled_arr,
                filename="regression_scaling_comparison.png",
                label="Regression"
            )
        else:
            print("--- Energy: Skipping scaling comparison plot because scaling is disabled ---")

        # Show regression-specific EDA using training data only.
        # These plots will now be saved under the preprocessing-graphs folder.
        Visualizer.plot_regression_eda(df_train, target_col="Heating_Load")

    def get_data(self, show_eda=False, preprocessing_enabled=True, scale_features=True):
        """
        Full data-preparation pipeline for the Energy dataset.

        Parameters
        ----------
        show_eda : bool, default=False
            Whether to display EDA plots.
        preprocessing_enabled : bool, default=True
            Master switch for preprocessing behavior.
            If False, the dataset is loaded and split, but cleaning, EDA,
            and scaling are skipped.
        scale_features : bool, default=True
            Whether to scale input features using StandardScaler fit on the
            training split only.

        Returns
        -------
        X_train, y_train, X_val, y_val, X_test, y_test : numpy.ndarray
            Dataset splits ready for modeling.
        """
        # ---------------------------------------------------------
        # 1. Load dataset
        # ---------------------------------------------------------
        if preprocessing_enabled:
            # Full current behavior: load, clean, sanity-check, remove duplicates
            df = self.load_and_clean()
        else:
            # Minimal behavior: load raw CSV only
            print("\n--- Energy: Preprocessing disabled ---")
            print("Loading raw dataset without cleaning, EDA, or scaling.")
            df = pd.read_csv(self.energy_path)

        # ---------------------------------------------------------
        # 2. Split dataset (always required for the training pipeline)
        # ---------------------------------------------------------
        df_train, df_val, df_test = self.split_dataframe(df)

        # ---------------------------------------------------------
        # 3. EDA display control + EDA generation
        # ---------------------------------------------------------
        Visualizer.SHOW_EDA = show_eda

        if preprocessing_enabled:
            self.perform_eda(df_train, scale_features=scale_features)
        elif show_eda:
            print("\n--- Energy: EDA skipped because preprocessing is disabled ---")

        # ---------------------------------------------------------
        # 4. Separate features and targets
        # ---------------------------------------------------------
        X_train = df_train.drop(columns=["Heating_Load", "Cooling_Load"]).values
        y_train = df_train["Heating_Load"].values.reshape(-1, 1)

        X_val = df_val.drop(columns=["Heating_Load", "Cooling_Load"]).values
        y_val = df_val["Heating_Load"].values.reshape(-1, 1)

        X_test = df_test.drop(columns=["Heating_Load", "Cooling_Load"]).values
        y_test = df_test["Heating_Load"].values.reshape(-1, 1)

        # ---------------------------------------------------------
        # 5. Optional feature scaling
        # ---------------------------------------------------------
        if preprocessing_enabled and scale_features:
            print("\n--- Energy: Scaling Data ---")

            # Fit scaler only on training features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            # Apply the same training statistics to validation and test
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        else:
            print("\n--- Energy: Feature scaling skipped ---")

        # ---------------------------------------------------------
        # 6. Final shape summary
        # ---------------------------------------------------------
        print("\n--- Energy: Final Dataset Split ---")
        print(f"Train set -> X: {X_train.shape}, y: {y_train.shape}")
        print(f"Validation set -> X: {X_val.shape}, y: {y_val.shape}")
        print(f"Test set -> X: {X_test.shape}, y: {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test