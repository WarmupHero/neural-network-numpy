import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils import PREPROCESSING_GRAPHS_DIR


class Visualizer:
    """
    Utility class for preprocessing and EDA visualizations.

    All methods are static because this class does not store state.
    It only receives data, creates plots, saves them to disk, and
    optionally displays them depending on SHOW_EDA.
    """

    SHOW_EDA = False

    @staticmethod
    def _ensure_output_dir():
        """
        Create the preprocessing-graphs output directory if it does not exist.
        """
        os.makedirs(PREPROCESSING_GRAPHS_DIR, exist_ok=True)

    @staticmethod
    def plot_scaling_comparison(X_unscaled, X_scaled_arr, filename="scaling_comparison.png", label="Dataset"):
        """
        Create and save side-by-side boxplots showing feature distributions
        before and after scaling.

        Parameters
        ----------
        X_unscaled : pandas.DataFrame
            Original feature values before scaling.
        X_scaled_arr : numpy.ndarray
            Scaled feature values as a NumPy array.
        filename : str
            Name of the output file to save.
        label : str
            Label used in the plot titles, for example:
            "Classification" or "Regression".
        """
        # Make sure the output folder exists before saving.
        Visualizer._ensure_output_dir()
        show = Visualizer.SHOW_EDA

        # Convert scaled array back into a DataFrame so it keeps the same
        # feature names as the unscaled version.
        X_scaled = pd.DataFrame(X_scaled_arr, columns=X_unscaled.columns)

        # Create one figure with two subplots:
        # left = before scaling, right = after scaling
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Plot feature distributions before scaling.
        sns.boxplot(data=X_unscaled, ax=axes[0])
        axes[0].set_title(f"{label}: Feature Distribution Before Scaling")
        axes[0].tick_params(axis="x", rotation=45)
        for tick_label in axes[0].get_xticklabels():
            tick_label.set_horizontalalignment("right")

        # Plot feature distributions after scaling.
        sns.boxplot(data=X_scaled, ax=axes[1])
        axes[1].set_title(f"{label}: Feature Distribution After Scaling")
        axes[1].tick_params(axis="x", rotation=45)
        for tick_label in axes[1].get_xticklabels():
            tick_label.set_horizontalalignment("right")

        # Adjust layout and save figure.
        plt.tight_layout()
        save_path = os.path.join(PREPROCESSING_GRAPHS_DIR, filename)
        plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)

    @staticmethod
    def plot_classification_eda(df, target_col="class"):
        """
        Create and save EDA plots for the classification dataset
        (Banknote Authentication).

        Saved plots
        -----------
        - classification_pairplot.png
        - classification_class_distribution.png
        """
        # Ensure output folder exists.
        Visualizer._ensure_output_dir()
        show = Visualizer.SHOW_EDA

        # Pairplot shows pairwise feature relationships, colored by class.
        pairplot = sns.pairplot(data=df, hue=target_col)
        pairplot.fig.suptitle("Classification: Feature Relationships by Class", y=1.02)
        pairplot.fig.savefig(
            os.path.join(PREPROCESSING_GRAPHS_DIR, "classification_pairplot.png"),
            bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close(pairplot.fig)

        # Countplot shows class balance.
        plt.figure(figsize=(6, 4))
        sns.countplot(x=target_col, data=df)
        plt.title("Classification: Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(
            os.path.join(PREPROCESSING_GRAPHS_DIR, "classification_class_distribution.png"),
            bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close()

    @staticmethod
    def plot_regression_eda(df, target_col="Heating_Load"):
        """
        Create and save EDA plots for the regression dataset
        (Energy Efficiency).

        Saved plots
        -----------
        - regression_target_distribution.png
        - regression_correlation_heatmap.png
        """
        # Ensure output folder exists.
        Visualizer._ensure_output_dir()
        show = Visualizer.SHOW_EDA

        # Histogram + KDE for the regression target variable.
        plt.figure(figsize=(8, 5))
        sns.histplot(df[target_col], kde=True)
        plt.title(f"Regression: Distribution of {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(
            os.path.join(PREPROCESSING_GRAPHS_DIR, "regression_target_distribution.png"),
            bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close()

        # Correlation heatmap for all numeric columns.
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Regression: Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(
            os.path.join(PREPROCESSING_GRAPHS_DIR, "regression_correlation_heatmap.png"),
            bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close()