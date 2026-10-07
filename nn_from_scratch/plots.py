"""
Exploratory data analysis (EDA) plots for the preprocessing step.

Input: pandas DataFrames (and one scaled NumPy array) passed in by
nn_from_scratch.features, always taken from the training split.

Output: PNG files in ``reports/figures/eda/``, each with the run
stamp in its name, optionally also shown on screen.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from nn_from_scratch.config import PREPROCESSING_GRAPHS_DIR, stamped_filename


class Visualizer:
    """
    Utility class for preprocessing and EDA visualizations.

    All methods are static because this class does not store per-instance
    state. It only receives data, creates plots, saves them to disk, and
    optionally displays them depending on SHOW_EDA.

    Every saved file name gets the run stamp inserted before its
    extension (see nn_from_scratch.config.stamped_filename), so plots from earlier
    runs are never overwritten.

    Attributes
    ----------
    SHOW_EDA : bool, default=False
        Class-level switch. When True, every plot is also displayed with
        ``plt.show()`` after it is saved. The preprocessing classes set it
        from their ``show_eda`` argument.
    """

    SHOW_EDA = False

    @staticmethod
    def _ensure_output_dir():
        """
        Create the preprocessing-graphs output directory if it does not exist.

        Parameters
        ----------
        None
            Uses the module-level constant ``PREPROCESSING_GRAPHS_DIR``.

        Returns
        -------
        None
            Creates ``reports/figures/eda/`` on disk if missing.

        Notes
        -----
        Processing:
        1. Call ``os.makedirs`` with ``exist_ok=True``, so an existing
           folder is not an error.
        """
        os.makedirs(PREPROCESSING_GRAPHS_DIR, exist_ok=True)

    @staticmethod
    def plot_scaling_comparison(X_unscaled, X_scaled_arr, filename="scaling_comparison.png", label="Dataset"):
        """
        Create and save side-by-side boxplots showing feature distributions
        before and after scaling.

        Parameters
        ----------
        X_unscaled : pandas.DataFrame of shape (n_samples, n_features)
            Original feature values before scaling. Its column names are
            used as the boxplot labels.
        X_scaled_arr : numpy.ndarray of shape (n_samples, n_features), dtype float64
            The same features after standard scaling.
        filename : str, default="scaling_comparison.png"
            Name of the output file to save (the run stamp is added).
        label : str, default="Dataset"
            Label used in the plot titles, for example:
            "Classification" or "Regression".

        Returns
        -------
        None
            Saves the figure to
            ``reports/figures/eda/<filename stem>_<stamp>.png``,
            shows it if ``Visualizer.SHOW_EDA`` is True, then closes it.

        Notes
        -----
        Processing:
        1. Make sure the output folder exists.
        2. Wrap ``X_scaled_arr`` in a DataFrame with the same column names
           as ``X_unscaled``.
        3. Draw one boxplot per feature before scaling (left) and after
           scaling (right), with x labels rotated 45 degrees.
        4. Save, optionally show, and close the figure.

        After standard scaling every box should be centred near 0 with a
        similar spread, which is what the right-hand panel lets you check.
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
        save_path = os.path.join(PREPROCESSING_GRAPHS_DIR, stamped_filename(filename))
        plt.savefig(save_path, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)

    @staticmethod
    def plot_classification_eda(df, target_col="class"):
        """
        Create and save EDA plots for the classification dataset
        (Banknote Authentication).

        Parameters
        ----------
        df : pandas.DataFrame of shape (n_samples, n_features + 1)
            Training split with the feature columns and the target column.
        target_col : str, default="class"
            Name of the class-label column, used for colouring the pairplot
            and for the class counts.

        Returns
        -------
        None
            Saves two files to ``reports/figures/eda/``:
            ``classification_pairplot_<stamp>.png`` and
            ``classification_class_distribution_<stamp>.png``. Each is shown
            if ``Visualizer.SHOW_EDA`` is True and then closed.

        Notes
        -----
        Processing:
        1. Make sure the output folder exists.
        2. Draw a seaborn pairplot: a scatter plot for every pair of
           features (and a distribution on the diagonal), coloured by class.
           It shows how well the classes separate.
        3. Draw a count plot of the target column, which shows the class
           balance.
        """
        # Ensure output folder exists.
        Visualizer._ensure_output_dir()
        show = Visualizer.SHOW_EDA

        # Pairplot shows pairwise feature relationships, colored by class.
        pairplot = sns.pairplot(data=df, hue=target_col)
        pairplot.fig.suptitle("Classification: Feature Relationships by Class", y=1.02)
        pairplot.fig.savefig(
            os.path.join(PREPROCESSING_GRAPHS_DIR, stamped_filename("classification_pairplot.png")),
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
            os.path.join(PREPROCESSING_GRAPHS_DIR, stamped_filename("classification_class_distribution.png")),
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

        Parameters
        ----------
        df : pandas.DataFrame of shape (n_samples, n_columns)
            Training split with the feature columns and the target
            column(s).
        target_col : str, default="Heating_Load"
            Name of the regression target whose distribution is plotted.

        Returns
        -------
        None
            Saves two files to ``reports/figures/eda/``:
            ``regression_target_distribution_<stamp>.png`` and
            ``regression_correlation_heatmap_<stamp>.png``. Each is shown if
            ``Visualizer.SHOW_EDA`` is True and then closed.

        Notes
        -----
        Processing:
        1. Make sure the output folder exists.
        2. Draw a histogram of the target column with a kernel density
           estimate (KDE) curve on top.
        3. Compute the Pearson correlation matrix of all numeric columns
           (including both targets if present) and draw it as an annotated
           heatmap with two decimals.
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
            os.path.join(PREPROCESSING_GRAPHS_DIR, stamped_filename("regression_target_distribution.png")),
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
            os.path.join(PREPROCESSING_GRAPHS_DIR, stamped_filename("regression_correlation_heatmap.png")),
            bbox_inches="tight"
        )

        if show:
            plt.show()

        plt.close()