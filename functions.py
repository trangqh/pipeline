from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns


def calculate_roc_auc(model, X, y):
    """Calculate roc auc score.

    Parameters:
    ===========
    model_pipe: sklearn model or pipeline
    X: features
    y: true target
    """
    y_proba = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_proba)


def _plot_numeric_classes(df, col, bins=10, hist=True, kde=True):
    sns.distplot(df[col], bins=bins, hist=hist, kde=kde)


def _distribution_numeric(df, numeric_cols, row=3, col=3, figsize=(20, 15), bins=10):
    """
    numeric_cols: list các tên cột
    row: số lượng dòng trong lưới đồ thị
    col: số lượng cột trong lưới đồ thị
    figsize: kích thước biểu đồ
    bins: số lượng bins phân chia trong biểu đồ distribution
    """
    print("number of numeric field: ", len(numeric_cols))
    assert row * (col - 1) < len(numeric_cols)
    plt.figure(figsize=figsize)
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5
    )
    for i in range(1, len(numeric_cols) + 1, 1):
        try:
            plt.subplot(row, col, i)
            _plot_numeric_classes(df, numeric_cols[i - 1], bins=bins)
            plt.title(numeric_cols[i - 1])
        except:
            print("Error {}".format(numeric_cols[i - 1]))
            break


def plot_roc_curves(X, y, models, model_names, figsize=(20, 18)):
    """
    Plots ROC curves for a list of models.

    Parameters:
    X (numpy.ndarray or pandas.DataFrame): input features for the models
    y (numpy.ndarray or pandas.DataFrame): target variable
    models (list): list of models to compare
    model_names (list): list of model names to display on the plot
    figsize (tuple): size of the figure to display the plot

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Loop over models and plot ROC curve
    for i, model in enumerate(models):
        y_pred = list(model.predict_proba(X)[:, 1])
        fpr, tpr, threshold = metrics.roc_curve(y, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(
            fpr, tpr, label=(model_names[i] + " AUC = %0.4f" % roc_auc), linewidth=2.0
        )

    ax.grid(False)
    ax.tick_params(length=6, width=2, labelsize=30, grid_color="r", grid_alpha=0.5)
    leg = plt.legend(loc="lower right", prop={"size": 25})
    leg.get_frame().set_edgecolor("b")
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=40)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.ylabel("True Positive Rate", fontsize=30)
    plt.xlabel("False Positive Rate", fontsize=30)
