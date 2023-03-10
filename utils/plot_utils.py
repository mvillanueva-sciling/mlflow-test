import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)

def dir_ceck(plot_file: str) -> None:
    """
    This function creates the directory path of a file before calling the function.
    """
    # Create dir path if it doesn't exist
    dir_path = os.path.dirname(plot_file)
    logger.info(f"Creating dir path: {dir_path}")
    try:
        os.makedirs(dir_path)
    except OSError:
        pass


def create_figure(func):
    """
    This function handles the creation of the figure.
    """
    def wrapper(*args, **kwargs):
        fig, plot_file = func(*args, **kwargs)
        fig.savefig(plot_file)
        plt.close(fig)

    return wrapper

@create_figure
def create_plot_file(y_test_set, y_predicted, plot_file):
    """
    Create a Ground truth vs predicted plot
    """
    plot_file = plot_file.replace('.png', '_ground_truth.png')
    dir_ceck(plot_file)
    fig, ax = plt.subplots()
    ax.scatter(y_test_set, y_predicted, edgecolors=(0, 0, 0))
    ax.plot([y_test_set.min(), y_test_set.max()], [y_test_set.min(), y_test_set.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Ground Truth vs Predicted")

    return fig, plot_file
    

@create_figure
def plot_roc_auc(y_test_set, y_predicted, fpr, tpr, threshold, roc_auc, plot_file):
    """
    Create a roc auc plot
    """
    plot_file = plot_file.replace('.png', '_roc_auc.png')
    fig, ax = plt.subplots()
    ax.title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.xlim([0, 1])
    ax.ylim([0, 1])
    ax.ylabel('True Positive Rate')
    ax.xlabel('False Positive Rate')

    return fig, plot_file
    