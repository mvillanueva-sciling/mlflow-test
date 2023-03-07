import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)


def create_plot_file(y_test_set, y_predicted, plot_file):

    # Create dir path if it doesn't exist
    dir_path = os.path.dirname(plot_file)
    logger.info(f"Creating dir path: {dir_path}")
    try:
        os.makedirs(dir_path)
    except OSError:
        logger.error(f"Creation of the directory {dir_path} failed")

    global image
    fig, ax = plt.subplots()
    ax.scatter(y_test_set, y_predicted, edgecolors=(0, 0, 0))
    ax.plot([y_test_set.min(), y_test_set.max()], [y_test_set.min(), y_test_set.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Ground Truth vs Predicted")

    image = fig
    fig.savefig(plot_file)
    plt.close(fig)

def plot_roc_auc(y_test_set, y_predicted, fpr, tpr, threshold, roc_auc, plot_file):
    """
    Save a roc auc plot
    """
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()