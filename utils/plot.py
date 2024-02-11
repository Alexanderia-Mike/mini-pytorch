import numpy as np
import matplotlib.pyplot as plt

def plot_loss(filename:str, logs):
    """
    Function to plot training and validation/test loss curves
    :param logs: dict with keys 'train_loss','test_loss' and 'epochs', where train_loss and test_loss are lists with 
    			the training and test/validation loss for each epoch
    """
    plt.clf()
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    t = np.arange(len(logs['train_loss']))
    plt.plot(t, logs['train_loss'], label='train_loss', lw=3)
    plt.plot(t, logs['test_loss'], label='test_loss', lw=3)
    plt.grid(1)
    plt.xlabel('epochs',fontsize=15)
    plt.ylabel('loss value',fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(filename)


def plot_decision_boundary(filename:str, X, y, pred_fn, boundry_level=None):
    """
    Plots the decision boundary for the model prediction
    :param X: input data
    :param y: true labels
    :param pred_fn: prediction function,  which use the current model to predict. i.e. y_pred = pred_fn(X)
    :boundry_level: Determines the number and positions of the contour lines / regions.
    :return:
    """
    plt.clf()
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = pred_fn(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.7, levels=boundry_level, cmap='viridis_r')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), alpha=0.7,s=50, cmap='viridis_r',)
    plt.savefig(filename)