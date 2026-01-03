"""
Lab Utilities - Common functions
Source: WiDS Google Drive - Cost Func files folder
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Load housing dataset for cost function analysis"""
    data = np.array([
        [2104, 3, 399900],
        [1600, 3, 329900],
        [2400, 3, 369000],
        [1416, 2, 232000],
    ])
    return data

def plot_cost(X, y, theta):
    """Plot cost function results"""
    plt.scatter(X, y, alpha=0.5)
    plt.xlabel('Feature')
    plt.ylabel('Price')
    plt.title('Cost Function Analysis')
    return plt
