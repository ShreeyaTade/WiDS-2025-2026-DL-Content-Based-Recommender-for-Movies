"""
Gradient Descent for Linear Regression
Source: WiDS Google Drive - Gradient Descent files folder
"""

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, theta, alpha=0.01, num_iters=100):
    """
    Perform gradient descent to find optimal parameters
    
    Args:
        X: Feature matrix
        y: Target values
        theta: Initial parameters
        alpha: Learning rate
        num_iters: Number of iterations
    
    Returns:
        theta: Optimized parameters
        J_history: Cost history
    """
    m = len(y)
    J_history = []
    
    for i in range(num_iters):
        h = X.dot(theta)
        error = h - y
        theta = theta - (alpha / m) * X.T.dot(error)
        
        cost = np.sum(error ** 2) / (2 * m)
        J_history.append(cost)
    
    return theta, J_history

def compute_cost(X, y, theta):
    """Compute the cost function"""
    m = len(y)
    h = X.dot(theta)
    cost = np.sum((h - y) ** 2) / (2 * m)
    return cost
