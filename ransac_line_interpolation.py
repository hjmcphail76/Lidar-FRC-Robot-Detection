import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression


ransac = RANSACRegressor(
    estimator=LinearRegression(),
    residual_threshold=0.2,
    min_samples=2,
    max_trials=1000,
)


def fit_ransac_line(xs, ys):
    # Convert to numpy arrays and reshape for sklearn
    X = np.array(xs).reshape(-1, 1)
    Y = np.array(ys)

    # Fit line with RANSAC
    ransac.fit(X, Y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = ~inlier_mask

    # Predict line
    line_x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    line_y = ransac.predict(line_x)

    return inlier_mask, outlier_mask, line_x, line_y
