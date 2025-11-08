import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# Example: random points simulating LiDAR wall + noise
np.random.seed(42)
x = np.linspace(-5, 5, 200)
y = 0.5 * x + 2 + np.random.normal(0, 0.2, size=x.shape)
# Add some random outliers
x = np.concatenate([x, np.random.uniform(-5, 5, 50)])
y = np.concatenate([y, np.random.uniform(-5, 5, 50)])


# Reshape for sklearn
X = x.reshape(-1, 1)

# Fit line with RANSAC
ransac = RANSACRegressor(
    estimator=LinearRegression(),
    residual_threshold=0.2,
    min_samples=2,
    max_trials=1000,
)

ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = ~inlier_mask

# Predict line
line_x = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
line_y = ransac.predict(line_x)

# Plot result
plt.figure(figsize=(8, 6))
plt.scatter(x[inlier_mask], y[inlier_mask], color="green", label="Inliers")
plt.scatter(x[outlier_mask], y[outlier_mask], color="red", label="Outliers")
plt.plot(line_x, line_y, color="blue", linewidth=2, label="RANSAC Line")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("RANSAC Line Detection from random Points")
plt.show()
