import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2 * x**2

# Create x and y data for the function curve
x = np.arange(0, 5, 0.001)          # np.arange already returns an array
y = f(x)

# Plot the function
plt.plot(x, y, label='f(x) = 2x²')

colors = ['k', 'g', 'r', 'b', 'c']

def approximate_tangent_line(x, approximate_derivative, b):
    return approximate_derivative * x + b

for i in range(5):
    p2_delta = 0.0001
    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    print(f"Points: ({x1}, {y1}), ({x2}, {y2})")

    approximate_derivative = (y2 - y1) / (x2 - x1)
    b = y2 - approximate_derivative * x2

    # Points to draw the tangent line segment
    to_plot = [x1 - 0.9, x1, x1 + 0.9]

    plt.scatter(x1, y1, c=colors[i])
    plt.plot(to_plot,
             [approximate_tangent_line(pt, approximate_derivative, b) for pt in to_plot],
             c=colors[i])

    print(f"Approximate derivative for f(x) where x = {x1} is {approximate_derivative:.4f}")

# Show everything together
plt.legend()
plt.show()