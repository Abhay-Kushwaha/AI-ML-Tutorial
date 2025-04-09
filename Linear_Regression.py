import matplotlib.pyplot as plt

def summation(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = 0
    sum_x2 = 0
    for i in range(len(x)):
        sum_xy += x[i] * y[i]
        sum_x2 += x[i] ** 2 
    return sum_x, sum_y, sum_xy, sum_x2

def Mean(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    return mean_x, mean_y

def M_C(x, y, n):
    sum_x, sum_y, sum_xy, sum_x2 = summation(x, y)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    mean_x, mean_y = Mean(x, y)
    c = mean_y - (m * mean_x)
    return m, c

def predict(x, m, c):
    y_pre = []
    for xi in x:
        y = (m * xi) + c
        y_pre.append(y)
    return y_pre

def MSE(y_true, y_pre, n):
    total_sq_error = 0
    for i in range(n):
        actual = y_true[i]
        pred = y_pre[i]
        total_sq_error += int((actual - pred)**2)
    mse = total_sq_error / n
    return mse

def R2(y_true, y_pre, n):
    mean_y = sum(y_true) / n
    numerator = 0
    denominator = 0
    for i in range(n):
        actual = y_true[i]
        pred = y_pre[i]
        numerator += (actual - mean_y) ** 2
        denominator += (actual - pred) ** 2
    r2 = 1 - (denominator / numerator)
    return r2

def graph(x, y_true, y_pre):
    plt.figure(figsize=(8, 6))
    for i in range(len(x)):
        plt.scatter(x[i], y_true[i], color='blue', label="Actual" if i == 0 else "")
        plt.scatter(x[i], y_pre[i], color='red', label="Predicted" if i == 0 else "")
        plt.plot([x[i], x[i]], [y_true[i], y_pre[i]], color='gray', linestyle='dashed')
    plt.plot(x, y_pre, color='green', label="Regression Line")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Linear Regression Plot")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    n = int(input("Enter number of data points: "))
    x = []
    y = []
    for i in range(n):
        xi = float(input(f"Enter x[{i+1}]: "))
        x.append(xi)
        yi = float(input(f"Enter y[{i+1}]: "))
        y.append(yi)
    m, c = M_C(x, y, n)
    y_pre = predict(x, m, c)
    mse = MSE(y, y_pre, n)
    r2 = R2(y, y_pre, n)
    
    print("\n--- Linear Regression Results ---")
    print(f"Slope (m): {m}")
    print(f"Intercept (c): {c}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")
    graph(x, y, y_pre)

# Run the program
if __name__ == "__main__":
    main()
