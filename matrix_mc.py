import matplotlib.pyplot as plt

def mtx_transpose(mtx):
    rows= len(mtx)
    cols= len(mtx[0])
    result= []
    for i in range(cols):
        row= []
        for j in range(rows):
            row.append(mtx[j][i])
        result.append(row)
    return result

def mtx_multiply(A, B):
    result= []
    for i in range(len(A)):
        row= []
        for j in range(len(B[0])):
            value= 0
            for k in range(len(B)):
                value += A[i][k]*B[k][j]
            row.append(value)
        result.append(row)
    return result

def mtx_inverse(mtx):
    size= len(mtx)
    mtx_copy= [row[:] for row in mtx]  
    iden= []
    for i in range(size):
        row= []
        for j in range(size):
            if i==j:
                row.append(1)
            else:
                row.append(0)
        iden.append(row)
    for i in range(size):
        if mtx_copy[i][i]==0:  
            for k in range(i+1, size):
                if mtx_copy[k][i] != 0:
                    mtx_copy[i], mtx_copy[k] = mtx_copy[k], mtx_copy[i]
                    iden[i], iden[k] = iden[k], iden[i]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted.")
        diag = mtx_copy[i][i]
        for j in range(size):
            mtx_copy[i][j] /= diag
            iden[i][j] /= diag
        for k in range(size):
            if k != i:
                factor= mtx_copy[k][i]
                for j in range(size):
                    mtx_copy[k][j] -= factor*mtx_copy[i][j]
                    iden[k][j] -= factor * iden[i][j]
    return iden

def mc_mtx(x, y):
    X1 = []
    for row in x:
        new_row = [1]
        for val in row:
            new_row.append(val)
        X1.append(new_row)
    Xt = mtx_transpose(X1)
    XtX = mtx_multiply(Xt, X1)
    XtX_inv = mtx_inverse(XtX)
    Y = []
    for val in y:
        Y.append([val])
    XtY = mtx_multiply(Xt, Y)
    B = mtx_multiply(XtX_inv, XtY)
    return B[0][0], B[1][0]

def predict(x, m, c):
    y_pred=[]
    for xi in x:
        val= m * xi + c
        y_pred.append(val)
    return y_pred

def MSE(y_true, y_pre, n):
    total_sq_error=0
    for i in range(n):
        total_sq_error += int((y_true[i] - y_pre[i]) ** 2)
    return total_sq_error / n

def R2(y_true, y_pre, n):
    mean_y = sum(y_true) / n
    numerator, denominator=0,0
    for i in range(n):
        numerator += (y_true[i] - y_pre[i]) ** 2
        denominator += (y_true[i] - mean_y) ** 2
    return 1 - (numerator / denominator)

def graph(x, y_true, y_pre):
    for i in range(len(x)):
        plt.scatter(x[i], y_true[i], color='blue', label="Actual Y" if i == 0 else "")
        plt.scatter(x[i], y_pre[i], color='red', label="Predicted Y" if i == 0 else "")
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
    x,y=[],[]
    for i in range(n):
        a=float(input(f"Enter x[{i+1}]: "))
        x.append(a)
    for i in range(n):
        b=float(input(f"Enter y[{i+1}]: "))
        y.append(b)
    c, m = mc_mtx(x, y)
    y_pre = predict(x, m, c)
    mse = MSE(y, y_pre, n)
    r2 = R2(y, y_pre, n)
    print("\n--- Linear Regression Results ---")
    print(f"Slope (m): {m}")
    print(f"Intercept (c): {c}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")
    graph(x, y, y_pre)

if __name__ == "__main__":
    main()
