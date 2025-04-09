def mtx_trans(mat):
    r=len(mat)
    c=len(mat[0])
    res=[]
    for i in range(c):
        row=[]
        for j in range(r):
            row.append(mat[j][i])
        res.append(row)
    return res

def mat_mul(A, B):
    res=[]
    for i in range(len(A)):
        row=[]
        for j in range(len(B[0])):
            val=0
            for k in range(len(B)):
                val += A[i][k]*B[k][j]
            row.append(val)
        res.append(row)
    return res

# ChatGPT
def mat_inv(mat):
    sz=len(mat)
    iden=[]
    for i in range(sz):
        row=[]
        for j in range(sz):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        iden.append(row)
    for i in range(sz):
        d=mat[i][i]
        for j in range(sz):
            mat[i][j]=mat[i][j]/d
            iden[i][j]=iden[i][j]/d
        for k in range(sz):
            if k != i:
                factor=mat[k][i]
                for j in range(sz):
                    mat[k][j] -= factor*mat[i][j]
                    iden[k][j] -= factor*iden[i][j]
    return iden

def lin_reg(X, Y):
    X1=[]
    for row in X:
        new_row=[1]
        for val in row:
            new_row.append(val)
        X1.append(new_row)
    Xt=mtx_trans(X1)
    XtX=mat_mul(Xt, X1)
    XtX_inv=mat_inv(XtX)
    Y1=[]
    for val in Y:
        Y1.append([val])
    XtY=mat_mul(Xt, Y1)
    B=mat_mul(XtX_inv, XtY)
    res=[]
    for row in B:
        res.append(row[0])
    return res

def predict(X, beta):
    X1=[]
    for row in X:
        new_row=[1]
        for val in row:
            new_row.append(val)
        X1.append(new_row)
    res=mat_mul(X1, [[b] for b in beta])
    y_pred=[]
    for row in res:
        y_pred.append(row[0])
    return y_pred

def MSE(y_true, y_pred):
    err=0
    for i in range(len(y_true)):
        err += int((y_true[i] - y_pred[i])**2)
    return err/len(y_true)

def r2(y_true, y_pred):
    sm=0
    for i in range(len(y_true)):
        sm += y_true[i]
    mean_y=sm/len(y_true)
    num=0
    den=0
    for i in range(len(y_true)):
        num += (y_true[i] - y_pred[i])**2
        den += (y_true[i] - mean_y)**2
    return 1 - (num/den)

def main():
    n=int(input("Enter number of data points: "))
    m=int(input("Enter number of independent variables: "))
    X=[]
    for i in range(n):
        row=[]
        for j in range(m):
            row.append(float(input(f"Enter x[{i+1}][{j+1}]: ")))
        X.append(row)
    Y=[]
    for i in range(n):
        Y.append(float(input(f"Enter y[{i+1}]: ")))
    beta=lin_reg(X, Y)
    y_pred=predict(X, beta)
    mse=MSE(Y, y_pred)
    R2=r2(Y, y_pred)
    print("\n--- Linear Regression Results ---")
    print(f"Coefficients (Beta): {beta}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {R2}")

if __name__ == "__main__":
    main()
