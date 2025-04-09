import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean(A):
    n=len(A)
    A_mean=sum(A)/n
    return A_mean

# R square
# def SS_res(Y,Y_bar):
#     val=0
#     for i in range(len(Y)):
#         val=val+((Y[i]-Y_bar[i])**2)
#     return val

# def SS_total(Y):
#     y_mean=mean(Y)
#     total=[]
#     for i in range(len(Y)):
#         total.append((Y[i]-y_mean)**2)
#     return sum(total)

X=[1,2,3,4,5]
Y=[3,5,7,9,11]
Y_bar=[2.8,4.9,7.1,9.2,10.8]
models={
    "X": [1,2,3,4,5],
    "Y": [3,5,7,9,11],
    "Y_bar": [2.8,4.9,7.1,9.2,10.8]
}
df= pd.DataFrame(models)
print(df)

results = []
for name, pred in models.items():
    mse = mean_squared_error(df[Y], df[Y_pred])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df[Y], df[Y_pred])
    r2 = r2_score(df[Y], df[Y_pred])
    results.append({
        "Model": name,
        "R² Score": round(r2, 3),
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "MSE": round(mse, 3)
    })

# Convert to DataFrame
df_results = pd.DataFrame(results)
print(df_results)