import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)
actual= np.linspace(10,100,20)+np.random.normal(0,5,20)

models={
    "Model A":actual+np.random.normal(0,5,20),
    "Model B":actual+np.random.normal(0,10,20),
    "Model C":actual+np.random.normal(0,2,20)
}

# print("Actual: ",actual)
# print("Model A", models["Model A"])
# print("Model B", models["Model B"])
# print("Model B", models["Model B"])

# Calculate matrics
results = []

for name, pred in models.items():
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
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

plt.figure(figsize=(10,6))
plt.plot(actual,"o-",label="Actual(Y)", color="blue")
plt.plot(pred,"s--",label="Predicted(Y_bar)", color="green")
for i in range(len(actual)):
    plt.plot([i,i],[actual[i],pred[i]], "orange", linestyle="--", linewidth=1)
plt.title(f"Mean Absolute Error(MAE) Visualization\nMAE={mae:.3f}" )
plt.xlabel("Observation Index")
plt.ylabel("Value")
plt.tight_layout()
plt.show()
