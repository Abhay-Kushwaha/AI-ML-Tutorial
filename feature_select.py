import pandas as pd
from scipy.stats import pearsonr
data={
    "Feature_x": [1,2,3,4,5],
    "Target_y": [2,3,5,6,8],
    "Feature_XY": [10,11,12,13,14]
}
df=pd.DataFrame(data)

# Pearson
corr= df.corr(method="pearson")
print("Pearson Correlation: \n", corr)
# Spearman
corr2= df.corr(method="spearman")
print("Spearman Correlation: \n", corr2)
# Kendall
corr3= df.corr(method="kendall")
print("Kendall Correlation: \n", corr3)
print()

# r_value, p_value= pearsonr(df['Feature_x'],df['Target_y'])
r_value, p_value= pearsonr(df['Feature_XY'],df['Target_y'])
print(f"Pearson Correlation: {r_value}")
print(f"Pearson Value: {p_value}")
if(p_value>0.05):
    print("NOT Statically Significant (> 0.05)")
else:
    print("Statically Significant (< 0.05)")