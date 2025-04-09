from sklearn.feature_selection import VarianceThreshold
import pandas as pd

data={
    "Feature_x": [1,2,3,4,5],
    "Target_y": [2,3,5,6,8],
    "Feature_XY": [10,11,12,13,14]
}
df=pd.DataFrame(data)
df['Constant_Feature']=[1,1,1,1,1]      
selector= VarianceThreshold(threshold=0.01)
df_reduced= selector.fit_transform(df)
print("Original Feature: ", df.columns)
print("Select Feature from variance Threshhold: \n", df_reduced)