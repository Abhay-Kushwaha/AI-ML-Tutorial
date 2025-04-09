def mean(A):
    n=len(A)
    A_mean=sum(A)/n
    return A_mean

# Variance
# def R2(X):
#     X_mean=mean(X)
#     r_sq=[]
#     for i in range(len(X)):
#         r_sq.append((X[i]-X_mean)**2)
#     return sum(r_sq)/len(X)

# R square
def SS_res(Y,Y_bar):
    val=0
    for i in range(len(Y)):
        val=val+((Y[i]-Y_bar[i])**2)
    return val

def SS_total(Y):
    y_mean=mean(Y)
    total=[]
    for i in range(len(Y)):
        total.append((Y[i]-y_mean)**2)
    return sum(total)

X=[1,2,3,4,5]
Y=[3,5,7,9,11]
Y_bar=[2.8,4.9,7.1,9.2,10.8]
print(f"Observation= {X}")
print(f"Actual= {Y}")
print(f"Predicted Y bar= {Y_bar}")
print("R square: ", 1-(SS_res(Y,Y_bar)/SS_total(Y)))