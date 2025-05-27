import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
x = np.arange(start=0,stop=10,step=0.2)
np.random.seed(42)
noise = np.random.rand(50) * 3
y = x*2+ 3 + noise
plt.figure(figsize=(15,10))
def LinearRegression(epochs,lr):
    weight = 5
    bias = 5
    for i in range(epochs):
        der_weight = 0
        der_bias = 0
        y_pred = (weight*x) + bias
        der_weight = -2 * np.sum(x * (y-y_pred))
        der_bias = -2 * np.sum(y-y_pred)
        der_weight /= len(x)
        der_bias /= len(x)  
        weight = weight - (lr*der_weight)   
        bias = bias - (lr*der_bias) 
    mean_dev = y - np.mean(y)
    mean_dev = np.sum(mean_dev ** 2)
    error = np.sum((y-((weight*x)+bias))**2)
    r2 = 1 - (error/mean_dev)
    return [weight,bias,error,r2]
linear = LinearRegression(epochs=1000,lr=0.01)
print(f"Weight : {linear[0]}\nBias : {linear[1]}\nMean Squared Error : {linear[2]/len(x)}\nR2 Score : {linear[3]}")
gs = gridspec.GridSpec(2,2)
plt.subplot(gs[:,0])
plt.title("Linear Regression")
plt.scatter(x,y)
plt.plot(x,linear[0]*x + linear[1],c="red")

plt.subplot(gs[0,1])
plt.title("Weight vs Error")
weights = np.arange(start=linear[0]-0.5,stop=linear[0]+0.5,step=0.01)
errors = np.array([np.sum((y-((weight*x)+linear[1]))**2) for weight in weights])
plt.plot(weights,errors/len(x))
plt.scatter(linear[0],np.sum((y-((linear[0]*x)+linear[1]))**2)/len(x),c="red",s=100)
plt.ylim(0,10)

plt.subplot(gs[1,1])
plt.title("Bias vs Error")
biases = np.arange(start=linear[1]-0.5,stop=linear[1]+0.5,step=0.01)
errors = np.array([np.sum((y-((linear[0]*x)+bias))**2) for bias in biases])
plt.ylim(0.7,1.2)

plt.plot(biases,errors/len(x))
plt.scatter(linear[1],np.sum((y-((linear[0]*x)+linear[1]))**2)/len(x),c="red",s=100)

# plt.tight_layout()
plt.show()