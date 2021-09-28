#imports
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import matplotlib.pyplot as plt
from sklearn.utils import resample

#----------------------------------
#define the function and design matrix

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X
#--------------------------------------------------------

np.random.seed(3155)

maxdegree = 7
error = np.zeros(maxdegree); bias = np.zeros(maxdegree); variance = np.zeros(maxdegree)
error_train = np.zeros(maxdegree); bias_train = np.zeros(maxdegree); variance_train = np.zeros(maxdegree) #train
polydegree = np.zeros(maxdegree); MSE_Predict = np.zeros(maxdegree)

N = 20
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
z = FrankeFunction(x, y)

n_boostraps = 100

for degree in range(maxdegree):

    X = create_X(x, y, n=degree)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    olsfit = skl.LinearRegression().fit(X_train, z_train)
    z_pred = np.empty((z_test.shape[0], n_boostraps))
    z_pred_train = np.empty((z_train.shape[0], n_boostraps))

    for i in range(n_boostraps):

        x_, z_ = resample(X_train, z_train)
        z_pred[:, i] = olsfit.fit(x_, z_).predict(X_test).ravel()

        z_pred_train[:, i] = olsfit.fit(x_, z_).predict(X_train).ravel()

    polydegree[degree] = degree

    error[degree] = np.mean( np.mean((z_test.reshape(-1,1) - z_pred)**2, axis=1, keepdims=True) )
    bias[degree] = np.mean( (z_test.reshape(-1,1) - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )
    MSE_Predict[degree] = bias[degree] + variance[degree] + error[degree]

plt.plot(polydegree, error, 'b--', linewidth=10, label='Error')
plt.plot(polydegree, bias, label='Bias')
plt.plot(polydegree, variance, label='Variance')
plt.plot(polydegree, MSE_Predict, label='MSE Predict')
plt.xlabel('Model complexity'); plt.title('Bias-Variance-tradeoff')
plt.legend(); plt.show()

#bias and variance plots
plt.plot(polydegree, variance, 'r--', label='Variance'); plt.legend(); plt.show()
plt.plot(polydegree, bias, label='Bias'); plt.legend(); plt.show()
