import numpy as np

l = 4
m = 100
a = np.linspace(-2*np.pi, 2*np.pi, m)
b = np.sin(a)

def phi_vectorized(x, t):
	"""
		Parameters:
		x: parameter vector
		t: values between -2pi and 2pi
		l: number of different mu, alpha and sigmas
		
	"""
	alphas = x[::3]
	mus = x[1::3]
	sigmas = x[2::3]
	
	t = np.atleast_1d(t)
	t = t.reshape(-1, 1)

	result = alphas * np.exp(-((t - mus)**2) / (2 * sigmas**2))

	return np.sum(result, axis=1)


def f(x):
	result = (phi_vectorized(x, a) - b)**2

	return 0.5 * np.sum(result)

def grad(x):
	res = phi_vectorized(x, a) - b
	grad = np.zeros_like(x)

	for i in range(l):
		alpha = x[i*3]
		mu = x[i*3 + 1]
		sigma = x[i*3 + 2]

		gauss =  np.exp(-((a - mu)**2 / (2 * sigma**2)))
		d_alpha =  gauss
		d_mu = alpha * gauss * (a - mu) / (sigma**2)
		d_sigma = alpha * gauss * (a - mu)**2 / (sigma**3)
		
		grad[i*3] = np.sum(res * d_alpha)
		grad[i*3 + 1] = np.sum(res * d_mu)
		grad[i*3 + 2] = np.sum(res * d_sigma)

	return grad