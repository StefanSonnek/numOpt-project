import numpy as np

def armijo_backtracking(f, grad_f, x, p, alpha=1.0, rho=0.8, c=1e-1, min_alpha=1e-8, max_iter=20):
	fx = f(x)
	grad_fx = grad_f(x)

	i = 0
	while f(x + alpha * p) > fx + c * alpha * np.dot(grad_fx, p):
		alpha = rho * alpha
		i += 1
		if alpha < min_alpha or i >= max_iter:
			break

	return alpha