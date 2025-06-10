import numpy as np
from backtracking import armijo_backtracking

def quasi_newton(x0, f, grad_f, tol=1e-6, max_iter=10000, verbose=None):
	print(f"Running Quasi-Newton with grad-norm-tolerance {tol} and max-iterations {max_iter}")
	xk = x0.copy()
	n = len(xk)
	H = np.eye(n) *2 # initialize Hessian approximation with the identity matrix
	x_k_list = []

	for k in range(1, max_iter+1):
		x_k_list.append(xk.copy())
		# compute new gradient at iterate xk
		gradk = grad_f(xk)
		grad_norm = np.linalg.norm(gradk)
		# stop if tolerance is reached
		if grad_norm <= tol:
			print(f"TERMINATING --- Reached gradient-norm tolerance. Gradient Norm at {grad_norm}.")
			break

		p = -H @ gradk              # compute update direction p
		alphak = armijo_backtracking(f, grad_f, xk, p)

		x_new = xk + alphak * p     # compute new x
		grad_new = grad_f(x_new)    # compute new gradient

		s = x_new - xk              # compute s (difference of new and previous x-value)
		y = grad_new - gradk        # compute y (differnece of new and previous gradient)

		H = bfgs_update(H, s, y, verbose=verbose is not None and k % verbose == 0)    # compute new hessian approximation

		xk = x_new
		if verbose is not None and k % verbose == 0:
			print(f"Iter {k}: f(x) = {f(xk):.6f}, ||grad|| = {np.linalg.norm(gradk):.6f}, alpha={alphak}")
			print(f"grad={grad_new}")
			print("="*50)

		
		if k == max_iter:
			print(f"TERMINATING --- Reached maximum iterations of {k}")
			
	return xk, k, np.linalg.norm(grad_f(xk)), x_k_list


def bfgs_update(Hk, sk, yk, verbose=False):
	rho = 1.0 / (yk.T @ sk)
	
	if np.dot(sk, yk) <= 1e-10:  # use a small threshold instead of exact 0
		if verbose:
			print("Skipping BFGS update: curvature condition violated.")
		return Hk

	I = np.eye(Hk.shape[0])
	Vk = I - rho * sk @ yk.T
	outer_sy = np.outer(sk, yk)
	outer_ys = np.outer(yk, sk)
	outer_ss = np.outer(sk, sk)

	Hk_new = (I - rho * outer_sy) @ Hk @ (I - rho * outer_ys) + rho * outer_ss
	return Hk_new