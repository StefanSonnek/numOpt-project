import numpy as np
from backtracking import armijo_backtracking
from enum import Enum

class QuasiNewtonMethod(Enum):
	BFGS = 1
	DFP = 2

def quasi_newton(x0, f, grad_f, 
				 method: QuasiNewtonMethod=QuasiNewtonMethod.BFGS, 
				 tol=1e-6, 
				 max_iter=10000, 
				 verbose=None):
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

		if method == QuasiNewtonMethod.BFGS:
			H = bfgs_update(H, s, y, verbose=verbose is not None and k % verbose == 0)    # compute new hessian approximation
		elif method == QuasiNewtonMethod.DFP:
			H = dfp_hessian_update(H, s, y, verbose=verbose is not None and k % verbose == 0)    # compute new hessian approximation

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

def dfp_hessian_update(Hk, sk, yk, verbose=False):
	sk_dot_yk = np.dot(sk, yk)

	if sk_dot_yk <= 1e-10:
		if verbose:
			print("Skipping DFP update: curvature condition violated.")
		return Hk
	
	Hk_yk = np.dot(Hk, yk)
	yk_Hk = np.dot(yk, Hk_yk)

	term1 = np.outer(sk, sk) / sk_dot_yk

	# Avoid division by zero in the second term
	if yk_Hk > 1e-12:
		term2 = -np.outer(Hk_yk, Hk_yk) / yk_Hk
		Hk = Hk + term1 + term2
	else: # Fallback to simpler rank-1 update if denominator is zero
		Hk = Hk + term1

	return Hk
	
