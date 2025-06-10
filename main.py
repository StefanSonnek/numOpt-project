from utils import initialize_starting_points, save_plots, create_latex_table
from functions import f, grad, l, a, b
from bfgs import quasi_newton
import numpy as np
import time

def start_runs(x0_list, optimizer, f, target_solution, method_name):
	results = []
	print(f"STARTING RUNS FOR METHOD '{method_name}'")
	for i, x0 in enumerate(x0_list, 1):
		print("="*80)
		print(f"RUNNING {method_name} FOR INITIAL GUESS #{i}")
		print(f"Initial x0 for Guess #{i}:")
		print(x0)

		tic = time.perf_counter()
		x_solution, iters, grad_norm, x_k_list = optimizer(x0)
		toc = time.perf_counter()
		print(f"{method_name} for Guess #{i} finisehd after {iters} iterations in {toc - tic:0.4f} seconds")

		print()
		print("Found solution x_bar:")
		print(x_solution)
		print(f"Final Loss: {f(x_solution)}")

		distance_to_target_solution = np.linalg.norm(x_solution - target_solution)
		print(f"Distance x_bar to x*: {distance_to_target_solution}")
		print(f"Final gradient norm: {grad_norm}")

		results.append({
			"run": i,
			"method_name": method_name,
			"x_solution": x_solution,
			"iterations": iters,
			"grad_norm": grad_norm,
			"x_k_list": x_k_list,
			"time": toc-tic,
			"distance_to_target_solution": distance_to_target_solution
		})

	save_plots(results, method_name)

	return results

if __name__ == "__main__":
	optimizer = lambda x: quasi_newton(x, f=f ,grad_f=grad, tol=1e-6, max_iter=10000, verbose=None)
	starts, target_solution = initialize_starting_points(l)
	
	results = start_runs(starts, optimizer=optimizer, f=f, target_solution=target_solution, method_name="BFGS")
	create_latex_table(results)