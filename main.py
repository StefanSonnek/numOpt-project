from utils import initialize_starting_points, save_plots, create_latex_table, initliazie_random_starting_points
from functions import f, grad, l, a, b
from quasiNewton import quasi_newton, QuasiNewtonMethod
import numpy as np
import time
import pandas as pd
from tabulate import tabulate

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

		#########################
		# Calculate l_k and q_k #
		#########################	
		# Compute norms to x_bar
		norms = [np.linalg.norm(x - x_solution) for x in x_k_list]

		# Get last 5 indices
		k_vals = list(range(len(x_k_list) - 10, len(x_k_list)))

		data = {
			"Iter k": k_vals,
			"||x_k - x̄||": [norms[k] for k in k_vals],
			"l_k": [],
			"q_k": []
		}

		last_k = 5
		for k in k_vals:
			if k != k_vals[-1]:
				nominator = np.linalg.norm(x_k_list[k+1] - x_solution)
				denominator = np.linalg.norm(x_k_list[k] - x_solution)
				
				l_k = nominator / denominator if denominator != 0 else np.nan
				q_k = nominator / denominator**2 if denominator != 0 else np.nan

				data["l_k"].append(l_k)
				data["q_k"].append(q_k)
			else:
				data["l_k"].append("(last step)")
				data["q_k"].append("(last step)")
			
		# Format the DataFrame
		df = pd.DataFrame(data)
		df["||x_k - x̄||"] = df["||x_k - x̄||"].map("{:.6e}".format)
		df.loc[df["l_k"] != "(last step)", "l_k"] = df.loc[df["l_k"] != "(last step)", "l_k"].map("{:.3e}".format)
		df.loc[df["q_k"] != "(last step)", "q_k"] = df.loc[df["q_k"] != "(last step)", "q_k"].map("{:.3e}".format)

		print(tabulate(df, headers='keys', tablefmt='psql'))


		# store everything in the results
		results.append({
			"run": i,
			"method_name": method_name,
			"x_solution": x_solution,
			"iterations": iters,
			"grad_norm": grad_norm,
			"x_k_list": x_k_list,
			"time": toc-tic,
			"distance_to_target_solution": distance_to_target_solution,
			"x0": x0,
			"l_k": data["l_k"][-2],
			"q_k": data["q_k"][-2]
		})

	

	return results

if __name__ == "__main__":
	optimizer = lambda x: quasi_newton(x, f=f ,grad_f=grad, 
									 method=QuasiNewtonMethod.BFGS,
									 tol=1e-6, 
									 max_iter=10000, 
									 verbose=None)
	
	starts, target_solution = initialize_starting_points(l)
	# starts, target_solution = initliazie_random_starting_points(10, l)
	
	method_name = "BFGS"
	results = start_runs(starts, optimizer=optimizer, f=f, target_solution=target_solution, method_name=method_name)
	save_plots(results, method_name=method_name)
	create_latex_table(results)