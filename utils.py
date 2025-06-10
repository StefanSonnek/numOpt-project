import numpy as np
from functions import f, phi_vectorized, a, b
import os
import datetime
import matplotlib.pyplot as plt

def smart_init(l):
	x0 = np.zeros(3 * l)
	centers = np.linspace(-2 * np.pi, 2 * np.pi, l)
	for i in range(l):
		x0[3*i] = 1.0                     # α_i
		x0[3*i + 1] = centers[i]          # μ_i
		x0[3*i + 2] = 1                 # σ_i (positive)
	return x0


def initialize_starting_points(l):
	x0_1 = np.array([
		1.0, -1.5 * np.pi, 1.5,
		-1.0, -0.5 * np.pi, 1.5,
		1.0, 0.5 * np.pi, 1.5,
		-1.0, 1.5 * np.pi, 1.5
	])

	x0_2 = x0_1 + 0.25 * np.random.randn(*x0_1.shape)

	for i in range(1, l + 1):
		idx = 3 * i - 1  # zero-based index
		if x0_2[idx] <= 0.1:
			x0_2[idx] = 0.5

	mu_vals = np.linspace(-1.9 * np.pi, 1.9 * np.pi, l)

	x0_3 = np.zeros(3 * l)
	for i in range(l):
		x0_3[3 * i] = 0.8
		x0_3[3 * i + 1] = mu_vals[i]
		x0_3[3 * i + 2] = 1.0

	x0_4 = np.array([
		1.1, -2.0 * np.pi, 0.7,
		-0.9, -0.2 * np.pi, 1.3,
		0.9, 0.3 * np.pi, 1.2,
		-1.1, 1.8 * np.pi, 0.8
	])

	x0_5 = np.array([
		1.5, -1.0 * np.pi, 2.0,
		-1.5, -0.8 * np.pi, 0.6,
		1.3, 1.0 * np.pi, 1.8,
		-1.3, 1.2 * np.pi, 0.7
	])

	x0_6 = smart_init(l)

	starts = [x0_1, x0_2, x0_3, x0_4, x0_5, x0_6]
	
	x_target = np.array([
		-3.47556557e+01,  2.55102404e-01, -1.97634873e+00,
		-4.35404002e+00, -2.09600472e+00,  1.38497184e+00,
		7.59309325e+02,  2.30507094e+00,  2.49178735e+00,
		-7.38736857e+02,  2.39822783e+00, -2.45891117e+00
	])

	return starts, x_target


def save_plots(results, method_name, save_path="plots"):
	
	save_path = os.path.join(save_path, method_name)
	os.makedirs(save_path, exist_ok=True)

	n_results = len(results)
	max_cols = 3

	# Calculate optimal subplot layout
	if n_results == 1:
		rows, cols = 1, 1
		figsize = (8, 6)
	elif n_results <= max_cols:
		rows, cols = 1, n_results
		figsize = (5 * n_results, 6)
	else:
		cols = max_cols
		rows = (n_results + cols - 1) // cols  # Ceiling division
		figsize = (5 * cols, 4 * rows)
		
	
	fig, axs = plt.subplots(rows, cols, figsize=figsize)

	if n_results == 1:
		axs = [axs]
	else:
		axs = axs.flatten() if rows > 1 or cols > 1 else [axs]

	for i, result in enumerate(results):
		ax = axs[i]
		ax.plot(a, phi_vectorized(result["x_solution"], a), label="approximation of sin")
		ax.plot(a, b, alpha=0.8, label="sin")

		ax.grid(True, alpha=0.3)
		ax.set_xlabel("x")
		ax.set_ylabel("y")

		title_parts = []
		if "iterations" in result:
			title_parts.append(f"Iters: {result['iterations']}")
		if "grad_norm" in result:
			title_parts.append(f"Grad: {result['grad_norm']:.2e}")
		if "x_solution" in result:
			try:
				loss_val = f(result["x_solution"])
				title_parts.append(f"Loss: {loss_val:.3f}")
			except NameError:
				pass
		
		ax.set_title(" | ".join(title_parts), fontsize=10, pad=10)
		ax.legend(fontsize=9)

	# Hide unused plots
	for i in range(n_results, len(axs)):
		axs[i].set_visible(False)

	fig.suptitle(f"Optimization Results - {method_name}", fontsize=14, y=0.98)

	plt.tight_layout(rect=[0, 0, 1, 0.96])

	# Save with timestamp
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	filename = f"{method_name}_{timestamp}.png"
	filepath = os.path.join(save_path, filename)

	fig.savefig(filepath, dpi=300, bbox_inches='tight')
	plt.close(fig)  # Free memory
	
	print(f"Saved plot: {filepath}")
	

def to_latex_scientific(x, precision=7):
    base, exponent = f"{x:.{precision}e}".split("e")
    return f"${base} \\times 10^{{{int(exponent)}}}$"

def create_latex_table(results):
	"""
	Creates a LaTeX table string with the same structure as Table 3,
	but with empty values represented by dashes.
	"""
	table = []
	table.append("\\begin{table}[ht]")
	table.append("\\begin{tabular}{ccccccccc}")
	table.append("\\hline")
	table.append("Run & Iters & Time (s) & $\\|\\tilde{x} - x^*\\|_2$ & $\|\\nabla f\\|$ & $\\ell_k$ & $q_k$ & Stopping Crit. & Convergence \\\\")
	table.append("\\hline")
	for result in results:
		run = result['run']
		iterations = result['iterations']
		time = result['time']
		distance_to_target_solution = to_latex_scientific(result['distance_to_target_solution'], precision=4)
		grad_norm = to_latex_scientific(result['grad_norm'], precision=4)

		table.append(f"{run}& {iterations} & {time:.2f} & {distance_to_target_solution} & {grad_norm} & - & - & - & - \\\\")
	table.append("\\end{tabular}")
	table.append("\\caption{Caption}")
	table.append("\\end{table}")
	
	print("\n".join(table))