
# Import the library functions
from convection_diffusion_solver import load_mesh, assemble_global_matrices, apply_dirichlet, solve_system, plot_solution
import numpy as np

# Paths to mesh data
xy_path = 'mesh/xy.dat'  # Adjust this path
mesh_path = 'mesh/mesh.dat'  # Adjust this path
dirnod_path = 'mesh/dirnod.dat'  # Path to Dirichlet node indices
dirval_path = 'mesh/dirval.dat'  # Path to Dirichlet node values

# Load mesh data
nodes, elements = load_mesh(xy_path, mesh_path)

# Define physical and simulation parameters
K = 0.01  # Diffusion coefficient
beta = np.array([1, 3])  # Velocity field (constant across the domain)
tau = 0.01  # SUPG/SUD stabilization parameter

# Assemble global matrices including SUD stabilization
global_stiffness, global_mass, global_stabilization = assemble_global_matrices(nodes, elements, K, beta, tau)

# Initial condition (assuming initial concentration is zero everywhere)
u0 = np.zeros(nodes.shape[0])

# Load and apply Dirichlet boundary conditions
dirichlet_nodes = np.loadtxt(dirnod_path, dtype=int) - 1  # Adjust index if necessary
dirichlet_values = np.loadtxt(dirval_path)
global_rhs = np.zeros(nodes.shape[0])  # Initialize the right-hand side vector
global_stiffness, global_rhs = apply_dirichlet(global_stiffness, global_rhs, dirichlet_nodes, dirichlet_values)

# Solve the system using GMRES
final_solution = solve_system(global_stiffness, global_mass, global_stabilization, u0, global_rhs)

# Plot the final solution
plot_solution(nodes, final_solution, title='Final Concentration Distribution After Simulation')