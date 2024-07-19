import numpy as np
from scipy.sparse.linalg import gmres
from scipy.sparse import csr_matrix, lil_matrix
import matplotlib.pyplot as plt

def load_mesh(xy_path, mesh_path):
    """
    Load mesh node coordinates and element connectivity.
    """
    nodes = np.loadtxt(xy_path)
    elements = np.loadtxt(mesh_path, dtype=int)
    elements[:, :3] -= 1  # Adjust to 0-based indexing
    return nodes, elements

def shape_functions_gradients(element_nodes):
    """
    Calculate gradients of shape functions in the local coordinate system of a triangle element.
    """
    x = element_nodes[:, 0]
    y = element_nodes[:, 1]
    area = 0.5 * np.linalg.det(np.array([[1, x[0], y[0]], [1, x[1], y[1]], [1, x[2], y[2]]]))
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2 * area)
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2 * area)
    return b, c, area

def element_matrices(nodes, element, K, beta, tau):
    """
    Compute element stiffness, mass, and SUD stabilization matrices.
    """
    element_nodes = nodes[element[:3]]  # Get the coordinates of the element nodes
    b, c, area = shape_functions_gradients(element_nodes)
    h = np.max(np.sqrt(np.sum((element_nodes[np.newaxis, :] - element_nodes[:, np.newaxis])**2, axis=2)))  # edge length

    stiffness = K * area * (np.outer(b, b) + np.outer(c, c))
    mass = (area / 12.0) * (2 * np.identity(3) + np.ones((3, 3)))  # Simplified lumped mass matrix

    beta_magnitude = np.linalg.norm(beta)
    stabilization = tau * h * K * beta_magnitude * area * (np.outer(beta[0]*b + beta[1]*c, beta[0]*b + beta[1]*c))

    return stiffness, mass, stabilization

def assemble_global_matrices(nodes, elements, K, beta, tau):
    """
    Assemble the global stiffness, mass, and SUD stabilization matrices.
    """
    N = nodes.shape[0]
    global_stiffness = lil_matrix((N, N))
    global_mass = lil_matrix((N, N))
    global_stabilization = lil_matrix((N, N))

    for element in elements:
        local_indices = element[:3]
        stiffness, mass, stabilization = element_matrices(nodes, local_indices, K, beta, tau)
        for i in range(3):
            for j in range(3):
                global_stiffness[local_indices[i], local_indices[j]] += stiffness[i, j]
                global_mass[local_indices[i], local_indices[j]] += mass[i, j]
                global_stabilization[local_indices[i], local_indices[j]] += stabilization[i, j]

    return csr_matrix(global_stiffness), csr_matrix(global_mass), csr_matrix(global_stabilization)

def apply_dirichlet(global_matrix, global_rhs, dirichlet_nodes, dirichlet_values):
    """
    Apply Dirichlet boundary conditions robustly to avoid GMRES convergence issues.
    """
    for node, value in zip(dirichlet_nodes, dirichlet_values):
        # Adjust the global matrix for Dirichlet conditions
        global_matrix[node, :] = 0
        global_matrix[:, node] = 0
        global_matrix[node, node] = 1
        global_rhs[node] = value
    return global_matrix, global_rhs

def solve_system(global_stiffness, global_mass, global_stabilization, u0, rhs):
    """
    Solve the system using GMRES with the assembled matrices.
    """
    system_matrix = global_stiffness + global_mass + global_stabilization
    solution, exitCode = gmres(system_matrix, rhs)
    return solution

def plot_solution(nodes, solution, title='Concentration Distribution'):
    """
    Plot the solution of the convection-diffusion problem.
    """
    fig, ax = plt.subplots()
    sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=solution, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Concentration')
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    plt.show()
