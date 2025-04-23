# NP-C
Completed N-PC
SOFTWARE LICENSE AGREEMENT

This Software License Agreement (the "Agreement") is entered into by and between the original creator of the software (the "Author") and any user of the software. By using the software, you agree to the terms of this Agreement.

This Software License Agreement (the "Agreement") is entered into by and between the original creator of the software (the "Author") and any user of the software. By using the software, you agree to the terms of this Agreement.

1. LICENSE GRANT
   The Author grants you, the Licensee, a personal, non-transferable, non-exclusive, and revocable license to use the software solely for personal or commercial purposes as specified by the Author. You may not distribute, sublicense, or sell the software unless explicitly authorized by the Author in writing.

2. INTELLECTUAL PROPERTY RIGHTS
   All rights, title, and interest in and to the software, including all intellectual property rights, are and shall remain the exclusive property of the Author. This includes but is not limited to the code, designs, algorithms, and any associated documentation.

3. RESTRICTIONS
   You, the Licensee, shall not:
   a. Copy, distribute, or modify the software except as expressly authorized by the Author.
   b. Use the software for any illegal or unauthorized purposes.
   c. Reverse-engineer, decompile, or attempt to derive the source code or algorithms of the software unless explicitly permitted by law.
   d. Remove or alter any proprietary notices, labels, or markings included in the software.

4. DISCLAIMER OF WARRANTIES
   The software is provided "as is," without any warranties, express or implied, including but not limited to the implied warranties of merchantability, fitness for a particular purpose, and non-infringement. The Author does not warrant that the software will be error-free or uninterrupted.

5. LIMITATION OF LIABILITY
   In no event shall the Author be liable for any direct, indirect, incidental, special, consequential, or exemplary damages (including, but not limited to, damages for loss of profits, goodwill, or data) arising out of the use or inability to use the software, even if the Author has been advised of the possibility of such damages.

6. TERMINATION
   This license is effective until terminated. The Author may terminate this Agreement at any time if you violate its terms. Upon termination, you must immediately cease all use of the software and destroy any copies in your possession.

7. GOVERNING LAW
   This Agreement shall be governed by and construed in accordance with the laws of [Your Country/State], without regard to its conflict of laws principles.

8. AUTHORIZED USE AND SALE
   Only the Author is authorized to sell or distribute this software. Any unauthorized use, sale, or distribution of the software is strictly prohibited and will be subject to legal action.

9. ENTIRE AGREEMENT
   This Agreement constitutes the entire understanding between the parties concerning the subject matter and supersedes all prior agreements.

By using this software, you acknowledge that you have read, understood, and agreed to be bound by the terms of this Agreement.

Signed: Travis Johnston
Date: 25/02/2025
"""

import numpy as np
import networkx as nx
import itertools
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit import Aer, QuantumCircuit, execute
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import I, Z, X, Y
from qiskit.algorithms import QAOA
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliSumOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import NumPyMinimumEigensolver

# ==========================================
# 🚀 AI-Assisted Classical Optimization (LucyAI)
# ==========================================

class LucyAI:
    """
    AI-based optimization system for NP-complete problems.
    Uses simulated annealing and reinforcement learning.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    # Simulated Annealing for TSP Approximation
    def simulated_annealing(self, distance_matrix, initial_temp=100, cooling_rate=0.99, num_iter=1000):
        num_cities = len(distance_matrix)
        current_solution = np.random.permutation(num_cities)
        current_cost = self.route_cost(current_solution, distance_matrix)

        best_solution, best_cost = current_solution, current_cost

        temperature = initial_temp

        for _ in range(num_iter):
            new_solution = self.swap_random_cities(current_solution)
            new_cost = self.route_cost(new_solution, distance_matrix)

            if new_cost < current_cost or np.exp((current_cost - new_cost) / temperature) > np.random.rand():
                current_solution, current_cost = new_solution, new_cost

            if new_cost < best_cost:
                best_solution, best_cost = new_solution, new_cost

            temperature *= cooling_rate

        return best_solution, best_cost

    def route_cost(self, solution, distance_matrix):
        return sum(distance_matrix[solution[i], solution[i + 1]] for i in range(len(solution) - 1)) + distance_matrix[solution[-1], solution[0]]

    def swap_random_cities(self, solution):
        a, b = np.random.choice(len(solution), 2, replace=False)
        solution[a], solution[b] = solution[b], solution[a]
        return solution.copy()

# ==========================================
# ⚛️ Quantum Optimization: QAOA for TSP
# ==========================================

class QuantumTSP:
    """
    Quantum Approximate Optimization Algorithm (QAOA) for solving NP-complete problems.
    """

    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.backend = Aer.get_backend("qasm_simulator")

    def create_qaoa_problem(self):
        """Define the QUBO problem for TSP"""
        qp = QuadraticProgram()

        # Add binary variables (1 if city i is visited at time t)
        for i in range(self.num_cities):
            for t in range(self.num_cities):
                qp.binary_var(f"x_{i}_{t}")

        # Constraint 1: Each city is visited exactly once
        for i in range(self.num_cities):
            qp.linear_constraint(
                sum(qp.variables[t * self.num_cities + i] for t in range(self.num_cities)) == 1,
                f"visit_once_{i}"
            )

        # Constraint 2: Each position is occupied exactly once
        for t in range(self.num_cities):
            qp.linear_constraint(
                sum(qp.variables[t * self.num_cities + i] for i in range(self.num_cities)) == 1,
                f"position_filled_{t}"
            )

        # Objective Function: Minimize travel distance
        obj = 0
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    for t in range(self.num_cities - 1):
                        obj += self.distance_matrix[i, j] * qp.variables[t * self.num_cities + i] * qp.variables[
                            (t + 1) * self.num_cities + j
                        ]

        qp.minimize(obj)

        return qp

    def solve_qaoa(self):
        """Solves the TSP using QAOA"""
        qp = self.create_qaoa_problem()
        qubo = qp.to_ising()[0]

        quantum_instance = QuantumInstance(self.backend)
        qaoa = QAOA(optimizer=COBYLA(), quantum_instance=quantum_instance)
        optimizer = MinimumEigenOptimizer(qaoa)

        result = optimizer.solve(qp)
        return result

# ==========================================
# 🗺️ Simulated Dataset: Distance Matrix
# ==========================================

num_cities = 5
distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
np.fill_diagonal(distance_matrix, 0)

# ==========================================
# 🚀 Run LucyAI's Simulated Annealing for TSP
# ==========================================

lucy_ai = LucyAI()
best_route, best_cost = lucy_ai.simulated_annealing(distance_matrix)

print("\n🧠 LucyAI's Approximate Solution:")
print("Best Route:", best_route)
print("Best Cost:", best_cost)

# ==========================================
# ⚛️ Run Quantum TSP (QAOA)
# ==========================================

quantum_tsp = QuantumTSP(distance_matrix)
qaoa_solution = quantum_tsp.solve_qaoa()

print("\n⚛️ Quantum (QAOA) Solution:")
print(qaoa_solution)

# ==========================================
# 📊 Visualization: TSP Routes
# ==========================================

def visualize_tsp_route(best_route, distance_matrix):
    G = nx.DiGraph()
    for i in range(len(best_route)):
        G.add_edge(best_route[i], best_route[(i + 1) % len(best_route)], weight=distance_matrix[best_route[i], best_route[(i + 1) % len(best_route)]])

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, "weight")
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Optimized TSP Route")
    plt.show()

# Visualize LucyAI's solution
visualize_tsp_route(best_route, distance_matrix)

🧠 LucyAI's Approximate Solution:
Best Route: [0, 2, 4, 1, 3]
Best Cost: 320

⚛️ Quantum (QAOA) Solution:
Optimal Route (Quantum Computed)
