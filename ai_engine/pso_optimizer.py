import numpy as np
import networkx as nx
from ai_engine.constraint_cost import calculate_constraint_cost
from copy import deepcopy
import random


# Represents a single particle in the PSO swarm.
class Particle:
    def __init__(self, graph, spec):
        self.graph = graph.copy()
        self.spec = spec
        self.position = self._get_position_vector()
        self.velocity = np.random.uniform(-1, 1, size=self.position.shape) * 0.5
        self.best_position = self.position.copy()
        self.current_cost, _ = calculate_constraint_cost(self.graph, spec)
        self.best_cost = self.current_cost

    # Extracts a 1D position vector from the graph's node positions.
    def _get_position_vector(self):
        return np.array(
            [
                pos
                for node_id in sorted(self.graph.nodes())
                for pos in self.graph.nodes[node_id].get("position", [0, 0])
            ]
        )

    # Updates the graph's node positions from a 1D position vector.
    def _set_position_vector(self, position):
        for i, node_id in enumerate(sorted(self.graph.nodes())):
            if i * 2 + 1 < len(position):
                self.graph.nodes[node_id]["position"] = [
                    position[i * 2],
                    position[i * 2 + 1],
                ]

    # Updates the particle's velocity.
    def update_velocity(self, global_best_position, w=0.729, c1=1.494, c2=1.494):
        r1, r2 = np.random.random(2)
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_component + social_component
        self.velocity = np.clip(self.velocity, -2.0, 2.0)

    # Updates the particle's position and recalculates its cost.
    def update_position(self):
        self.position += self.velocity
        self._apply_position_constraints()
        self._set_position_vector(self.position)
        self.current_cost, _ = calculate_constraint_cost(self.graph, self.spec)
        if self.current_cost < self.best_cost:
            self.best_cost = self.current_cost
            self.best_position = self.position.copy()

    # Applies constraints to keep the particle's position within bounds.
    def _apply_position_constraints(self):
        for i, node_id in enumerate(sorted(self.graph.nodes())):
            if i * 2 + 1 < len(self.position):
                x, y = np.clip(self.position[i * 2], -50, 50), np.clip(
                    self.position[i * 2 + 1], -50, 50
                )
                self.graph.nodes[node_id]["position"] = [x, y]
                self.position[i * 2], self.position[i * 2 + 1] = x, y


# Optimizer that uses a PSO algorithm to refine a PCB layout.
class PSOOptimizer:
    def __init__(self, num_particles=30, max_iterations=150, tolerance=1e-6):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    # Optimizes the PCB layout using the PSO algorithm.
    def optimize_layout(self, initial_graph, spec):
        particles = [Particle(initial_graph, spec) for _ in range(self.num_particles)]
        global_best_particle = min(particles, key=lambda p: p.current_cost)
        global_best_position = global_best_particle.best_position.copy()
        previous_best_cost = float("inf")
        for iteration in range(self.max_iterations):
            w = 0.9 - (0.9 - 0.4) * (iteration / self.max_iterations)
            for particle in particles:
                particle.update_velocity(global_best_position, w=w)
                particle.update_position()
                if particle.best_cost < global_best_particle.best_cost:
                    global_best_particle = deepcopy(particle)
                    global_best_position = particle.best_position.copy()
            if (
                abs(previous_best_cost - global_best_particle.best_cost)
                < self.tolerance
            ):
                break
            previous_best_cost = global_best_particle.best_cost
            if iteration % 20 == 0:
                print(
                    f"PSO Iteration {iteration}, Best Cost: {global_best_particle.best_cost:.4f}"
                )
        print(f"PSO completed. Final cost: {global_best_particle.best_cost:.4f}")
        return global_best_particle.graph

    # Optimizes the routing of the PCB by adjusting edge properties.
    def optimize_routing(self, graph, spec):
        optimized_graph = graph.copy()
        for u, v, data in optimized_graph.edges(data=True):
            pos_u, pos_v = optimized_graph.nodes[u].get(
                "position", [0, 0]
            ), optimized_graph.nodes[v].get("position", [0, 0])
            data["length"] = np.sqrt(
                (pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2
            )
            signal_u, signal_v = optimized_graph.nodes[u].get(
                "signal_type", "signal"
            ), optimized_graph.nodes[v].get("signal_type", "signal")
            if "power" in signal_u or "power" in signal_v:
                data["width"] = max(data.get("width", 0.2), 0.5)
        return optimized_graph


# Example usage
if __name__ == "__main__":
    g = nx.Graph()
    for i in range(5):
        g.add_node(i, signal_type="signal", position=[i * 1.5, i * 1.5])
    for i in range(4):
        g.add_edge(i, i + 1, length=1.0, width=0.2, layer=0)
    spec = {"max_trace_length": 10.0, "layers": 2}
    optimizer = PSOOptimizer(num_particles=15, max_iterations=50)
    optimized_graph = optimizer.optimize_layout(g, spec)
    original_cost, _ = calculate_constraint_cost(g, spec)
    optimized_cost, _ = calculate_constraint_cost(optimized_graph, spec)
    print(f"Original cost: {original_cost:.4f}")
    print(f"Optimized cost: {optimized_cost:.4f}")
