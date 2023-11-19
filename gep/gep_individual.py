import numpy as np

class GEPIndividual:
    def __init__(self, gene_length):
        self.gene = np.random.randint(2, size=gene_length)  # Random binary gene
        self.fitness = None

    def mutate(self):
        mutation_point = np.random.randint(len(self.gene))
        self.gene[mutation_point] = 1 - self.gene[mutation_point]  # Flip bit

    def evaluate(self, data):
        # Placeholder for evaluation logic
        self.fitness = np.random.rand()  # Example: assign random fitness
