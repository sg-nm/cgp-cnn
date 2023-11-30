from gep_individual import GEPIndividual

class GEPPopulation:
    def __init__(self, size, gene_length):
        self.individuals = [GEPIndividual(gene_length) for _ in range(size)]

    def evolve(self, data):
        for individual in self.individuals:
            individual.evaluate(data)
            individual.mutate()  # Example: mutate every individual

        # Add selection and crossover logic here
