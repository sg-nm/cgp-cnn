from gep_population import GEPPopulation

POPULATION_SIZE = 50
GENE_LENGTH = 10

def main():
    population = GEPPopulation(POPULATION_SIZE, GENE_LENGTH)
    
    # Example dataset
    data = []

    for _ in range(100):  # Number of generations
        population.evolve(data)
        # Add logic to check for termination condition

    # Extract best individual
    best_individual = max(population.individuals, key=lambda ind: ind.fitness)
    print("Best Individual:", best_individual.gene, "Fitness:", best_individual.fitness)

if __name__ == "__main__":
    main()
