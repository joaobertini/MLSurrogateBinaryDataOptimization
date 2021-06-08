
import numpy as np

# tournament selection
def selection(pop, scores, k=3):
    #first random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if fitbest(scores[ix],sum(pop[ix])) > fitbest(scores[selection_ix],sum(pop[selection_ix])):  # maximizando  vpl/num_poço
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children - um ponto
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if np.random.rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = np.random.randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if np.random.rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]

import scipy.stats
media = 17
desvio = 7.5 # duas vezes o desvio
d = scipy.stats.norm(media,desvio)

def fitbest(score, numPoco):
    return score if numPoco < media else score * d.pdf(numPoco)/d.pdf(media)

# genetic algorithm

def zero_one(x):
    """Returns True only if x is odd."""
    if x < 0.85:
        return 0
    else:
        return 1


def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut, logger):
    # initial population of random bitstring
    pop = [list(map(zero_one, np.random.rand(n_bits).tolist())) for _ in range(n_pop)]

    # keep track of best solution
    best, best_eval = 0, objective(np.array([pop[0]]))  # pop[0].reshape(1,-1)
    best = pop[0]
    #logger(best)
    # enumerate generations
    #best, best_eval = pop[0], scores[0]
    for gen in range(n_iter):
        if gen % 100 == 0 or gen == n_iter - 1:
          logger("Iteration " + str(gen + 1) + " of " + str(n_iter))

        # evaluate all candidates in the population
        scores = [objective(np.array([c])) for c in pop]   # c.reshape(1,-1)) for c in pop
        # check for new best solution
        #if gen == 1:
        for i in range(n_pop):
            if fitbest(scores[i],sum(pop[i])) > fitbest(best_eval,sum(best)): #scores[i] > best_eval:  # maximizando
                best, best_eval = pop[i], scores[i]
                logger(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]

