import numpy as np

# for now: S is a np.ndarray of dim=1 and fixed length with real entries

# SINGLE STATE METHODS
def hill_climbing(S, Tweak, Copy, Quality, optimal, maxiter=100):
    # S - initial candidate solution
    # Tweak - problem specific function, 
    # takes candidate solution, returns randomly slightly different solution
    # Copy - problem specific function, 
    # takes candidate solution, returns a copy
    # Quality - problem specific function,
    # takes candidate solution, returns quality of it (goodness)
    # optimal - threshold of optimal solution
    
    for _ in range(maxiter):
        R = Tweak(Copy(S))
        QR = Quality(R)
        if QR > Quality(S):
            S = R
            if QR >= optimal:
                break
    return S

def steepest_ascent_hill_climbing(S, Tweak, Copy, Quality, n, optimal, maxiter=100):
    # S - initial candidate solution
    # Tweak - problem specific function, 
    # takes candidate solution, returns randomly slightly different solution
    # Copy - problem specific function, 
    # takes candidate solution, returns a copy
    # Quality - problem specific function,
    # takes candidate solution, returns quality of it (goodness)
    # n - number of Tweaks to sample new candidate solution
    # optimal - threshold of optimal solution
    
    for _ in range(maxiter):
        R = Tweak(Copy(S))
        for i in range(n):
            W = Tweak(Copy(S))
            if Quality(W) > Quality(R):
                R = W
        QR = Quality(R)
        if QR > Quality(S):
            S = R
            if QR >= optimal:
                break
    return S

def steepest_ascent_hill_climbing_w_replacement(S, Tweak, Copy, Quality, n, optimal, maxiter=100):
    # S - initial candidate solution
    # Tweak - problem specific function, 
    # takes candidate solution, returns randomly slightly different solution
    # Copy - problem specific function, 
    # takes candidate solution, returns a copy
    # Quality - problem specific function,
    # takes candidate solution, returns quality of it (goodness)
    # n - number of Tweaks to sample new candidate solution
    # optimal - threshold of optimal solution
    
    Best = S
    QB = Quality(Best)
    for _ in range(maxiter):
        R = Tweak(Copy(S))
        for i in range(n):
            W = Tweak(Copy(S))
            if Quality(W) > Quality(R):
                R = W
        S = R
        QS = Quality(S)
        if QS > Quality(Best):
            Best = S
            QB = QS
            if QB >= optimal:
                break
    return S


# random sample
def random_candidate(l, min, max):
    return np.random.uniform(low=min, high=max, size=l)

# simple Tweak implementation
def bounded_uniform_convolution(v, p, r, min, max):
    for i in range(len(v)):
        if p >= np.random.uniform():
            n = np.random.uniform(-r, r)
            vin = v[i]+n
            while vin < min or vin > max:
                n = np.random.uniform(-r, r)
                vin = v[i]+n
            v[i] = vin
    return v

                

def random_search(Generate, Quality, optimal, maxiter=100):
    # Best - random initial candidate solution
    # Generate - generates random candidate solution
    Best = Generate()
    QB = Quality(Best)
    for _ in range(maxiter):
        S = Generate()
        QS = Quality(S)
        if QS > QB:
            Best = S
            QB = QS
            if QB >= optimal:
                break
    return S
                    

def random_sample(arr, size: int = 1):
    return arr[np.random.choice(len(arr), size=size, replace=False)]

def hill_climbing_w_random_restarts(T, Generate, Tweak, Copy, Quality, optimal, maxiter=100):
    # T - distribution of possible time intervals
    # Tweak - problem specific function, 
    # takes candidate solution, returns randomly slightly different solution
    # Copy - problem specific function, 
    # takes candidate solution, returns a copy
    # Quality - problem specific function,
    # takes candidate solution, returns quality of it (goodness)
    # optimal - threshold of optimal solution
    
    S = Generate()
    Best = S
    for _ in range(maxiter):
        time = random_sample(T)
        for i in range(time):
            R = Tweak(Copy(S))
            if Quality(R) > Quality(S):
                S = R
                if Quality(S) >= optimal:
                    return S
        if Quality(S) > Quality(Best):
            Best = S
            if Quality(Best) >= optimal:
                return Best
        S = Generate()
    return Best


def gaussian_convolution(v, p, var, min, max):
    for i in range(len(v)):
        if p >= np.random.uniform():
            n = np.random.normal(loc=0,scale=np.sqrt(var))
            vin = v[i]+n
            while vin < min or vin > max:
                n = np.random.normal(loc=0,scale=np.sqrt(var))
                vin = v[i]+n
            v[i] = vin
    return v
            
# generates two random numbers (gaussian)
def box_muller_marsaglia_polar_method(mean, var):
    x = np.random.uniform(-1.0, 1.0)
    y = np.random.uniform(-1.0, 1.0)
    w = x**2 + y**2
    while w < 0 or w > 1:
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        w = x**2 + y**2
    
    g = mean + x * var * np.sqrt(-2*np.log(w)/w)
    h = mean + y * var * np.sqrt(-2*np.log(w)/w)
    return g, h


def simulated_annealing(t, Decrease, Generate, Tweak, Copy, Quality, optimal, maxiter=100):
    # t - starting temperature
    
    S = Generate()
    Best = S
    QB = Quality(Best)
    for _ in range(maxiter):
        R = Tweak(Copy(S))
        QR = Quality(R)
        QS = Quality(S)
        if QR > QS or np.random.uniform() < np.exp((QR-QS)/t):
            S = R
        t = Decrease(t)
        if t <= 0:
            break
        if QS > QB:
            Best = S
            QB = QS
            if QB >= optimal:
                return Best
    return Best

class FixedLengthQueue:
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        self.queue = []
        self.num = 0
        
    def enqueue(self, value):
        if self.num+1 >= self.capacity:
            del self.queue[0]
        else:
            self.num += 1
        self.queue.append(value)
    
    def dequeue(self):
        assert len(self.queue)!= 0
        item = self.queue[0]
        del self.queue[0]
        self.num -= 1
        return item

    def isin(self, value):
        return value in self.queue
            
        

def tabu_search(l, n, Generate, Tweak, Copy, Quality, optimal, maxiter=100):
    # l - max len of taboo list
    # n - number of tweaks
    queue = FixedLengthQueue(l)
    S = Generate()
    Best = S
    QB = Quality(Best)
    queue.enqueue(S)
    for _ in range(maxiter):
        R = Tweak(Copy(S))
        for i in range(n):
            W = Tweak(Copy(S))
            if not queue.isin(W) and (Quality(W) > Quality(R) or queue.isin(R)):
                R = W
            if not queue.isin(R):
                S = R
                queue.enqueue(R)
            QS = Quality(S)
            if QS > QB:
                Best = S
                QB = QS
                if QB >= optimal:
                    return Best
    return Best
                

def iterated_local_search_w_random_restarts(T, Home, Perturb, Generate, Tweak, Copy, Quality, optimal, maxiter=100):
    # T - distribution of possible time intervals
    # Home - function, determines wether to adopt new home base
    # Tweak - problem specific function, 
    # takes candidate solution, returns randomly slightly different solution
    # Copy - problem specific function, 
    # takes candidate solution, returns a copy
    # Quality - problem specific function,
    # takes candidate solution, returns quality of it (goodness)
    # optimal - threshold of optimal solution
    
    S = Generate()
    Best = S
    H = S # home base
    Best = S
    QB = Quality(Best)
    for _ in range(maxiter):
        time = random_sample(T)
        for i in range(time):
            R = Tweak(Copy(S))
            QS = Quality(S)
            if Quality(R) > QS:
                S = R
                if QS >= optimal:
                    return S
                
        QS = Quality(S)
        if QS > QB:
            Best = S
            QB = QS
            if QB >= optimal:
                return Best
            
        H = Home(H, S) # tricky to implement exploration vs. exploitation
        S = Perturb(H)
    return Best



# POPULATION METHODS
def abstract_generational_ea(InitialPopulation, Fitness, Breed, Join, optimal, maxiter):
    P = InitialPopulation()
    Best = None
    for _ in range(maxiter):
        for i in range(len(P)):
            if Best is None or Fitness(P[i]) > Fitness(Best):
                Best = P[i]
                if Fitness(Best) >= optimal:
                    return Best
        P = Join(P, Breed(P))
    return Best



def mu_lambda_evolution_strategy(mu, lam, Choose_mu, InitialPopulation, Fitness, Copy, Mutate, optimal, maxiter):
    # mu - number of parents selected
    # lam - number of children generated
    
    P = InitialPopulation(lam)
    Best = None
    for _ in range(maxiter):
        for i in range(len(P)):
            if Best is None or Fitness(P[i]) > Fitness(Best):
                Best = P[i]
                if Fitness(Best) >= optimal:
                    return Best
        
        Q = Choose_mu(mu, P, P_fitness)
        P = []
        for i in range(len(Q)):
            for _ in range(int(lam/mu)):
                P.append(Mutate(Copy(Q[i])))
    return Best
    
def mu_plus_lambda_evolution_strategy(mu, lam, Choose_mu, InitialPopulation, Fitness, Copy, Mutate, optimal, maxiter):
    # mu - number of parents selected
    # lam - number of children generated
    
    P = InitialPopulation(lam)
    Best = None
    for _ in range(maxiter):
        for i in range(len(P)):
            if Best is None or Fitness(P[i]) > Fitness(Best):
                Best = P[i]
                if Fitness(Best) >= optimal:
                    return Best
        
        Q = Choose_mu(mu, P, P_fitness) # choose mu best individuals
        P = Q
        for i in range(len(Q)):
            for _ in range(int(lam/mu)):
                P.append(Mutate(Copy(Q[i])))
    return Best
    
    
def genetic_algorithm(popsize, InitialPopulation, Fitness, SelectWithReplacement, Crossover, Copy, Mutate, optimal, maxiter):
    
    P = InitialPopulation(popsize)
    Best = None
    for _ in range(maxiter):
        for i in range(len(P)):
            if Best is None or Fitness(P[i]) > Fitness(Best):
                Best = P[i]
                if Fitness(Best) >= optimal:
                    return Best
                
        Q = []
        for _ in range(int(popsize/2)):
            Pa = SelectWithReplacement(P)
            Pb = SelectWithReplacement(P)
            Ca, Cb = Crossover(Copy(Pa), Copy(Pb))
            Q.append(Mutate(Ca))
            Q.append(Mutate(Cb))
        P = Q
    return Best
            
    
def random_boolean_vector(l):
    return np.random.choice(a=[False, True], size=l, p=[0.5, 0.5])

def bit_flip_mutation(p, v):
    for i in range(len(v)):
        if p >= np.random.uniform():
            v[i] = not v[i]
    return v

def one_point_crossover(v, w):
    c = np.random.randint(1, len(v))
    if c != 1:
        for i in range(c, len(v)):
            v[i],w[i] = w[i],v[i]
    return v, w

def two_point_crossover(v, w):
    c = np.random.randint(1, len(v))
    d = np.random.randint(1, len(v))
    if c > d:
        c, d = d, c
    if c != d:
        for i in range(c, d):
            v[i],w[i] = w[i],v[i]
    return v, w

def uniform_crossover(p, v, w):
    for i in range(len(v)):
        if p >= np.random.uniform():
            v[i],w[i] = w[i],v[i]
    return v, w
        

def random_shuffle(v):
    for i in range(len(v)-1,1,-1):
        j = np.random.randint(0,i)
        v[i], v[j] = v[j], v[i]
    return v

def k_uniform_crossover(p, W):
    l = len(W[0])
    k = len(W)

    v = np.zeros(k)
    for i in range(l):
        if p >= np.random.uniform():
            for j in range(k):
                v[j] = W[j][i]
            v = random_shuffle(v)
            for j in range(k):
                w = W[j]
                w[i] = v[j]
                W[j] = w
    return W

def line_recombination(p, v, w):
    a = np.random.uniform(-p, 1+p)
    b = np.random.uniform(-p, 1+p)
    for i in range(len(v)):
        t = a * v[i] + (1 - a) * w[i]
        s = b * w[i] + (1 - a) * v[i]
        if t >= -p and t <= 1+p and s >= -p and t <= 1+p:
            v[i] = t
            w[i] = s

    return v, w

def fitness_prop_selection(p, f, gen=False):
    # selects an individual in proportion to fitness value
    # p - population vector of individuals
    # f - fitness value for each individual in p
    l = len(p) # = len(f)

    if gen:
        f = np.asarray(f)
        if f.all() == 0.0:
            f = np.ones(l)
        for i in range(1, l):
            f[i] += f[i-1]
    n = np.random.uniform(0, f[-1])
    for i in range(1, l):
        if n > f[i-1] and n < f[i]:
            return p[i]
    return p[0]

def stochastic_universal_sampling():
    pass
    # maybe later

def tournament_selection(P, t, Select, Fitness):
    # P - population
    # t - tournament size

    Best = np.random.choice(P)
    for i in range(2, t+1):
        Next = np.random.choice(P)
        if Fitness(Next) > Fitness(Best):
            Best = Next
    return Best


# AssessFitness - computes fitness for each individual in a population
# Fitness - computes fitness for one individual in a population
def genetic_algorithm_w_elitism(popsize, nelite, Choose_mu, Generate, Fitness, Select, Crossover, Copy, Mutate, optimal, maxiter):
    # popsize - population size
    # nelite - number of elites per generation
    

    assert popsize-nelite % 2 == 0
    P = Generate(popsize)
    Best = None
    for _ in range(maxiter):
        for i in range(popsize):
            if Best is None or Fitness(P[i]) > Fitness(Best):
                Best = P[i]
                if Fitness(Best) >= optimal:
                    return Best
        Q = Choose_mu(nelite, P, P_fitness)
        for i in range(int((popsize-nelite)/2)):
            Pa = Select(P)
            Pb = Select(P)
            Ca, Cb = Crossover(Copy(Pa),Copy(Pb))
            Q.append(Mutate(Ca))
            Q.append(Mutate(Cb))
        P = Q
    return Best
        
        
def scatter_search(Seeds, initsize, t, n, m, GenDiverse, Fitness, Crossover, Copy, Mutate, choose_fit, choose_div, Tweak, optimal, maxiter):
    P = Seeds
    for _ in range(int(initsize-len(Seeds))):
        P.append(GenDiverse(P))
    Best = None
    P_fitness = [0 for i in range(len(P))]
    for i in range(len(P)):
        P[i] = hill_climbing(P[i], Tweak, Copy, Fitness, optimal)
        P_fitness[i] = Fitness(P[i])
        if Best is None or P_fitness[i] > Fitness(Best):
            Best = P[i]
    
    for _ in range(maxiter):
        B = choose_fit(n, P, P_fitness)
        D = choose_div(m, P) # most diverse
        P = B + D
        Q = []
        for i in range(len(P)):
            for j in range(len(P)):
                if i == j: continue
                
                Ca, Cb = Crossover(Copy(P[i]), Copy(P[j]))
                Ca = Mutate(Ca)
                Cb = Mutate(Cb)
                Ca = hill_climbing(Ca, Tweak, Copy, Fitness, optimal, maxiter=t)
                Cb = hill_climbing(Cb, Tweak, Copy, Fitness, optimal, maxiter=t)
                if Fitness(Ca) > Fitness(Best):
                    Best = Ca
                    if Fitness(Best) >= optimal:
                        return Best
                if Fitness(Cb) > Fitness(Best):
                    Best = Cb
                    if Fitness(Best) >= optimal:
                        return Best
                Q.extend([Ca, Cb])
        P += Q
    return Best



def differential_evolution(a, popsize, Generate, Copy, Crossover, Fitness, optimal, maxiter):
    # a - mutation rate
    # popsize - population size
    
    P = Generate(popsize)
    Q = None # parents
    Best = None
    
    for _ in range(maxiter):
        for i in range(len(P)):
            if Q is not None and Fitness(Q[i]) > Fitness(P[i]):
                P[i] = Q[i]
            if Best is None or Fitness(P[i]) > Fitness(Best):
                Best = P[i]
                if Fitness(Best) >= optimal:
                    return Best
        Q = P
        for i in range(len(Q)):
            alp = random_sample(Q)
            bet = random_sample(Q)
            cet = random_sample(Q)
            d = alp + a*(bet - cet) # mutation
            P[i], _ = Crossover(d, Copy(Q[i]))
    return Best



# Need A Particle Class for that
def particle_swarm_optimization(swarmsize, al, be, ga, th, ep, Fitness, Generate, optimal, maxiter):
    P = Generate(swarmsize) # random particles with random velocity
    Best = None
    
    for _ in range(maxiter):
        for i in range(len(P)):
            if Best is None or Fitness(P[i].x) > Fitness(Best):
                Best = P[i]
                if Fitness(Best) >= optimal:
                    return Best
        for i in range(len(P)):
            x_star = P[i].prev
            x_plus = P[i].prev_inf
            x_ex   = P[i].prev_all
            for j in range(len(P[i].x)):
                b = np.random.uniform(low=0.0, high=be)
                c = np.random.uniform(low=0.0, high=ga)
                d = np.random.uniform(low=0.0, high=th)
                P[i].v[j] = al * P[i].v[j] + b*(x_star[j] - P[i].x[j]) + c*(x_plus[j] - P[i].x[j]) + d*(x_ex[j] - P[i].x[j])
        for i in range(len(P)):
            P[i].x = P[i].x + ep*P[i].v
    return Best


def gray_coding(v):
    # v - boolean vector
    w = v.copy()
    for i in range(1,len(w)):
        if v[i-1]:
            w[i] = not v[i]
    return w


# for integers
def random_integer_mutation(v, p, low, high):
    for i in range(len(v)):
        if p >= np.random.uniform():
            v[i] = np.random.randint(low, high)
    return v

def random_walk_mutation(v, p, b, low, high):
    for i in range(len(v)):
        if p >= np.random.uniform():
            while b < np.random.uniform():
                n = np.random.choice([1,-1])
                if v[i] + n <= high:
                    v[i] += n
                elif v[i] - n >= low:
                    v[i] -= n
    return v

def integer_line_recombination(v, w, p):
    pass
    # maybe later


class Node:
    def __init__(self, op, arity):
        self.op = op
        self.arity = arity
        self.children = [None for _ in range(arity)]

    def is_leaf(self):
        return self.arity == 0

def grow_tree(max_depth, function_set, Copy):
    def do_grow(curr_depth, max_depth, function_set):
        if curr_depth >= max_depth:
            return Copy(np.random.choice(function_set))
        else:
            n = Copy(np.random.choice(function_set))
            l = n.arity # elems in function_set are objects with operation, arity and children
            for i in range(l):
                n.children[i] = do_grow(depth+1, max_depth, function_set)
            return n

def production_ruleset_generation(t, v, p, r, d):
    # t - set of terminal/leaf symbols (dont expand)
    # v - vector of unique symbols ['a','b',...]
    # p - probability of picking terminal symbol
    # r - (bool) recursion allowed or not
    # d - (bool) disconnected rules allowed or not
    def random_symbol(t, h):
        # random element from t which is not in h
        pass

    n = np.random.randint(1, 50)
    rules = []
    for i in range(n):
        l = np.random.randint(1, 50)
        h = []
        for j in range(l):
            if (not r and i == n-1) or p < np.random.uniform():
                h.append(random_symbol(t, h))
            elif not r:
                h.append(random_symbol(v[i+1:], h))
            else:
                h.append(random_symbol(v, h))
        rules.append([v[i],h])
    if not d:
        for i in range(1,n):
            if v[i] not in [x[0] for x in rules]:
                l = np.random.uniform(0,i-1)
                rules[l] = [rules[l][0],[v[i]]+rules[l][1][1:]]
    return rules


from multiprocessing import Pool
def pfe(fitness, population):    
    p = Pool()
    return p.map(fitness, population) # runs fitness eval for each individual in parallel

# maybe like that
def pfe2(fitness, population, num_processes):
    l = len(population)
    n = num_processes
    p = Pool(n)
    a = [np.floor(l/n)*(i-1)+1 for i in range(n)]
    b = [np.floor(l/n)*i for i in range(n-1)] + [l]
    return p.map(lambda population: pfe(fitness, population), [population[i:j] for i,j in zip(a,b)]
                
        

            



