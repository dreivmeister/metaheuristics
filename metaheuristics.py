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
def abstract_generational_ea(InitialPopulation, AssesFitness, Fitness, Breed, Join, optimal, maxiter):
    P = InitialPopulation()
    Best = None
    for _ in range(maxiter):
        P_fitness = AssesFitness(P) # fitness for each individual
        for i in range(len(P)):
            if Best is None or P_fitness[i] > Fitness(Best):
                Best = P[i]
                if Fitness(Best) >= optimal:
                    return Best
        P = Join(P, Breed(P))
    return Best



def mu_lambda_evolution_strategy(mu, lam, Choose_mu, InitialPopulation, AssesFitness, Fitness, Copy, Mutate, optimal, maxiter):
    # mu - number of parents selected
    # lam - number of children generated
    
    P = InitialPopulation(lam)
    Best = None
    for _ in range(maxiter):
        P_fitness = AssesFitness(P) # fitness for each individual
        for i in range(len(P)):
            if Best is None or P_fitness[i] > Fitness(Best):
                Best = P[i]
                if Fitness(Best) >= optimal:
                    return Best
        
        Q = Choose_mu(mu, P, P_fitness)
        P = []
        for i in range(len(Q)):
            for _ in range(int(lam/mu)):
                P.append(Mutate(Copy(Q[i])))
    return Best
    
def mu_plus_lambda_evolution_strategy(mu, lam, Choose_mu, InitialPopulation, AssesFitness, Fitness, Copy, Mutate, optimal, maxiter):
    # mu - number of parents selected
    # lam - number of children generated
    
    P = InitialPopulation(lam)
    Best = None
    for _ in range(maxiter):
        P_fitness = AssesFitness(P) # fitness for each individual
        for i in range(len(P)):
            if Best is None or P_fitness[i] > Fitness(Best):
                Best = P[i]
                if Fitness(Best) >= optimal:
                    return Best
        
        Q = Choose_mu(mu, P, P_fitness) # choose mu best individuals
        P = Q
        for i in range(len(Q)):
            for _ in range(int(lam/mu)):
                P.append(Mutate(Copy(Q[i])))
    return Best
    
    
def genetic_algorithm(popsize, InitialPopulation, AssesFitness, Fitness, SelectWithReplacement, Crossover, Copy, Mutate, optimal, maxiter):
    
    P = InitialPopulation(popsize)
    Best = None
    for _ in range(maxiter):
        P_fitness = AssesFitness(P) # fitness for each individual
        for i in range(len(P)):
            if Best is None or P_fitness[i] > Fitness(Best):
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
            
    