import numpy as np

# for now: S is a np.ndarray of dim=1 and fixed length with real entries


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
            while vin < min and vin > max:
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

            
        