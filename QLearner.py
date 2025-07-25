	   		 	   		  		  		    	 		 		   		 		  


import random as rand
import numpy as np


class QLearner(object):


    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Q = np.zeros((num_states, num_actions))
        self.experience = []
        self.s = 0
        self.a = 0

    def querysetstate(self, s):
        self.s = s
        if rand.uniform(0, 1) < self.rar:
            act = rand.randint(0, self.num_actions - 1)
        else:
            act = np.argmax(self.Q[s])
        if self.verbose:
            print(f"s = {s}, a = {act}")
        return act
    def query(self, s_prime, r):

        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (
                    r + self.gamma * np.max(self.Q[s_prime]))
        self.experience.append((self.s, self.a, s_prime, r))
        for _ in range(self.dyna):
            s_dyna, a_dyna, s_prime_dyna, r_dyna = rand.choice(self.experience)
            self.Q[s_dyna, a_dyna] = (1 - self.alpha) * self.Q[s_dyna, a_dyna] + self.alpha * (
                        r_dyna + self.gamma * np.max(self.Q[s_prime_dyna]))
        self.s = s_prime
        if rand.uniform(0, 1) < self.rar:
            act = rand.randint(0, self.num_actions - 1)
        else:
            act = np.argmax(self.Q[s_prime])
        self.rar *= self.radr
        self.a = act
        if self.verbose:
            print(f"s = {s_prime}, a = {act}, r = {r}")
        return act
    def author(self):
        return 'pvenieris3'  # replace tb34 with your Georgia Tech username.


if __name__ == "__main__":
    learner = QLearner(num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=200, verbose=True)
    initial_location = 0
    goal_location = 99
    quicksand_location = 50
    s = initial_location
    a = learner.querysetstate(s)
    for epoch in range(50):
        converged = False
        while not converged:
            if s == goal_location:
                r = 1.0
                s_prime = initial_location
                converged = True
            elif s == quicksand_location:
                r = -100.0
                s_prime = initial_location
            else:
                r = -1.0
                s_prime = (s + a) % learner.num_states  # Example state transition
            a = learner.query(s_prime, r)
            s = s_prime
