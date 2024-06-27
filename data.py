import itertools
import numpy as np
from numba import jit

@jit(nopython=True)
def generate_permutation_array(N, num_agents):
    P = np.zeros((N, num_agents))
    for i in range(N): P[i] = np.random.permutation(num_agents)
    return P
    

class Data(object):    
    def __init__(self, num_agents, prob = 0.20, corr = 0.0): 
        self.num_agents = num_agents
        self.prob = prob
        self.corr = corr
                
    def sample_ranking(self, N):   
        
        N_trunc = int(N * self.prob)
        P = generate_permutation_array(N, self.num_agents) + 1
               
        if N_trunc > 0:
            
            # Choose indices to truncate
            idx = np.random.choice(N, N_trunc, replace = False)
            
            # Choose a position to truncate
            trunc = np.random.randint(self.num_agents, size = N_trunc)
            
            # Normalize so preference to remain single has 0 payoff
            swap_vals = P[idx, trunc]
            P[idx, trunc] = 0
            P[idx] = P[idx] - swap_vals[:, np.newaxis]
        
        return P/self.num_agents
    
    def generate_all_ranking(self, include_truncation = True):    
        if include_truncation is False:
            M = np.array(list(itertools.permutations(np.arange(self.num_agents)))) + 1.0
        else:
            M = np.array(list(itertools.permutations(np.arange(self.num_agents + 1))))
            M = (M - M[:, -1:])[:, :-1]
            
        return M/self.num_agents
    
    def generate_all_misreports(self, P, Q, agent_idx, is_P, include_truncation = True):        
        M = self.generate_all_ranking(include_truncation = include_truncation)
        num_misreports = M.shape[-2]
        P_mis = np.tile(P[:, None, :, :], [1, num_misreports, 1, 1])
        Q_mis = np.tile(Q[:, None, :, :], [1, num_misreports, 1, 1])
        
        if is_P: P_mis[:, :, agent_idx, :] = M
        else: Q_mis[:, :, :, agent_idx] = M
        
        return P_mis, Q_mis
    
        
    def generate_batch(self, batch_size):

        N = batch_size * self.num_agents
        
        P = self.sample_ranking(N).reshape(-1, self.num_agents, self.num_agents)   
        Q = self.sample_ranking(N).reshape(-1, self.num_agents, self.num_agents)
                
        if self.corr > 0:
            P_common = self.sample_ranking(batch_size)[:, None, :]
            Q_common = self.sample_ranking(batch_size)[:, None, :]
        
            P_idx = np.random.binomial(1, self.corr, [batch_size, self.num_agents, 1])
            Q_idx = np.random.binomial(1, self.corr, [batch_size, self.num_agents, 1])
        
            P = P * (1 - P_idx) + P_common * P_idx
            Q = Q * (1 - Q_idx) + Q_common * Q_idx
        
        Q = Q.transpose(0, 2, 1)        
        return P, Q
            


    