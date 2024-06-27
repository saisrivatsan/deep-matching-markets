import os
import time
import pickle
import random
import logging
import argparse
from typing import List

import numpy as np

import baselines
from data import Data
from utils import init_logger


class Args: 
    num_agents: int = 4
    """ Number of agents """

    prob: float = 0.20
    """ Probability of truncation """

    corr: float = 0.00
    """ Correlation Probability: To be filled later """

    batch_size: int = 256
    """ Batch size """
    
    num_tst_samples: int = 20480
    """ Number of Test batches """
    
    num_tst_batches = num_tst_samples//batch_size

    seed: int = 42
    """ Random Seed """
    
# Loss functions numpy
def STABILITY_VIOLATION_BATCH(P, Q, R):
    WP = np.maximum(P[:, :, np.newaxis, :] - P[:, :, :, np.newaxis], 0)
    WQ = np.maximum(Q[:, :, np.newaxis, :] - Q[:, np.newaxis, :, :], 0)   
       
    T = (1 - np.sum(R, axis = 1, keepdims = True))
    S = (1 - np.sum(R, axis = 2, keepdims = True))
    
    RGT_1 = np.einsum('bjc,bijc->bic', R, WQ) + T * np.maximum(Q, 0)
    RGT_2 = np.einsum('bia,biac->bic', R, WP) + S * np.maximum(P, 0)
    
    REGRET =  RGT_1 * RGT_2 
    
    return REGRET.sum(-1).mean()

def IR_VIOLATION_BATCH(P, Q, R):
    IR_1 = R * np.maximum(-Q, 0)
    IR_2 = R * np.maximum(-P, 0)
    IR = IR_1 + IR_2
    return IR.sum(-1).mean()

def IC_FOSD_VIOLATION_BATCH(P, Q, R, mechanism):
    
    batch_size = P.shape[0]
    num_agents = P.shape[1]
    
    IC_viol_P = np.zeros(num_agents)
    IC_viol_Q = np.zeros(num_agents)
    
    

    for agent_idx in range(num_agents):
        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = True, include_truncation = True)
        R_mis = mechanism(P_mis.reshape(-1, num_agents, num_agents),
                                 Q_mis.reshape(-1, num_agents, num_agents))
        R_mis = R_mis.reshape(batch_size, -1, num_agents, num_agents)

        
        R_diff = (R_mis[:, :, agent_idx, :] - R[:, None, agent_idx, :])*(P[:, None, agent_idx, :] > 0)
        IDX = np.argsort(-P[:, agent_idx, :])
        IDX = np.tile(IDX[:, None, :], (1, R_mis.shape[1], 1))
        
        FOSD_viol = np.cumsum(np.take_along_axis(R_diff, IDX, axis=-1), -1)
        IC_viol_P[agent_idx] = np.maximum(FOSD_viol, 0).max(-1).max(-1).mean(-1)
        
        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = False, include_truncation = True)
        R_mis = mechanism(P_mis.reshape(-1, num_agents, num_agents),
                                 Q_mis.reshape(-1, num_agents, num_agents))
        R_mis = R_mis.reshape(batch_size, -1, num_agents, num_agents)

        R_diff = (R_mis[:, :, :, agent_idx] - R[:, None, :, agent_idx])*(Q[:, None, :, agent_idx] > 0)
        IDX = np.argsort(-Q[:, :, agent_idx])
        IDX = np.tile(IDX[:, None, :], (1, R_mis.shape[1], 1))
        
        FOSD_viol = np.cumsum(np.take_along_axis(R_diff, IDX, axis=-1), -1)
        IC_viol_Q[agent_idx] = np.maximum(FOSD_viol, 0).max(-1).max(-1).mean(-1)
        
    IC_viol = (IC_viol_P.mean() + IC_viol_Q.mean())*0.5
    return IC_viol


def WELFARE_BATCH(P, Q, R):
    VAL_WF = ((P*R).sum(-1).mean() + (Q*R).sum(-2).mean())/2
    return VAL_WF

def compute_violations(mech):
    np.random.seed(args.seed)
    random.seed(args.seed)
    VAL_ST_LOSS = 0.0
    VAL_IC_LOSS = 0.0 
    VAL_WF = 0.0

    for j in range(args.num_tst_batches):
        P, Q = G.generate_batch(args.batch_size)
        R = mech(P, Q)
        
        ST_LOSS = STABILITY_VIOLATION_BATCH(P, Q, R) + IR_VIOLATION_BATCH(P, Q, R)
        IC_LOSS = IC_FOSD_VIOLATION_BATCH(P, Q, R, mech)
        WF = WELFARE_BATCH(P, Q, R)
        
        VAL_ST_LOSS += ST_LOSS
        VAL_IC_LOSS += IC_LOSS
        VAL_WF += WF
        
        
    VAL_ST_LOSS = VAL_ST_LOSS/args.num_tst_batches
    VAL_IC_LOSS = VAL_IC_LOSS/args.num_tst_batches
    VAL_WF = VAL_WF/args.num_tst_batches
    return VAL_ST_LOSS, VAL_IC_LOSS, VAL_WF

def STABILITY_VIOLATION_BATCH_COMBO(P, Q, mech_1, mech_2, lams):
    ST_VIOL = np.zeros(len(lams))
    R_1 = mech_1(P, Q)
    R_2 = mech_2(P, Q)
    for idx, lam in enumerate(lams):        
        R = lam * R_1 + (1 - lam) * R_2
        ST_VIOL[idx] = STABILITY_VIOLATION_BATCH(P, Q, R)
        
    return ST_VIOL

def IR_VIOLATION_BATCH_COMBO(P, Q, mech_1, mech_2, lams):
    IR_VIOL = np.zeros(len(lams))
    R_1 = mech_1(P, Q)
    R_2 = mech_2(P, Q)
    for idx, lam in enumerate(lams):        
        R = lam * R_1 + (1 - lam) * R_2
        IR_VIOL[idx] = IR_VIOLATION_BATCH(P, Q, R)
    return IR_VIOL


def IC_VIOLATION_BATCH_COMBO(P, Q, mech_1, mech_2, lams):
    
    batch_size = P.shape[0]
    num_agents = P.shape[1]
    
    R_1 = mech_1(P, Q)
    R_2 = mech_2(P, Q)
    
    IC_viol_P = np.zeros(len(lams))
    IC_viol_Q = np.zeros(len(lams))
    
    for agent_idx in range(num_agents):
        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = True, include_truncation = True)
        
        R_mis_1 = mech_1(P_mis.reshape(-1, num_agents, num_agents),
                                 Q_mis.reshape(-1, num_agents, num_agents))
        R_mis_1 = R_mis_1.reshape(batch_size, -1, num_agents, num_agents)
        
        R_mis_2 = mech_2(P_mis.reshape(-1, num_agents, num_agents),
                                 Q_mis.reshape(-1, num_agents, num_agents))
        R_mis_2 = R_mis_2.reshape(batch_size, -1, num_agents, num_agents)

        
        for idx, lam in enumerate(lams):
            
            R = lam * R_1 + (1 - lam) * R_2
            R_mis = lam * R_mis_1 + (1 - lam) * R_mis_2
            
            R_diff = (R_mis[:, :, agent_idx, :] - R[:, None, agent_idx, :])*(P[:, None, agent_idx, :] > 0)
            IDX = np.argsort(-P[:, agent_idx, :])
            IDX = np.tile(IDX[:, None, :], (1, R_mis.shape[1], 1))
        
            FOSD_viol = np.cumsum(np.take_along_axis(R_diff, IDX, axis=-1), -1)
            IC_viol_P[idx]  += np.maximum(FOSD_viol, 0).max(-1).max(-1).mean(-1)
            
            
        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx = agent_idx, is_P = False, include_truncation = True)
        
        R_mis_1 = mech_1(P_mis.reshape(-1, num_agents, num_agents),
                                 Q_mis.reshape(-1, num_agents, num_agents))
        R_mis_1 = R_mis_1.reshape(batch_size, -1, num_agents, num_agents)
        
        R_mis_2 = mech_2(P_mis.reshape(-1, num_agents, num_agents),
                                 Q_mis.reshape(-1, num_agents, num_agents))
        R_mis_2 = R_mis_2.reshape(batch_size, -1, num_agents, num_agents)

        for idx, lam in enumerate(lams):
            R = lam * R_1 + (1 - lam) * R_2
            R_mis = lam * R_mis_1 + (1 - lam) * R_mis_2
            
            R_diff = (R_mis[:, :, :, agent_idx] - R[:, None, :, agent_idx])*(Q[:, None, :, agent_idx] > 0)
            IDX = np.argsort(-Q[:, :, agent_idx])
            IDX = np.tile(IDX[:, None, :], (1, R_mis.shape[1], 1))
        
            FOSD_viol = np.cumsum(np.take_along_axis(R_diff, IDX, axis=-1), -1)
            IC_viol_Q[idx] += np.maximum(FOSD_viol, 0).max(-1).max(-1).mean(-1)
        
    IC_viol = (IC_viol_P + IC_viol_Q)/(2*num_agents)
    return IC_viol

def compute_combination_violations(mech_1, mech_2, lams):
    random.seed(args.seed)
    np.random.seed(args.seed)
    

    VAL_ST_LOSS = np.zeros(len(lams))
    VAL_IC_LOSS = np.zeros(len(lams))

    for j in range(args.num_tst_batches):
        P, Q = G.generate_batch(args.batch_size)

        ST_LOSS = STABILITY_VIOLATION_BATCH_COMBO(P, Q, mech_1, mech_2, lams) + IR_VIOLATION_BATCH_COMBO(P, Q, mech_1, mech_2, lams)
        IC_LOSS = IC_VIOLATION_BATCH_COMBO(P, Q, mech_1, mech_2, lams)

        VAL_ST_LOSS += ST_LOSS
        VAL_IC_LOSS += IC_LOSS

    ST_arr = VAL_ST_LOSS/args.num_tst_batches
    IC_arr = VAL_IC_LOSS/args.num_tst_batches
    
    return ST_arr, IC_arr

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n', '--num_agents', action='store',
                    dest='num_agents', required=True, type=int,
                    help='Num Agents')
    
    parser.add_argument('-p', '--prob', action='store',
                    dest='prob', required=True, type=float,
                    help='Truncation Probability')
    
    parser.add_argument('-c', '--corr', action='store',
                    dest='corr', required=True, type=float,
                    help='Correlation Probability')
    
    cmd_args = parser.parse_args()
        
    args = Args()
    args.num_agents = cmd_args.num_agents
    args.prob = cmd_args.prob
    args.corr = cmd_args.corr
    

    """ Loggers """
    root_dir =  os.path.join("experiments", "agents_4",  "corr_%.2f"%(args.corr), "baselines") 
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    log_fname = os.path.join(root_dir, "LOG")
    logger = init_logger(log_fname)


    """ Seed for reproducibility """
    random.seed(args.seed)
    np.random.seed(args.seed)

    G = Data(args.num_agents, args.prob, args.corr) 

    mech = baselines.compute_DA_batch
    s1, i1, w1 = compute_violations(mech)
    mech = baselines.compute_DA_batch_switch
    s2, i2, w2 = compute_violations(mech)
    DA_ST, DA_IC, DA_WF = max(s1, s2), max(i1, i2), max(w1, w2)
    logger.info("[Mechanism]: DA, [ST-Loss]: %f, [IC-Loss]: %f, [WF]: %f"%(DA_ST, DA_IC, DA_WF))

    mech = baselines.compute_TTC_batch
    s1, i1, w1 = compute_violations(mech)
    mech = baselines.compute_TTC_batch_switch
    s2, i2, w2 = compute_violations(mech)
    TTC_ST, TTC_IC, TTC_WF = max(s1, s2), max(i1, i2), max(w1, w2)
    logger.info("[Mechanism]: TTC, [ST-Loss]: %f, [IC-Loss]: %f, [WF]: %f"%(TTC_ST, TTC_IC, TTC_WF))

    mech = baselines.compute_one_RSD_batch
    s1, i1, w1 = compute_violations(mech)
    mech = baselines.compute_one_RSD_batch
    s2, i2, w2 = compute_violations(mech)
    RSD_ST, RSD_IC, RSD_WF = max(s1, s2), max(i1, i2), max(w1, w2)
    logger.info("[Mechanism]: RSD, [ST-Loss]: %f, [IC-Loss]: %f, [WF]: %f"%(RSD_ST, RSD_IC, RSD_WF))

    # Choose whichever side performed better 
    lams = np.linspace(0, 1, 11)
    mech_1 = baselines.compute_one_RSD_batch
    mech_2 = baselines.compute_TTC_batch
    mech_3 = baselines.compute_DA_batch
    ST_1, IC_1 = compute_combination_violations(mech_1, mech_2, lams)
    ST_2, IC_2 = compute_combination_violations(mech_2, mech_3, lams)

    data = dict()
    data['DA_ST'] = DA_ST
    data['DA_IC'] = DA_IC
    data['DA_WF'] = DA_WF

    data['RSD_ST'] = RSD_ST
    data['RSD_IC'] = RSD_IC
    data['RSD_WF'] = RSD_WF

    data['TTC_ST'] = TTC_ST
    data['TTC_IC'] = TTC_IC
    data['TTC_WF'] = TTC_WF

    data['COMBO_RSD_TTC_ST'] = ST_1
    data['COMBO_RSD_TTC_IC'] = IC_1

    data['COMBO_TTC_DA_ST'] = ST_2
    data['COMBO_TTC_DA_IC'] = IC_2

    with open(os.path.join(root_dir, "data.p"), 'wb') as f:
        pickle.dump(data, f)