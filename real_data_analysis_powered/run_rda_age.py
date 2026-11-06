import pandas as pd
import numpy as np
from reader_rda import read_trajectory_from_csv, read_trajectory_from_dataframe
from sklearn.model_selection import train_test_split
from preprocessor_m_rda import MSMDMPreprocessor 
from preprocessor_baselines_rda import SequentialPreprocessor, UnawarenessPreprocessor, ConcatenatePreprocessor
from preprocessor_flap_rda import FLAP
from preprocessor_ecocf_rda import ECOCF
from agents_rda import FQI, RandomAgent
from evaluation_rda import evaluate_fairness_through_model, evaluate_reward_through_fqe
import torch

def run_exp_one(seed, z_label, methods, is_bootstrap):
    # Step 1
    # 1.2 read data
    impute_data = pd.read_csv('./impute_data.csv') # PowerED data
    impute_data["rldecision"] = impute_data["rldecision"].replace(0, 1)
    impute_data["rldecision"] = impute_data["rldecision"].replace(1, 0)
    impute_data["rldecision"] = impute_data["rldecision"].replace(2, 1)
    impute_data["rldecision"] = impute_data["rldecision"].replace(3, 2)
    impute_data["sex_b"] = (impute_data["sex"] == 2).astype(int)
    impute_data["age_b"] = (impute_data["age"] >= 40).astype(int)
    impute_data["education_b"] = (impute_data["education"] == "college").astype(int)
    impute_data["race_b"] = (impute_data["race"] == "white").astype(int)
    impute_data["ethnicity_b"] = (impute_data["ethnicity"] == 2).astype(
            int
        )  # 0: HISPANIC, 1: NOT HISPANIC
    impute_data = impute_data.sort_values(["pt_id", "week"])
    zs, xs, actions, rewards, ids = read_trajectory_from_dataframe(data=impute_data, 
                                                        z_labels=[z_label], 
                                                        state_labels=["weekly_pain_score", "weekly_pain_interference"], 
                                                        action_label='rldecision', 
                                                        reward_label='reward', 
                                                        id_label='pt_id', 
                                                        T=12
                                                        )

    # 1.3 train-test split
    np.random.seed(seed)
    torch.manual_seed(seed)
    (zs_train, zs_test, 
    xs_train, xs_test, 
    actions_train, actions_test, 
    rewards_train, rewards_test, 
    ids_train, ids_test) = train_test_split(zs, xs, actions, rewards, ids, test_size=0.2)

    # Step 2 policy learning
    agents = []
    for prepro_type in methods:
        if prepro_type == 'cfsmdm_kfold':
            specs_init = ['state1 ~ 1 + zs', 'state2 ~ 1 + zs']
            specs_state = [
                'next_state1 ~ 1 + zs + state1 + I(state1**2) + I(state1**3) + state2 + I(state2**2) + I(state2**3) + action1 + action2 + I(zs*state1) + I(zs*state2) + I(zs*action1) + I(zs*action2) + I(state1*action1) + I(state1*action2) + I(state2*action1) + I(state2*action2) + I(state1*state2)', 
                'next_state2 ~ 1 + zs + state1 + I(state1**2) + I(state1**3) + state2 + I(state2**2) + I(state2**3) + action1 + action2 + I(zs*state1) + I(zs*state2) + I(zs*action1) + I(zs*action2) + I(state1*action1) + I(state1*action2) + I(state2*action1) + I(state2*action2) + I(state1*state2)'
            ]
            spec_reward = 'rewards ~ 1 + zs + state1 + I(state1**2) + I(state1**3) + state2 + I(state2**2) + I(state2**3) + action1 + action2 + I(zs*state1) + I(zs*state2) + I(zs*action1) + I(zs*action2) + I(state1*action1) + I(state1*action2) + I(state2*action1) + I(state2*action2) + I(state1*state2)'
            mp = MSMDMPreprocessor(initial_model_specs=specs_init, 
                                   state_model_specs=specs_state, 
                                   reward_model_spec=spec_reward, 
                                   z_space=[[0], [1]], 
                                   action_space=[[0], [1], [2]], 
                                   cross_folds=10,
                                   reg_model='qr', 
                                   is_action_onehot=True, 
                                   max_iter=2000) 
            xs_tilde, rs_tilde=mp.train_preprocessor(
                                  zs=zs_train, 
                                  xs=xs_train, 
                                  actions=actions_train, 
                                  rewards=rewards_train)
            agent = FQI(model_type='nn', action_space=[[0], [1], [2]], hidden_dims=[128, 128], epochs=1000, preprocessor=mp, learning_rate=0.01)
            final_fqi_loss = agent.train(zs_train, xs_tilde, actions_train, rs_tilde, max_iter=100, preprocess=False)
        elif prepro_type == 'cfsdp_kfold':
            sp = SequentialPreprocessor(z_space=np.unique(zs_train, axis=0), 
                                        action_space=[[0], [1], [2]], 
                                        cross_folds=10, 
                                        reg_model='nn', 
                                        is_action_onehot=True,
                                        batch_size=128, 
                                        learning_rate=1e-4, 
                                        hidden_dims=[128, 128], 
                                        epochs=1000, 
                                        is_early_stopping=True, 
                                        early_stopping_patience=5, 
                                        early_stopping_min_delta=0.00)
            xs_tilde, rs_tilde = sp.train_preprocessor(zs_train, xs_train, actions_train, rewards_train)
            agent = FQI(model_type='nn', action_space=[[0], [1], [2]], hidden_dims=[128, 128], epochs=1000, preprocessor=sp, learning_rate=0.01)
            final_fqi_loss = agent.train(zs_train, xs_tilde, actions_train, rs_tilde, max_iter=100, preprocess=False)
        elif prepro_type == 'cfsdp':
            sp = SequentialPreprocessor(z_space=np.unique(zs_train, axis=0), 
                                        action_space=[[0], [1], [2]], 
                                        cross_folds=1, 
                                        reg_model='nn', 
                                        is_action_onehot=True,
                                        batch_size=128, 
                                        learning_rate=1e-4, 
                                        hidden_dims=[128, 128], 
                                        epochs=1000, 
                                        is_early_stopping=True, 
                                        early_stopping_patience=5, 
                                        early_stopping_min_delta=0.00)
            sp.train_preprocessor(zs_train, xs_train, actions_train, rewards_train)
            agent = FQI(model_type='nn', action_space=[[0], [1], [2]], hidden_dims=[128, 128], epochs=1000, preprocessor=sp, learning_rate=0.01)
            final_fqi_loss = agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=100, preprocess=True)
        elif prepro_type == 'flap_m':
            flap = FLAP(z_space=np.unique(zs_train, axis=0), 
                        action_space=[[0], [1], [2]], 
                        mode='m')
            flap.train_preprocessor(zs_train, xs_train, actions_train, rewards_train)
            agent = FQI(model_type='nn', action_space=[[0], [1], [2]], hidden_dims=[128, 128], epochs=1000, preprocessor=flap, learning_rate=0.01)
            final_fqi_loss = agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=100, preprocess=True)
        elif prepro_type == 'ecocf_m':
            agent = ECOCF(model_type='nn', z_space=np.unique(zs_train, axis=0), action_space=[[0], [1], [2]], hidden_dims=[128, 128], epochs=1000, learning_rate=0.01)
            final_fqi_loss = agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=100, preprocess=True)
        elif prepro_type == 'unaware':
            up = UnawarenessPreprocessor(
                            z_space=np.unique(zs_train, axis=0), 
                            action_space=np.array([[0], [1], [2]])
                        )
            agent = FQI(model_type='nn', action_space=[[0], [1], [2]], hidden_dims=[128, 128], epochs=1000, preprocessor=up, learning_rate=0.01)
            final_fqi_loss = agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=100, preprocess=True)
        elif prepro_type == 'full':
            fp = ConcatenatePreprocessor(
                            z_space=np.unique(zs_train, axis=0), 
                            action_space=np.array([[0], [1], [2]])
                        )
            agent = FQI(model_type='nn', action_space=[[0], [1], [2]], hidden_dims=[128, 128], epochs=1000, preprocessor=fp, learning_rate=0.01)
            final_fqi_loss = agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=100, preprocess=True)
        elif prepro_type == 'random':
            agent = RandomAgent(3)
        else:
            print('Undefined preprocessor.')
            exit(1)
        agents.append(agent)



    # Step 3
    # 3.2 train transition kernel
    specs_init = ['state1 ~ 1 + zs', 'state2 ~ 1 + zs']
    specs_state = [
        'next_state1 ~ 1 + zs + state1 + I(state1**2) + I(state1**3) + state2 + I(state2**2) + I(state2**3) + action1 + action2 + I(zs*state1) + I(zs*state2) + I(zs*action1) + I(zs*action2) + I(state1*action1) + I(state1*action2) + I(state2*action1) + I(state2*action2) + I(state1*state2)', 
        'next_state2 ~ 1 + zs + state1 + I(state1**2) + I(state1**3) + state2 + I(state2**2) + I(state2**3) + action1 + action2 + I(zs*state1) + I(zs*state2) + I(zs*action1) + I(zs*action2) + I(state1*action1) + I(state1*action2) + I(state2*action1) + I(state2*action2) + I(state1*state2)'
    ]
    spec_reward = 'rewards ~ 1 + zs + state1 + I(state1**2) + I(state1**3) + state2 + I(state2**2) + I(state2**3) + action1 + action2 + I(zs*state1) + I(zs*state2) + I(zs*action1) + I(zs*action2) + I(state1*action1) + I(state1*action2) + I(state2*action1) + I(state2*action2) + I(state1*state2)'
    preprocessor = MSMDMPreprocessor(z_space=[[0], [1]], 
                                    action_space=[[0], [1], [2]], 
                                    cross_folds=1, 
                                    reg_model='qr', 
                                    is_action_onehot=True, 
                                    max_iter=2000,
                                    initial_model_specs=specs_init, 
                                    state_model_specs=specs_state, 
                                    reward_model_spec=spec_reward)
    preprocessor.train_preprocessor(zs=zs, 
                                    xs=xs, 
                                    actions=actions, 
                                    rewards=rewards)
    


    out = None
    for i, method in enumerate(methods):
        # Step 4 calculate the CF metric
        cf_metric = evaluate_fairness_through_model(preprocessor=preprocessor, 
                                                    zs=zs_test, 
                                                    states=xs_test, 
                                                    actions=actions_test, 
                                                    policy=agents[i], 
                                                    seed=seed)
        
        # Step 5 FQE
        value, final_fqe_loss = evaluate_reward_through_fqe(zs_test, 
                                                            xs_test, 
                                                            actions_test, 
                                                            rewards_test, 
                                                            zs_train, 
                                                            xs_train, 
                                                            actions_train, 
                                                            rewards_train, 
                                                            'nn', 
                                                            agents[i], 
                                                            hidden_dims=[128, 128],
                                                            lr=0.01, 
                                                            max_iter=100, 
                                                            seed=seed)
        
        tmp = pd.DataFrame(
                {
                    "method": [method],
                    "z_label": [z_label],
                    "cf_metric": [cf_metric],
                    "value": [value],
                    "final_fqi_loss": [agents[i].final_fqi_loss], 
                    "final_fqe_loss": [final_fqe_loss], 
                    "seed": [seed],
                }
            )
        out = pd.concat([out, tmp], axis=0, ignore_index=True)

    return out



def run_exp(rep, z_labels, methods, start_seed=1, print_res=True, export_res=False, 
            is_bootstrap=True, export_path='./res.csv'):
    res = None
    i = 0
    while i < rep:
        try:
            for z_label in z_labels: 
                res_add = run_exp_one(seed=start_seed+i, z_label=z_label, methods=methods, 
                                    is_bootstrap=is_bootstrap)
                res = pd.concat([res, res_add], axis=0, ignore_index=True)
                if print_res:
                    print(res_add)
            i += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            i += 1
    if export_res:
        res.to_csv(export_path)



run_exp(rep=50, methods=['random', 'full', 'unaware', 'cfsdp_kfold', 'cfsmdm_kfold', 'flap_m', 'ecocf_m'], z_labels=["age_b"], is_bootstrap=False, 
        start_seed=10, print_res=True, export_res=True, 
        export_path='./results/results_age.csv')