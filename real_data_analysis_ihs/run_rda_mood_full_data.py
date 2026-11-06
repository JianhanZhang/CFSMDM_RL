import pandas as pd
import numpy as np
from reader_rda import read_trajectory_from_csv, read_trajectory_from_dataframe
from sklearn.model_selection import train_test_split
from preprocessor_m_qrnn_rda import MSMDMPreprocessorAdd
from preprocessor_baselines_rda import SequentialPreprocessor, UnawarenessPreprocessor, ConcatenatePreprocessor
from preprocessor_flap_rda import FLAP
from preprocessor_ecocf_rda import ECOCF
from agents_rda import FQI, RandomAgent
from evaluation_rda import evaluate_fairness_through_model, evaluate_reward_through_fqe
import torch

def run_exp_one(seed, z_label, methods, is_bootstrap):
    # Step 1
    # 1.2 read data
    impute_data = pd.read_csv('./imputed_ihs_data_combined_avgreward_full_data.csv') # IHS data
    impute_data["msg_type"] = impute_data["msg_type"].replace("none", 0)
    impute_data["msg_type"] = impute_data["msg_type"].replace("mood", 1)
    impute_data["msg_type"] = impute_data["msg_type"].replace("sleep", 1)
    impute_data["msg_type"] = impute_data["msg_type"].replace("step", 1)
    impute_data["Sex"] = impute_data["Sex"].replace(1, 0)
    impute_data["Sex"] = impute_data["Sex"].replace(2, 1)
    impute_data["Child"] = impute_data["Child"].replace(1, 0)
    impute_data["Child"] = impute_data["Child"].replace(2, 1)
    impute_data = impute_data.sort_values(["STUDY_PRTCPT_ID", "week"])
    zs, xs, actions, rewards, ids = read_trajectory_from_dataframe(data=impute_data, 
                                                            z_labels=[z_label], 
                                                            state_labels=["step_count", "sleep_count", "mood"], 
                                                            action_label='msg_type', 
                                                            reward_label='mood', 
                                                            id_label='STUDY_PRTCPT_ID', 
                                                            T=25
                                                            )
    #actions = actions - 1 # actions are indexed starting from 0

    # 1.3 train-test split
    np.random.seed(seed)
    torch.manual_seed(seed)
    (zs_base, zs_test, 
    xs_base, xs_test, 
    actions_base, actions_test, 
    rewards_base, rewards_test, 
    ids_base, ids_test) = train_test_split(zs, xs, actions, rewards, ids, test_size=0.2)

    # split data for preprocessor/fqi training and global model training
    (zs_train, zs_eval, 
    xs_train, xs_eval, 
    actions_train, actions_eval, 
    rewards_train, rewards_eval, 
    ids_train, ids_eval) = train_test_split(zs_base, xs_base, actions_base, rewards_base, ids_base, test_size=0.5)

    # Step 2 policy learning
    agents = []
    for prepro_type in methods:
        if prepro_type == 'cfsmdm_qrnn_kfold':
            qrnnp = MSMDMPreprocessorAdd(
                            initial_model_specs=[], 
                            state_model_specs=[], 
                            reward_model_spec=[], 
                            z_space=np.array([[0], [1]]), 
                            action_space=np.array([[0], [1]]), 
                            max_iter=2000, 
                            is_action_onehot=True, 
                            reg_model='qrnn_joint', 
                            cross_folds=10, 
                            hidden_dims=[128, 128], 
                            learning_rate=1e-4, 
                            batch_size=128, 
                            is_early_stopping=False,
                            epochs=200
                        )
            xs_tilde, rs_tilde = qrnnp.train_preprocessor(
                                    zs=zs_train, 
                                    xs=xs_train, 
                                    actions=actions_train, 
                                    rewards=rewards_train)
            agent = FQI(model_type='nn', action_space=[[0], [1]], hidden_dims=[128, 128], epochs=200, preprocessor=qrnnp, learning_rate=0.001)
            final_fqi_loss = agent.train(zs_train, xs_tilde, actions_train, rs_tilde, max_iter=100, preprocess=False)
        elif prepro_type == 'cfsdp_kfold':
            sp = SequentialPreprocessor(z_space=np.unique(zs_train, axis=0), 
                                        action_space=[[0], [1]], 
                                        cross_folds=10, 
                                        reg_model='nn', 
                                        is_action_onehot=True,
                                        is_normalized=False, 
                                        batch_size=128, 
                                        learning_rate=1e-4, 
                                        hidden_dims=[128, 128], 
                                        epochs=200, 
                                        is_early_stopping=False, 
                                        early_stopping_patience=5, 
                                        early_stopping_min_delta=0.00)
            xs_tilde, rs_tilde = sp.train_preprocessor(zs_train, xs_train, actions_train, rewards_train)
            agent = FQI(model_type='nn', action_space=[[0], [1]], hidden_dims=[128, 128], epochs=200, preprocessor=sp, learning_rate=0.001)
            final_fqi_loss = agent.train(zs_train, xs_tilde, actions_train, rs_tilde, max_iter=100, preprocess=False)
        elif prepro_type == 'flap_m':
            flap = FLAP(z_space=np.unique(zs_train, axis=0), 
                        action_space=[[0], [1]], 
                        mode='m')
            flap.train_preprocessor(zs_train, xs_train, actions_train, rewards_train)
            agent = FQI(model_type='nn', action_space=[[0], [1]], hidden_dims=[128, 128], epochs=200, preprocessor=flap, learning_rate=0.001)
            final_fqi_loss = agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=100, preprocess=True)
        elif prepro_type == 'ecocf_m':
            agent = ECOCF(model_type='nn', z_space=np.unique(zs_train, axis=0), action_space=[[0], [1]], hidden_dims=[128, 128], epochs=200, learning_rate=0.001)
            final_fqi_loss = agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=100, preprocess=True)
        elif prepro_type == 'unaware':
            up = UnawarenessPreprocessor(
                            z_space=np.unique(zs_train, axis=0), 
                            action_space=np.array([[0], [1]])
                        )
            agent = FQI(model_type='nn', action_space=[[0], [1]], hidden_dims=[128, 128], epochs=200, preprocessor=up, learning_rate=0.001)
            final_fqi_loss = agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=100, preprocess=True)
        elif prepro_type == 'full':
            fp = ConcatenatePreprocessor(
                            z_space=np.unique(zs_train, axis=0), 
                            action_space=np.array([[0], [1]])
                        )
            agent = FQI(model_type='nn', action_space=[[0], [1]], hidden_dims=[128, 128], epochs=200, preprocessor=fp, learning_rate=0.001)
            final_fqi_loss = agent.train(zs_train, xs_train, actions_train, rewards_train, max_iter=100, preprocess=True)
        elif prepro_type == 'random':
            agent = RandomAgent(2)
        else:
            print('Undefined preprocessor.')
            exit(1)
        agents.append(agent)



    # Step 3
    # 3.2 train transition kernel
    preprocessor = MSMDMPreprocessorAdd(z_space=np.array([[0], [1]]), 
                                        action_space=np.array([[0], [1]]),
                                        cross_folds=1, 
                                        reg_model='qrnn_joint', 
                                        is_action_onehot=True, 
                                        max_iter=2000,
                                        initial_model_specs=[], 
                                        state_model_specs=[], 
                                        reward_model_spec=[], 
                                        hidden_dims=[128, 128], 
                                        learning_rate=1e-4, 
                                        batch_size=128, 
                                        is_early_stopping=False, 
                                        early_stopping_patience=20, 
                                        epochs=100)
    preprocessor.train_preprocessor(zs=zs_eval, 
                                    xs=xs_eval, 
                                    actions=actions_eval, 
                                    rewards=rewards_eval)
    


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
        value, final_fqe_loss = evaluate_reward_through_fqe(zs_eval, 
                                                            xs_eval, 
                                                            actions_eval, 
                                                            rewards_eval, 
                                                            zs_test, 
                                                            xs_test, 
                                                            actions_test, 
                                                            rewards_test, 
                                                            'nn', 
                                                            agents[i], 
                                                            action_size=2, 
                                                            hidden_dims=[128, 128],
                                                            lr=0.001, 
                                                            max_iter=200, 
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
        for z_label in z_labels: 
            res_add = run_exp_one(seed=start_seed+i, z_label=z_label, methods=methods, 
                                is_bootstrap=is_bootstrap)
            res = pd.concat([res, res_add], axis=0, ignore_index=True)
            if print_res:
                print(res_add)
        i += 1
    if export_res:
        res.to_csv(export_path)



run_exp(rep=50, methods=['cfsmdm_qrnn_kfold', 'cfsdp_kfold', 'random', 'full', 'unaware', 'flap_m', 'ecocf_m'], z_labels=["Sex"], is_bootstrap=False, 
        start_seed=1, print_res=True, export_res=True, 
        export_path='./results/results_rda_mood_full_data.csv')