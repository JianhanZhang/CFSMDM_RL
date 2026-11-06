import pandas as pd
import numpy as np

def _read_trajectory_helper(data, z_labels, state_labels, action_label, reward_label, id_label, T):
    # read some basic information
    ids = data[id_label].unique()
    N = ids.shape[0]
    T = T

    # read trajectory data
    z_labels = np.array(z_labels)
    state_labels = np.array(state_labels)
    zs = np.zeros((N, z_labels.shape[0]))
    xs = np.zeros((N, T, state_labels.shape[0]))
    actions = np.zeros((N, T-1))
    rewards = np.zeros((N, T-1))

    for i, id in enumerate(ids):
        subdata = data[data[id_label] == id].iloc[0:T]
        for k in range(z_labels.shape[0]): # populate sensitive attributes
            zs[i][k] = subdata[z_labels[k]].iat[0]
        for k in range(state_labels.shape[0]): # populate states at t = 0
            xs[i][0][k] = subdata[state_labels[k]].iat[0]
        for t in range(1, T):
            actions[i][t-1] = subdata[action_label].iat[t] # populate actions
            rewards[i][t-1] = subdata[reward_label].iat[t] # populate rewards
            for k in range(state_labels.shape[0]): # populate states
                xs[i][t][k] = subdata[state_labels[k]].iat[t]
    
    return zs, xs, actions, rewards, ids



def read_trajectory_from_csv(path, z_labels, state_labels, action_label, reward_label, id_label, T):
    # import trajectory data
    data = pd.read_csv(path)

    # read trajectory data
    return _read_trajectory_helper(data, z_labels, state_labels, action_label, reward_label, 
                                        id_label, T)



def read_trajectory_from_dataframe(data, z_labels, state_labels, action_label, 
                                   reward_label, id_label, T):
    # read trajectory data
    return _read_trajectory_helper(data, z_labels, state_labels, action_label, reward_label, 
                                        id_label, T)



def convert_trajectory_to_dataframe(zs, states, actions, rewards, ids, z_labels=None, 
                                    state_labels=None, action_label=None, reward_label=None, 
                                    id_label=None, T_label=None):
    N = states.shape[0]
    T = states.shape[1]
    zs = np.array(zs)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    ids = np.array(ids)

    # define labels of the dataframe
    if z_labels is None:
        z_labels = []
        for i in range(zs.shape[-1]):
            z_labels.append('z' + str(i+1)) # default naming: z1, z2, ...
        z_labels = np.array(z_labels)
    else:
        z_labels = np.array(z_labels)

    if state_labels is None:
        state_labels = []
        for i in range(states.shape[-1]):
            state_labels.append('state' + str(i+1)) # default naming: state1, state2, ...
        state_labels = np.array(state_labels)
    else: 
        state_labels = np.array(state_labels)

    if action_label is None:
        action_label = 'action'

    if reward_label is None:
        reward_label = 'reward'

    if id_label is None:
        id_label = 'ID'

    if T_label is None:
        T_label = 'timestamp'

    # reorganize data
    ids_r = np.repeat(ids, repeats=T).reshape(-1, 1)
    times_r = np.tile(np.arange(1, T+1), reps=N).reshape(-1, 1)
    zs_r = np.repeat(zs, repeats = T, axis=0)
    states_r = states.reshape(N*T, states.shape[-1])
    actions_r = np.concatenate((actions, np.tile(np.array([[np.nan]]), reps=(N, 1))), axis=1)
    actions_r = actions_r.reshape(N*T, 1)
    rewards_r = np.concatenate((rewards, np.tile(np.array([[np.nan]]), reps=(N, 1))), axis=1)
    rewards_r = rewards_r.reshape(N*T, 1)
    data_arr = np.concatenate((ids_r, times_r, zs_r, actions_r, rewards_r, states_r), axis=1)

    # create the trajectory dataframe
    labels = np.concatenate((np.array([id_label]), 
                                np.array([T_label]), 
                                z_labels, 
                                np.array([action_label]), 
                                np.array([reward_label]), 
                                state_labels), 
                                axis=0)
    data = pd.DataFrame(data_arr, columns=labels)

    return data



def export_trajectory_to_csv(path, zs, states, actions, rewards, ids, z_labels=None, 
                             state_labels=None, action_label=None, reward_label=None, 
                             id_label=None, T_label=None, **to_csv_kwargs):
    data = convert_trajectory_to_dataframe(zs, states, actions, rewards, ids, z_labels, state_labels, 
                                           action_label, reward_label, id_label, T_label)
    data.to_csv(path, **to_csv_kwargs)





