import numpy as np
import pickle
import d3rlpy

def get_rl_action_and_state(state):
    time_step = state['age'][1][0] - state['age'][0][0]
    abs_u = (state['lat_m'][1:] - state['lat_m'][:-1])/time_step
    abs_v = (state['lon_m'][1:] - state['lon_m'][:-1])/time_step
    abs_w = (state['d2b'][1:] - state['d2b'][:-1])/time_step
    action = {'u': abs_u - (state['u'][:-1]+state['u'][1:])/2,
            'v': abs_v - (state['v'][:-1]+state['v'][1:])/2,
            'w': abs_w - (state['w'][:-1]+state['w'][1:])/2
            }

    n_particle = state['lat_m'].shape[1]
    rl_state = {
        "age": state['age'],
        "depth": state['depth'],
        "d2b": state['d2b'],
        "u": state['u'],
        "v": state['v'],
        "w": state['w'],
        "temp": state['temp'],
        
        "last_action_u": np.vstack([np.zeros(n_particle), action['u']]),
        "last_action_v": np.vstack([np.zeros(n_particle), action['v']]),
        "last_action_w": np.vstack([np.zeros(n_particle), action['w']]),
    }
    # some information from the last state so that the RL agent can sense the "delta"
    for key in ['depth', "d2b", "temp"]:
        rl_state['last_' + key] = np.vstack([state[key][0], state[key][:-1]])
    rl_state_arr = np.array([rl_state[key] for key in rl_state.keys()]).transpose(1,2, 0)
    rl_state_arr = rl_state_arr[:-1] #remove the last one as it doesn't have an action or reward
    rl_action_arr = np.array([action[key] for key in action.keys()]).transpose(1,2, 0)
    return rl_state_arr, rl_action_arr 

def reward_function(age, ocean_depth, d2b, temp):
    #reward: 26-30到达终点
    #1： 海底深度范围：z[0] in [-15, -64]
    #2： 粒子到海底距离：|depth - z[0] | < 5
    #3： 粒子所在位子温度： 21 < temp < 25
    age_d = age/60/60/24
    is_proper_spot = (-64<=ocean_depth<=-15 and d2b<=5 and 21<=temp<=25)
    if 26<=age_d<=30 and is_proper_spot:
        return 1
    if age_d >= 30:
        return -1
    return 0

def get_rl_reward(state):
    all_rewards = []
    for i_t in range(1, state['age'].shape[0]): # skip the first one as the reward of the first action is derived from the second state.
        rewards = []
        for i_p in range(state['age'].shape[1]):
            rewards.append(reward_function(state['age'][i_t, i_p], state['depth'][i_t, i_p], state['d2b'][i_t, i_p], state['temp'][i_t, i_p]))
        all_rewards.append(np.array(rewards))
    all_rewards_arr = np.array(all_rewards)
    return all_rewards_arr

def convert_to_mdp(rl_state, rl_action, rl_reward, rl_terminals):
    state_mdp = []
    action_mdp = []
    reward_mdp = []
    terminal_mdp = []
    for i_p in range(rl_state.shape[1]):
        idx = np.where(rl_terminals[:, i_p] == 1.0)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        state_mdp.append(rl_state[:idx+1, i_p])
        action_mdp.append(rl_action[:idx+1, i_p])
        reward_mdp.append(rl_reward[:idx+1, i_p])
        terminal_mdp.append(rl_terminals[:idx+1, i_p])
    return np.vstack(state_mdp), np.vstack(action_mdp), np.concatenate(reward_mdp), np.concatenate(terminal_mdp)

if __name__ == "__main__":
    #dataset_cartpole, env = d3rlpy.datasets.get_cartpole()
    with open("processed_data/crocobuoy_DanceBooker_20160821.pkl", "rb") as f:
        state = pickle.load(f)

    rl_state, rl_action = get_rl_action_and_state(state)
    rl_reward = get_rl_reward(state)
    rl_terminals = ((rl_reward==1) | (rl_reward==-1)).astype(float) # when reward =1 or -1, the episode ends

    state_mdp, action_mdp, reward_mdp, terminal_mdp = convert_to_mdp(rl_state, rl_action, rl_reward, rl_terminals)

    dataset = d3rlpy.dataset.MDPDataset(
    observations=state_mdp,
    actions=action_mdp,
    rewards=reward_mdp,
    terminals=terminal_mdp
    )
    rl_model = d3rlpy.algos.DDPGConfig().create()
    rl_model.build_with_dataset(dataset)
    rl_model.fit(dataset, n_steps=1000000)
    rl_model.save_model("rl_model.pkl")



