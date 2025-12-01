import numpy as np
import torch
import torch.optim as optim
from action import ActionSet, get_action, select_exploratory_action, select_greedy_action
from learning import perform_qlearning_step, update_target_net
from model import DQN
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from utils import get_state, visualize_training
from evaluate_racing import evaluate
import os
import matplotlib
import time
import signal
import time

def log_sigma_stats(policy_net, timestep):
    """Log sigma parameter statistics for noisy layers."""
    sigma_weights = []
    sigma_biases = []
    
    for name, param in policy_net.named_parameters():
        if 'sigma' in name:
            if 'weight' in name:
                sigma_weights.append(param.data.abs().mean().item())
            elif 'bias' in name:
                sigma_biases.append(param.data.abs().mean().item())
    
    if sigma_weights:
        avg_sigma_w = sum(sigma_weights) / len(sigma_weights)
        avg_sigma_b = sum(sigma_biases) / len(sigma_biases)
        print(f"[{timestep}] Sigma Weight Mean: {avg_sigma_w:.6f}, Sigma Bias Mean: {avg_sigma_b:.6f}")

class StepTimeout(Exception):
    pass

def handler(signum, frame):
    raise StepTimeout()

signal.signal(signal.SIGALRM, handler)

def safe_step(env, env_action, timeout=1):
    signal.alarm(timeout)
    try:
        new_obs, rew, term, trunc, _ = env.step(env_action)
    except StepTimeout:
        print("env.step() timeout; resetting env")
        obs, _ = env.reset()
        return obs, 0.0, True, True, {}
    finally:
        signal.alarm(0)
    return new_obs, rew, term, trunc, _

def learn(env,
          lr=1e-4,
          total_timesteps = 100000,
          buffer_size = 50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          action_repeat=4,
          batch_size=32,
          learning_starts=100,
          gamma=0.99,
          target_network_update_freq=500,
          new_actions = None,
          model_identifier='agent',
          outdir = "",
          use_doubleqlearning = False,
          noisy=False):
    """ Train a deep q-learning model.
    Parameters
    -------
    env: gym.Env
        environment to train on
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to take
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
    action_repeat: int
        selection action on every n-th frame and repeat action for intermediate frames
    batch_size: int
        size of a batched sampled from replay buffer for training
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    model_identifier: string
        identifier of the agent
    """
    buffer_size = 500_000
    target_network_update_freq=4000
    learning_starts=10_000
    use_doubleqlearning = True
    exp_number = 8
    n_step = 3
    train_freq=1
    print ( "buffersize:              {0}".format ( buffer_size ) )
    print ( "target_network_update_freq:              {0}".format (target_network_update_freq ) )
    print ( "n_step:              {0}".format ( n_step ) )
    print ( "train_freq:              {0}".format ( train_freq ) )
    

    # set float as default
    torch.set_default_dtype (torch.float32)
    
    if torch.cuda.is_available():
        print("\nUsing CUDA.")
        print (torch.version.cuda,"\n")
    else:
        print ("\nNot using CUDA.\n")

    

    episode_rewards = [0.0]
    training_losses = []
    action_manager = ActionSet()

    if new_actions is not None:
        print ( "Set new actions")
        action_manager.set_actions(new_actions)

    actions = action_manager.get_action_set()

    action_size = len(actions)
    print ( action_size )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set model subfolder
    if noisy:
        subfolder = f"lr:{lr}_totts:{total_timesteps}_bufsz:{buffer_size}_trainfreq:{train_freq}_actionrep:{action_repeat}_bs:{batch_size}_gamma:{gamma}_tnupdate:{target_network_update_freq}_dd_q:{use_doubleqlearning}_actionsize:{action_size}_n_step:{n_step}_dueling:True_noisy:True_exp_num:{exp_number}"
    else:
        subfolder = f"lr:{lr}_totts:{total_timesteps}_bufsz:{buffer_size}_explfr{exploration_fraction}_explfeps:{exploration_final_eps}_trainfreq:{train_freq}_actionrep:{action_repeat}_bs:{batch_size}_gamma:{gamma}_tnupdate:{target_network_update_freq}_dd_q:{use_doubleqlearning}_actionsize:{action_size}_n_step:{n_step}_dueling:True_noisy:False"
    outdir = os.path.join(outdir, "current_experiments", subfolder)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # Build networks
    policy_net = DQN(action_size, device, noisy).to(device)
    target_net = DQN(action_size, device, noisy).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Create replay buffer
    replay_buffer = ReplayBuffer(buffer_size, gamma, n_step=n_step)

    # Create optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=1.5e-4)

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize environment and get first state
    obs, _ = env.reset()
    obs = get_state(obs)
    start = time.time()
    best_mv_avg_reward = float('-inf')

    is_greedy = True if noisy else False
    
    # Iterate over the total number of time steps
    for t in range(total_timesteps):

        # Select action
        env_action, action_id = get_action ( obs, policy_net, action_size, actions, exploration, t, is_greedy=is_greedy)

        # TODO: if you want to implement the network associated with the continuous action set or the prioritized replay buffer, you need to reimplement the replay buffer.

        # Perform action fram_skip-times
        for f in range(action_repeat):
            new_obs, rew, term, trunc, _ = safe_step(env, env_action)
            done = term or trunc
            episode_rewards[-1] += rew
            if done:
                break
            
        # Store transition in the replay buffer.
        new_obs = get_state(new_obs)
        replay_buffer.add(obs, action_id, rew, new_obs, float(done))
        obs = new_obs

        if done:
            # Start new episode after previous episode has terminated
            print("timestep: " + str(t) + " \t reward: " + str(episode_rewards[-1]))
            obs, _ = env.reset()
            obs = get_state(obs)
            episode_rewards.append(0.0)

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            loss = perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, t, use_doubleqlearning)
            training_losses.append(loss)

            # Log sigma stats every 1000 steps if using noisy net
            if noisy and t % 1000 == 0:
                log_sigma_stats(policy_net, t)
            

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target_net(policy_net, target_net)

        if t % 1000 == 0:
            end = time.time()
            print(f"\n** {t} th timestep - {end - start:.5f} sec passed**\n")

        mv_avg_reward = sum(episode_rewards[-10:]) / 10
        # Save the trained policy network
        if len(episode_rewards) >= 10 and mv_avg_reward > best_mv_avg_reward:
            torch.save(policy_net.state_dict(), os.path.join(outdir, model_identifier + '.pth'))
            best_mv_avg_reward = mv_avg_reward
            print(f"Saved model with moving average reward: {mv_avg_reward}")

    end = time.time()
    print(f"\n** Total {end - start:.5f} sec passed**\n")


    mean_eval_score = evaluate(env, new_actions, load_path=os.path.join(outdir, model_identifier + '.pth'))
    # Visualize the training loss and cumulative reward curves
    visualize_training(episode_rewards, training_losses, model_identifier, mean_eval_score, outdir)
 
