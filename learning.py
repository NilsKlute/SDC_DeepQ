import numpy as np
import torch
import torch.nn.functional as F

def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, use_doubleqlearning=False):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """

    if not use_doubleqlearning:
        # Run single Q-learning step

        optimizer.zero_grad()

        # 1. Sample transitions from replay_buffer
        transitions = replay_buffer.sample(batch_size)
        obses_t, actions, rewards, obses_tp1, dones = transitions

        # 2. Compute Q(s_t, a)
        actions = torch.from_numpy(actions).long().to(device)
        q_values_prediction = policy_net(obses_t).gather(1, actions.unsqueeze(1))

        # 3. Compute \max_a Q(s_{t+1}, a) for all next states.
        q_values_target = target_net(obses_tp1).max(dim=1, keepdim=True)[0]

        # 4. Mask next state values where episodes have terminated
        not_terminated = torch.Tensor(dones).to(device) == 0
        q_values_prediction = q_values_prediction[not_terminated]
        q_values_target = q_values_target[not_terminated]
        rewards = torch.from_numpy(rewards).to(device)[not_terminated]

        # 5. Compute the target
        target = rewards + gamma * q_values_target

        # 6. Compute the loss
        loss = 1/torch.sum(not_terminated) * torch.sum(torch.square(target - q_values_prediction))

        # 7. Calculate the gradients
        loss.backward()

        # 8. Clip the gradients
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)

        # 9. Optimize the model
        optimizer.step()

        return loss.item()



    # Tip: You can use use_doubleqlearning to switch the learning modality.

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    target_net.load_state_dict(policy_net.state_dict())
