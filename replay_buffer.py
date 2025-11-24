import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size, gamma, n_step):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._gamma = gamma
        self._n_step = n_step

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """ Add a transition to replay memory. 
        Parameters
        ----------
        obs_t: 
            State s_t
        action: 
            Action a_t taken in s_t
        reward: 
            Received reward r_t
        obs_tp1: 
            Follow-up state s_{t+1}
        done: bool
            Whether episode has terminated at s_{t+1}
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data

            returns = 0
            gamma_step = 1
            for step in range(self._n_step):
                if (i + step) % len(self._storage) < (i + step):
                    done = True
                    break

                transition = self._storage[i + step]
                _, _, reward, obs_tp1, done = transition

                returns += reward * gamma_step

                if done:
                    break
                gamma_step *= self._gamma

            obses_t.append(obs_t)
            actions.append(np.array(action, copy=False))
            rewards.append(returns)
            obses_tp1.append(obs_tp1)
            dones.append(done)

        return np.squeeze(np.array(obses_t), axis=1), np.array(actions), np.array(rewards, dtype=np.float32), np.squeeze(np.array(obses_tp1), axis=1), np.array(dones, dtype=np.float32)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)