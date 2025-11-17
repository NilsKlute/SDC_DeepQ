import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def get_state(state): 
    """ Helper function to transform state """ 
    state = np.ascontiguousarray(state, dtype=np.float32) 
    return np.expand_dims(state, axis=0)

def visualize_training(episode_rewards, training_losses, model_identifier, ourdir =""):
    """ Visualize training by creating reward + loss plots
    Parameters
    -------
    episode_rewards: list
        list of cumulative rewards per training episode
    training_losses: list
        list of training losses
    model_identifier: string
        identifier of the agent
    """
    print(episode_rewards)
    plt.plot(np.array(episode_rewards))
    plt.xlabel("Episode Number")
    plt.ylabel("Rewards per Episode")
    plt.title("Rewards per Episode over Time")
    plt.savefig( os.path.join (ourdir, "episode_rewards-"+model_identifier+".png"))
    plt.close()
    plt.plot(np.array(training_losses))
    plt.xlabel("Time Step")
    plt.ylabel("Loss")
    plt.title("Training Loss per Time Step")
    plt.savefig( os.path.join (ourdir,"training_losses-"+model_identifier+".png"))
    plt.close()

