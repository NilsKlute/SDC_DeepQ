import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def get_state(state): 
    """ Helper function to transform state """ 
    state = np.ascontiguousarray(state, dtype=np.float32) 
    return np.expand_dims(state, axis=0)

def visualize_training(episode_rewards, training_losses, model_identifier, mean_eval_score, ourdir =""):
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
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(np.array(episode_rewards))
    plt.xlabel("Episode Number")
    plt.ylabel("Rewards per Episode")
    plt.title(f"Rewards per Episode over Time. Mean Evaluation Score: {mean_eval_score:.3f}")
    plt.savefig(os.path.join(ourdir, f"episode_rewards-{model_identifier}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(np.array(training_losses))
    plt.xlabel("Time Step")
    plt.ylabel("Loss")
    plt.title("Training Loss per Time Step")
    plt.savefig(os.path.join(ourdir, f"training_losses-{model_identifier}.png"), dpi=300)
    plt.close()


