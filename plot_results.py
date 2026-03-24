import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curves():
    if not os.path.exists("results_scores.npy") or not os.path.exists("results_losses.npy"):
        print("Results files not found. Run training first.")
        return

    scores = np.load("results_scores.npy")
    losses = np.load("results_losses.npy")

    plt.figure(figsize=(12, 5))

    # Score plot
    plt.subplot(1, 2, 1)
    plt.plot(scores, label='Score per Episode')
    # Rolling average
    if len(scores) > 10:
        rolling_avg = np.convolve(scores, np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(scores)), rolling_avg, label='10-Ep Rolling Avg', color='red')
    plt.title("Training Scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("Saved training_results.png")
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()
