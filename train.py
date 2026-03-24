import numpy as np
import torch
from env.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
from config import *
import os

def train():
    env = SnakeGame(render_mode=False)
    # State dimension: 11 based on snake_game.py implementation
    # head(2), food_rel(2), gravity(2), length(1), danger(4)
    state_dim = 11
    action_dim = 4
    
    agent = DQNAgent(state_dim, action_dim)
    
    num_episodes = 500
    scores = []
    losses = []
    
    print(f"Starting training on {DEVICE}...")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            agent.memory.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            loss = agent.train_step()
            if loss:
                losses.append(loss)
            
            if agent.steps_done % TARGET_SYNC == 0:
                agent.update_target_net()
                
            agent.steps_done += 1
            
        scores.append(env.score)
        
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f}")

    # Save results
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(agent.policy_net.state_dict(), "models/snake_dqn.pth")
    np.save("results_scores.npy", np.array(scores))
    np.save("results_losses.npy", np.array(losses))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
