import torch
from env.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
from config import *
import time

def evaluate():
    env = SnakeGame(render_mode=True)
    state_dim = 11
    action_dim = 4
    
    agent = DQNAgent(state_dim, action_dim)
    # Load model
    try:
        agent.policy_net.load_state_dict(torch.load("models/snake_dqn.pth", map_location=DEVICE))
        agent.epsilon = 0.0 # No exploration
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No model found. Running with random agent.")
        agent.epsilon = 1.0

    for episode in range(5):
        state = env.reset()
        done = False
        print(f"Episode {episode+1} starting...")
        
        while not done:
            # Handle pygame events (so window doesn't freeze)
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            # Slow down for visualization
            time.sleep(0.1)
            
        print(f"Episode {episode+1} finished. Score: {env.score}")

if __name__ == "__main__":
    evaluate()
