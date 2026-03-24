import pygame
import random
import numpy as np
from env.antigravity import GravityManager
from config import *

class SnakeGame:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.grid_size = GRID_SIZE
        self.block_size = BLOCK_SIZE
        self.w = self.grid_size * self.block_size
        self.h = self.grid_size * self.block_size
        
        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('AntiGravity Snake')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('arial', 20)
            
        self.gravity_mgr = GravityManager()
        self.reset()
        
    def reset(self):
        # Initialize snake: start in middle
        self.head = [self.grid_size // 2, self.grid_size // 2]
        self.snake = [self.head.copy(), 
                      [self.head[0]-1, self.head[1]], 
                      [self.head[0]-2, self.head[1]]]
        
        self.direction = (1, 0) # Moving right
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.gravity_mgr.shift()
        
        return self.get_state()

    def _place_food(self):
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()

    def get_state(self):
        # State: 11 values
        # [Danger Straight, Danger Right, Danger Left,
        # Dir Left, Dir Right, Dir Up, Dir Down,
        # Food Left, Food Right, Food Up, Food Down,
        # Grav X, Grav Y] -> Let's use the user's requested state space actually.
        
        # Phase 2 State Space Recommendation:
        # Snake head position (x, y)
        # Food relative position (dx, dy)
        # Current gravity vector (gx, gy)
        # Snake length
        # Danger in 4 directions
        
        head = self.snake[0]
        
        # Relative food position
        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]
        
        # Gravity
        gx, gy = self.gravity_mgr.current_gravity
        
        # Danger
        danger = [
            self.is_collision([head[0], head[1]-1]), # Up
            self.is_collision([head[0], head[1]+1]), # Down
            self.is_collision([head[0]-1, head[1]]), # Left
            self.is_collision([head[0]+1, head[1]])  # Right
        ]
        
        # Action-based danger (Straight, Right, Left) is common in Snake RL
        # But we'll follow the prompt + some robustness
        state = [
            head[0] / self.grid_size, head[1] / self.grid_size, # Normalized head
            food_dx / self.grid_size, food_dy / self.grid_size, # Normalized food relative
            gx, gy, # Gravity vector
            len(self.snake) / (self.grid_size * self.grid_size), # Normalized length
            *danger # Danger flags
        ]
        
        return np.array(state, dtype=np.float32)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt[0] >= self.grid_size or pt[0] < 0 or pt[1] >= self.grid_size or pt[1] < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def step(self, action):
        """
        Action: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        """
        self.frame_iteration += 1
        reward = 0
        done = False
        
        # 1. Action Movement
        move_vectors = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        move_dir = move_vectors[action]
        
        # 2. Gravity Drift
        # Gravity pulls snake 1 cell every tick if no action taken? 
        # The prompt says: "Snake "drifts" 1 cell per gravity tick if no action taken"
        # Since the AI *always* takes an action in DQN, I'll interpret this as 
        # Movement = Action + Gravity Drift
        gx, gy = self.gravity_mgr.current_gravity
        
        new_head = [self.head[0] + move_dir[0] + gx, 
                    self.head[1] + move_dir[1] + gy]
        
        # Update head
        self.head = new_head
        self.snake.insert(0, self.head)
        
        # 3. Handle Physics / Collision
        if self.is_collision():
            done = True
            reward = REWARD_DIE
            return self.get_state(), reward, done

        # 4. Check Food
        if self.head == self.food:
            self.score += 1
            reward = REWARD_EAT
            self._place_food()
        else:
            self.snake.pop()
            
            # Reward shaping
            # Closer to food?
            dist_curr = abs(self.food[0] - self.head[0]) + abs(self.food[1] - self.head[1])
            prev_head = self.snake[1]
            dist_prev = abs(self.food[0] - prev_head[0]) + abs(self.food[1] - prev_head[1])
            
            if dist_curr < dist_prev:
                reward += REWARD_CLOSER
            else:
                reward += REWARD_AWAY
                
            # Gravity rewards
            # Riding gravity: action matches gravity vector
            if move_dir == (gx, gy) and (gx != 0 or gy != 0):
                reward += REWARD_GRAVITY_RIDE
            # Fighting gravity: action opposite to gravity vector
            elif move_dir == (-gx, -gy) and (gx != 0 or gy != 0):
                reward += REWARD_GRAVITY_FIGHT

        # 5. Shift Gravity
        if self.frame_iteration % GRAVITY_SHIFT == 0:
            self.gravity_mgr.shift()

        # 6. Periodic timeout to prevent infinite loops
        if self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = REWARD_DIE

        if self.render_mode:
            self.render()
            
        return self.get_state(), reward, done

    def render(self):
        self.display.fill((0, 0, 0)) # Black
        
        # Draw Snake
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), # Green
                             pygame.Rect(pt[0]*BLOCK_SIZE, pt[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        # Draw Food
        pygame.draw.rect(self.display, (255, 0, 0), # Red
                         pygame.Rect(self.food[0]*BLOCK_SIZE, self.food[1]*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        # Gravity Info
        text = self.font.render(f"Gravity: {self.gravity_mgr.get_gravity_name()}", True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        
        pygame.display.flip()
        self.clock.tick(FPS)
