import pygame
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Configs
CELL_SIZE = 20
MAZE_WIDTH = 15
MAZE_HEIGHT = 15
WINDOW_WIDTH = MAZE_WIDTH * CELL_SIZE
WINDOW_HEIGHT = MAZE_HEIGHT * CELL_SIZE
BLACK, WHITE, RED, GREEN, BLUE = (0,0,0), (255,255,255), (255,0,0), (0,255,0), (0,0,255)

# Pygame init
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()

# Maze generator
def generate_maze(w, h):
    maze = np.zeros((h, w), dtype=int)
    def carve(x, y):
        dirs = [(2,0), (-2,0), (0,2), (0,-2)]
        maze[y][x] = 1
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0<nx<w-1 and 0<ny<h-1 and maze[ny][nx]==0:
                maze[y+dy//2][x+dx//2]=1
                carve(nx, ny)
    carve(1, 1)
    maze[h-2][w-2]=1
    
    return maze

maze = generate_maze(MAZE_WIDTH, MAZE_HEIGHT)

# Agent class
class Agent:
    def __init__(self):
        self.x, self.y, self.angle = 1, 1, 0
    def reset(self): self.x, self.y, self.angle = 1, 1, 0
    def move(self, dx, dy):
        nx, ny = self.x+dx, self.y+dy
        if 0<=nx<MAZE_WIDTH and 0<=ny<MAZE_HEIGHT and maze[ny][nx]==1:
            self.x, self.y = nx, ny
            return -1
        return -10
    def rotate(self, da): self.angle = (self.angle+da)%360
    def cast_ray(self, offset, max_dist=10):
        ang = math.radians(self.angle+offset)
        dx, dy = math.cos(ang), math.sin(ang)
        dist = 0.0
        while dist<max_dist:
            tx, ty = int(self.x+dx*dist), int(self.y+dy*dist)
            if tx<0 or ty<0 or tx>=MAZE_WIDTH or ty>=MAZE_HEIGHT or maze[ty][tx]==0:
                break
            dist += 0.1
        return round(dist, 1)
    def get_state(self):
        return torch.tensor([self.cast_ray(a) for a in [-90, -45, 0, 45, 90]], dtype=torch.float32).unsqueeze(0)

# Neural net Q-approximator
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x): return self.fc(x)

# Replay memory
memory = deque(maxlen=10000)
batch_size = 64

# Setup
agent = Agent()
qnet = QNet()
optimizer = optim.Adam(qnet.parameters(), lr=0.001)
criterion = nn.MSELoss()

epsilon, gamma = 1.0, 0.95

# Draw function
def draw():
    screen.fill(BLACK)
    for y in range(MAZE_HEIGHT):
        for x in range(MAZE_WIDTH):
            c = WHITE if maze[y][x]==1 else BLACK
            pygame.draw.rect(screen, c, (x*CELL_SIZE,y*CELL_SIZE,CELL_SIZE,CELL_SIZE))
    pygame.draw.rect(screen, BLUE, ((MAZE_WIDTH-2)*CELL_SIZE, (MAZE_HEIGHT-2)*CELL_SIZE, CELL_SIZE, CELL_SIZE))
    for a in [-90,-45,0,45,90]:
        d=agent.cast_ray(a)
        ex=agent.x+math.cos(math.radians(agent.angle+a))*d
        ey=agent.y+math.sin(math.radians(agent.angle+a))*d
        pygame.draw.line(screen, GREEN, (agent.x*CELL_SIZE+10,agent.y*CELL_SIZE+10), (ex*CELL_SIZE+10, ey*CELL_SIZE+10))
    pygame.draw.circle(screen, RED, (agent.x*CELL_SIZE+10, agent.y*CELL_SIZE+10), 6)
    pygame.display.flip()

# Main loop
episode, total_steps = 0, 0
while True:
    clock.tick(1000)
    for e in pygame.event.get():
        if e.type==pygame.QUIT: pygame.quit(); quit()

    state = agent.get_state()
    if random.random()<epsilon:
        action = random.randint(0,2)
    else:
        with torch.no_grad():
            qvals = qnet(state)
            action = torch.argmax(qvals).item()

    if action==0:
        dx, dy = round(math.cos(math.radians(agent.angle))), round(math.sin(math.radians(agent.angle)))
        reward = agent.move(dx,dy)
    elif action==1:
        agent.rotate(-90); reward=-1
    else:
        agent.rotate(90); reward=-1

    done = agent.x==MAZE_WIDTH-2 and agent.y==MAZE_HEIGHT-2
    if done: reward=+10

    next_state = agent.get_state()
    memory.append((state, action, reward, next_state, done))

    if done:
        agent.reset()
        episode+=1
        epsilon = max(0.05, epsilon*0.99)
        print(f"Episode {episode} | Epsilon {epsilon:.2f}")

    if len(memory)>=batch_size:
        batch = random.sample(memory, batch_size)
        states = torch.cat([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch])
        next_states = torch.cat([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch])

        qvals = qnet(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_qvals = qnet(next_states).max(1)[0]
            targets = rewards + gamma * next_qvals * (1-dones.float())

        loss = criterion(qvals, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    draw()
    total_steps +=1
