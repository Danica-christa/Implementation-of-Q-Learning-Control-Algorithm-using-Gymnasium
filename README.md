# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
The objective is to train an autonomous agent to navigate a slippery 4x4 grid world (`FrozenLake-v1`) from the starting position (S) to the goal state (G) while avoiding hazardous holes (H). Because the surface is slippery, the agent's movement direction is uncertain and only partially dependent on the chosen action. Using the model-free **Q-Learning** algorithm, the agent must iteratively discover an optimal policy that maximizes cumulative discounted rewards despite the stochastic environment dynamics.

## Software Requirements
1. Python: 3.8+
2. Gymnasium: gymnasium>=0.29.0
3. NumPy: numpy>=1.22.0
4. Matplotlib: matplotlib>=3.5.0
5. Jupyter Notebook / Google Colab
---

## Environment Description
The `FrozenLake-v1` environment represents a 4x4 grid of frozen and broken ice tiles:

* **Grid Layout**:
```text
S F F F
F H F H
F F F H
H F F G
```

* **Legend**:
  * `S`: Starting point (safe)
  * `F`: Frozen surface (safe)
  * `H`: Hole (fall into hole -> episode terminates with 0 reward)
  * `G`: Goal (reach goal -> episode terminates with +1 reward)

* **State Space**: Discrete space of 16 states (0 to 15).

* **Action Space**: Discrete space with 4 possible actions:
  * `0`: Left
  * `1`: Down
  * `2`: Right
  * `3`: Up

* **Transition Dynamics**: With `is_slippery=True`, taking an action moves the agent in the intended direction with probability 1/3, and in either of the two perpendicular directions with probability 1/3 each.

* **Reward Mechanism**:
  * Reaching Goal (`G`): +1
  * Falling into Hole (`H`): 0
  * Moving to Frozen tile (`F`): 0


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |


## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
### Q-Learning Algorithm

1. Create the `FrozenLake-v1` environment.
2. Determine the number of states and actions.
3. Initialize the Q-table with zeros.
4. Set the learning rate `α`.
5. Set the discount factor `γ`.
6. Initialize the exploration rate `ε`.
7. Repeat for the specified number of episodes:

   * Reset the environment.
   * Obtain the initial state.
   * Select an action using the epsilon-greedy strategy.
   * Execute the action.
   * Observe the reward and next state.
   * Calculate the Q-Learning target.
   * Update the Q-value.
   * Move to the next state.
   * Continue until the episode terminates.
   * Store the total reward.
   * Reduce epsilon.
8. Calculate the state-value function from the Q-table.
9. Extract the greedy policy from the Q-table.
10. Display the final Q-table.
11. Display the estimated state-value function.
12. Display the learned policy.
13. Calculate the average reward over the last 1000 episodes.
14. Plot the learning curve.
15. Close the environment.
```

## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

num_states = env.observation_space.n
num_actions = env.action_space.n

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 20000
max_steps = 100

alpha = 0.1          # learning rate
gamma = 0.99         # discount factor

epsilon = 1.0        # starting exploration rate
epsilon_min = 0.01
epsilon_decay = 0.0005   # exponential decay rate

# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((num_states, num_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

rng = np.random.default_rng(42)

def choose_action(state, epsilon):
    if rng.random() < epsilon:
        return env.action_space.sample()
    # random tie-breaking among the maximal action values
    best = np.flatnonzero(Q[state] == Q[state].max())
    return int(rng.choice(best))

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-learning update
        if terminated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] += alpha * (target - Q[state, action])

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

# decay exploration
    epsilon = epsilon_min + (1.0 - epsilon_min) * np.exp(-epsilon_decay * episode)

    episode_rewards.append(total_reward)

episode_rewards = np.array(episode_rewards)

# derive the value function and greedy policy from Q
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)
```
## Output

### Final Q-table:
<img width="350" height="350" alt="image" src="https://github.com/user-attachments/assets/3f4f3728-434d-44af-bc4e-9aeb09dc1f53" />


### Estimated State-Value Function:
<img width="422" height="117" alt="image" src="https://github.com/user-attachments/assets/7a0f4e6e-ab0f-40f8-a599-f60cb6256355" />


### Learned Policy:
<img width="301" height="117" alt="image" src="https://github.com/user-attachments/assets/4578b15b-70aa-4e7b-901e-6058ac176d2c" />


### Average reward over last 1000 episodes:
<img width="451" height="35" alt="image" src="https://github.com/user-attachments/assets/306febb7-c915-4d0d-8a28-71ef66e5a208" />

### Plot Learning Curve:
<img width="957" height="521" alt="image" src="https://github.com/user-attachments/assets/d43a101a-ed8d-44e3-bad6-38c0042fb682" />


---

## Result

The Q-Learning algorithm was successfully implemented on the Gymnasium `FrozenLake-v1` environment. The agent learned the optimal action-value function ($Q$) and derived a policy that successfully navigates the slippery grid world from start to goal while avoiding holes, achieving an average reward of ~0.43 over the last 1000 training episodes.

---

## Inference

1. **Convergence on Stochastic Dynamics**: Despite the slippery environment making transitions non-deterministic (with only a 33% chance of moving in the intended direction), the Q-learning agent successfully converged toward a high-reward policy.
2. **Exploration vs. Exploitation Balance**: The exponential decay of epsilon ($\epsilon$) ensured sufficient state-space exploration early on and shifted the agent toward stable exploitation in later episodes.
3. **Safe Policy Formulation**: The learned policy directs the agent into walls and safe boundaries near holes rather than directly toward the goal, deliberately minimizing the probability of accidentally slipping into holes.











