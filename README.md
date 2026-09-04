# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium
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

---


