import json

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit

from sofa_env.scenes.reach.reach_env import ActionType, RenderMode, ObservationType, ReachEnv

# Load the trained policy
model = SAC.load("../reach/runs/SAC_STATE_continuous_actions=True_sphere_radius=0.008_10saved_model.pth")

observation_type = ObservationType.STATE

# Recreate the SAME environment setup used in training

reward_amount_dict = {
            "distance_to_target": -1.0,
            "delta_distance_to_target": -1.0,
            "successful_task": 10.0,
            "time_step_cost": 0.0,
            "workspace_violation": 0.0,
}

env_kwargs = {
        "image_shape": (64, 64),
        "render_mode": RenderMode.NONE,
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS, 
        "distance_to_target_threshold": 0.003, 
        "time_step": 0.1,
        "frame_skip": 1,
        "observe_target_position": True,
        "reward_amount_dict": reward_amount_dict,
        "on_reset_callbacks": None,
        "create_scene_kwargs": {
            "show_bounding_boxes": True,
        },
        "sphere_radius": 0.008,
    }

env = ReachEnv(**env_kwargs)
env = TimeLimit(env, max_episode_steps=500)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)
env = VecNormalize.load("../reach/runs/SAC_STATE_continuous_actions=True_sphere_radius=0.008_10vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

n_episodes = 100
results = []

for ep in range(n_episodes):
    obs = env.reset()
    done = False
    
    ep_workspace_violations = 0
    ep_obstacle_collisions = 0
    ep_reward = 0.0
    ep_success = 0
    ep_steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        info = infos[0]
        
        if info.get("workspace_violation", 0.0) > 0: 
            ep_workspace_violations += 1

        if info.get("obstacle_collision", 0.0) > 0: 
            ep_obstacle_collisions += 1

        if info.get("successful_task", 0.0) > 0:
            ep_success = 1
            
        ep_reward += float(reward[0])
        ep_steps += 1
        
    results.append(
        {
            "episode": ep,
            "workspace violations": ep_workspace_violations,
            "obstacle collisions": ep_obstacle_collisions,
            "reward": ep_reward,
            "success": ep_success,
            "steps": ep_steps
        }
    )

    print(
        f"Episode {ep}: "
        f"workspace violations={ep_workspace_violations}, obstacle collisions={ep_obstacle_collisions}, success={ep_success}, "
        f"reward={ep_reward:.2f}, steps={ep_steps}"
    )

avg_workspace_violations = sum(r["workspace violations"] for r in results) / n_episodes
avg_obstacle_collisions = sum(r["obstacle collisions"] for r in results) / n_episodes
success_rate = sum(r["success"] for r in results) / n_episodes

summary = {
    "avg_workspace_violations": avg_workspace_violations,
    "avg_obstacle_collisions": avg_obstacle_collisions,
    "success_rate": success_rate,
    "n_episodes": n_episodes
}

env.close()

with open("violation_log10.json", "w") as f:
    json.dump({"episodes": results, "summary": summary}, f, indent=2)


print("\nSummary:")
print(f"Average workspace violations per episode: {avg_workspace_violations:.2f}")
print(f"Average obstacle collisions per episode: {avg_obstacle_collisions:.2f}")
print(f"Success rate: {success_rate:.2f}")
print("Saved to violation_log10.json")
