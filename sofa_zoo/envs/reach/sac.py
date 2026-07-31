import numpy as np
from stable_baselines3 import SAC

from sofa_env.scenes.reach.reach_env import ActionType, RenderMode, ObservationType, ReachEnv

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, SAC_KWARGS


if __name__ == "__main__":

    add_render_callback = False
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    # observation_type, sphere_radius
    parameters = ["STATE", "normal"]

    sphere_radii = {
        "normal": 0.008,
        "big": 0.02,
        "small": 0.003,
    }

    sphere_radius = sphere_radii[parameters[1]]

    observation_type = ObservationType[parameters[0]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
      #  "render_mode": RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE,
        "render_mode": RenderMode.NONE,
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "distance_to_target_threshold": 0.003,  # m
        "time_step": 0.1,
        "frame_skip": 1,
        "observe_target_position": False if image_based else True,
        "reward_amount_dict": {
            "distance_to_target": -1.0,
            "delta_distance_to_target": -1.0,
            "successful_task": 10.0,
            "time_step_cost": 0.0,
            "workspace_violation": 0.0,
        },
        "on_reset_callbacks": None,
        "create_scene_kwargs": {
            "show_bounding_boxes": True,
        },
        "sphere_radius": sphere_radius,
    }

    config = {"max_episode_steps": 500, **CONFIG}
    config["frame_stack"] = 4

    if image_based:
        sac_kwargs = SAC_KWARGS["image_based"]
    else:
        sac_kwargs = SAC_KWARGS["state_based"]

    info_keywords = [
        "distance_to_target",
        "ret_del_dis_to_tar",
        "ret_dis_to_tar",
        "ret_suc_tas",
        "ret_tim_ste_cos",
        "ret_wor_vio",
        "successful_task",
    ]

    config["sac_config"] = sac_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    random_seed = 1
    print(f"[REACH SAC] Using random_seed: {random_seed}")

    model, callback = configure_learning_pipeline(
        env_class=ReachEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        normalize_observations=False,
        algo_class=SAC,
        algo_kwargs=sac_kwargs,
        render=add_render_callback,
        normalize_reward=normalize_reward,
        model_checkpoint_distance=config["checkpoint_distance"],
        random_seed=random_seed,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        tb_log_name= f"SAC_{observation_type.name}_{continuous_actions=}_{sphere_radius=}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
    model.get_env().save(log_path + "vecnormalize.pkl")

    with open(log_path + "seed.txt", "w") as f:
        f.write(str(random_seed))
