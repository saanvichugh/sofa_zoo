import numpy as np
from stable_baselines3 import SAC

from sofa_env.scenes.pick_and_place.pick_and_place_env import ActionType, RenderMode, ObservationType, PickAndPlaceEnv, Phase

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, SAC_KWARGS


if __name__ == "__main__":

    add_render_callback = False
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    reward_amount_dict = {
        Phase.ANY: {
            "lost_grasp": -10.0,
            "grasped_torus": 0.0,
            "gripper_jaw_peg_collisions": -0.0,
            "gripper_jaw_floor_collisions": -0.0,
            "unstable_deformation": -0.0,
            "torus_velocity": -0.0,
            "gripper_velocity": -0.0,
            "torus_dropped_off_board": -0.0,
            "action_violated_state_limits": -0.0,
            "action_violated_cartesian_workspace": -0.0,
            "successful_task": 50.0,
        },
        Phase.PICK: {
            "established_grasp": 10.0,
            "gripper_distance_to_torus_center": -0.0,
            "delta_gripper_distance_to_torus_center": -0.0,
            "gripper_distance_to_torus_tracking_points": -1.0,
            "delta_gripper_distance_to_torus_tracking_points": -1.0,
            "distance_to_minimum_pick_height": -1.0,
            "delta_distance_to_minimum_pick_height": -1.0,
        },
        Phase.PLACE: {
            "torus_distance_to_active_pegs": -0.0,
            "delta_torus_distance_to_active_pegs": -1.0,
        },
    }

    observation_type = ObservationType.STATE
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

    env_kwargs = {
        "image_shape": (64, 64),
        "render_mode": RenderMode.NONE,
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "time_step": 0.01,
        "frame_skip": 3,
        "settle_steps": 50,
        "reward_amount_dict": reward_amount_dict,
        "on_reset_callbacks": None,
        "create_scene_kwargs": None,
        "num_active_pegs": 1,
        "randomize_color": False,
        "num_torus_tracking_points": 5,
        "start_grasped": False,
        "randomize_torus_position": False,
        "only_learn_pick": False,
        "minimum_lift_height": 30.0,
        "block_done_when_torus_unstable": False,
    }

    config = {"max_episode_steps": 500, **CONFIG}
    config["frame_stack"] = 4

    if image_based:
        sac_kwargs = SAC_KWARGS["image_based"]
    else:
        sac_kwargs = SAC_KWARGS["state_based"]

    info_keywords = [
        "successful_task",
        "grasped_torus",
        "established_grasp",
        "lost_grasp",
        "torus_distance_to_active_pegs",
        "gripper_distance_to_torus_center",
        "gripper_jaw_peg_collisions",
        "gripper_jaw_floor_collisions",
        "unstable_deformation",
        "torus_dropped_off_board",
        "ret_suc_tas",
        "ret_los_gra",
        "ret_est_gra",
    ]

    config["sac_config"] = sac_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    random_seed = 1
    print(f"[PICK AND PLACE SAC] Using random_seed: {random_seed}")

    model, callback = configure_learning_pipeline(
        env_class=PickAndPlaceEnv,
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
        tb_log_name=f"SAC_{observation_type.name}_{continuous_actions=}_num_active_pegs={env_kwargs['num_active_pegs']}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
    model.get_env().save(log_path + "vecnormalize.pkl")

    with open(log_path + "seed.txt", "w") as f:
        f.write(str(random_seed))
