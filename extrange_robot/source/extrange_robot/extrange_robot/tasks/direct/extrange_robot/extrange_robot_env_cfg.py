import numpy as np
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg

from .extrange_robot_cfg import EXTRANGE_ROBOT_CFG

@configclass
class ExtrangeRobotEnvCfg(DirectRLEnvCfg):
    viewer = ViewerCfg(
        eye=(2.0, 2.0, 1.5),
        lookat=(0.0, 0.0, 0.3),
        cam_prim_path="/OmniverseKit_Persp",
        resolution=(1280, 720),
        origin_type="world",
    )
    
    # env
    decimation = 2
    episode_length_s = 5.0

    # - spaces definition
    action_space = 3  # 3 DoF control
    observation_space = 18  # pos(3) + vel(3) + torq(3) + imu(9) of each DOF
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot configuration
    robot_cfg = EXTRANGE_ROBOT_CFG

    # scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0, 
        replicate_physics=True
    )

    # joint names from the USD robot
    joint_names = ["base_to_wheel_joint", "extension_rotation_joint", "extension_tilt_joint"]

    # control config
    action_scale = 1.0

    # reward parameters
    # --- Recompensas ---
    rew_alive = 2.0                         # incentivo base por seguir "vivo"
    rew_upright_bonus = 1.5                 # incentivo suave por mantener verticalidad
    rew_wheel_balance_correction = 0.5      # incentivo por ayudar a balancear con la rueda
    rew_tilt_balance_correction = 1.0       # incentivo por ayudar a balancear con tilt

    # --- Penalizaciones (más suaves que antes) ---
    rew_base_tilt_penalty = 0.1             # castigo leve por estar inclinado
    rew_com_projection_penalty = 0.3        # castigo por COM fuera de base
    rew_com_velocity_penalty = 0.05         # penalización por velocidad del COM
    rew_tilt_torque_penalty = 0.005          # castigo por esfuerzo excesivo (evita saturación)
    
    rew_yaw_torque_penalty = 0.01
    rew_tilt_acc_penalty = 0.05             # penalización por aceleraciones angulares bruscas
    rew_yaw_acc_penalty = 0.01

    rew_rotation_target_penalty = 0.02      # castigo suave por desviarse del target de yaw
    rew_yaw_velocity_penalty = 0.05         # castigo por moverse en yaw sin control
    rew_tilt_target_penalty = 0.01          # castigo por desviarse del target tilt
    rew_vel_penalty = 0.01                  # castigo leve por mover articulaciones

    rew_com_x_balance_gain = 20.0
    rew_com_y_acc_penalty = 1.0


    target_rotation_angle = np.pi/2.0
    target_tilt_angle = 0
    
    # noise scale
    obs_noise_std = 0.01
    perturbation_interval = 250               # cada 1 segundo (si 120 Hz)
    perturbation_impulse_mag = 0.01           # magnitud en N·s
#    # --- rewards ---
#    rew_upright_bonus = 1.0                # ↑ incentivo fuerte por mantener verticalidad
#    rew_wheel_balance_correction = 8.0          # recompensa para mantener los actuadores en la posición target
#    rew_tilt_balance_correction = 8.0
#        
#    rew_alive = 0.050                         # ← recompensa base por estar en pie
#    
#    # --- Penalizaciones  ---
#    rew_tilt_target_penalty = 1.0
#    rew_rotation_target_penalty = 1
#    
#    rew_base_tilt_penalty = 15.0                 # castigo por inclinación (ya lo premias con upright)
#    rew_vel_penalty = 0.1                # valor bajo permite moverse sin miedo
#    rew_position_penalty = 0.0             # ← no bloquees la exploración espacial
#    
#    # target positions
#    target_y_position = 0.0                
#    target_tilt_angle = 0.0
#    target_rotation_angle = 1.57


    # reset conditions
    initial_joint_pos_range = [-0.1, 0.1]
    max_base_orientation_angle = 1.20
    min_upper_joint_height = 0.18