import numpy as np
import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


EXTRANGE_ROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/extrange_robot/source/extrange_robot/assets/usd/extrange-robot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.01),
        joint_pos={
            "base_to_wheel_joint": 0.0,
            "extension_rotation_joint": 1.57,
            "extension_tilt_joint": 0.0
        },
    ),
    actuators={
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["base_to_wheel_joint"],
            effort_limit=40.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=2.0,
        ),
        "yaw": ImplicitActuatorCfg(
            joint_names_expr=["extension_rotation_joint"],
            effort_limit=40.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=2.0,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["extension_tilt_joint"],
            effort_limit=40.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=2.0,
        ),
    }
)
