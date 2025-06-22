from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .extrange_robot_env_cfg import ExtrangeRobotEnvCfg
from .extrange_robot_rewards import CoMBalanceReward 




class ExtrangeRobotEnv(DirectRLEnv):
    cfg: ExtrangeRobotEnvCfg

    def __init__(self, cfg: ExtrangeRobotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._frame_count = 0  # contador de frames para debug
        self._joint_indices = []
        for name in self.robot.cfg.actuators:
            dof_ids, _ = self.robot.find_joints(self.robot.cfg.actuators[name].joint_names_expr[0])
            self._joint_indices.append(dof_ids[0])

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self._prev_tilt_vel = torch.zeros((self.num_envs,), device=self.device)
        self._prev_yaw_vel = torch.zeros((self.num_envs,), device=self.device)
        self._prev_com_pos = torch.zeros((self.num_envs,3), device=self.device)
        self._prev_com_vel = torch.zeros((self.num_envs, 3), device=self.device)
        
        self.root_ang_vel = self.robot.data.root_ang_vel_w
        self.root_orn = self.robot.data.root_quat_w
        self._prev_root_vel = self.robot.data.root_lin_vel_w.clone()
        #self.joint_torque = self.robot.data.joint_torque

        self._goal_pos = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.reward_strategy = CoMBalanceReward(self) 

        body_id = self.robot.body_names.index("upper_joint")
        self._top_link_id = torch.full((self.num_envs,), body_id, device=self.device, dtype=torch.long)
        self._top_link_prim_paths = [f"/World/envs/env_{i}/Robot/upper_joint" for i in range(self.num_envs)]
        self._perturb_forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
        self._perturb_torques = torch.zeros_like(self._perturb_forces)
        
        
    def quaternion_to_euler_xyz(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack([roll, pitch, yaw], dim=-1)
    
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=3500.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)
        #point_light_cfg = sim_utils.SphereLightCfg(radius=1.5, intensity=10000.0, color=(1.0, 1.0, 1.0))
        #point_light_cfg.func("/World/PointLight", point_light_cfg)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._frame_count += 1

    def _apply_action(self) -> None:
        # Aplicar control del agente
        self.robot.set_joint_effort_target(
            self.actions * self.cfg.action_scale,
            joint_ids=self._joint_indices
        )

        self.apply_random_impulses()





    def _get_observations(self) -> dict:
        pos = torch.stack([self.joint_pos[:, i] for i in self._joint_indices], dim=-1)
        vel = torch.stack([self.joint_vel[:, i] for i in self._joint_indices], dim=-1)
        torq = torch.stack([self.robot.data.applied_torque[:, i] for i in self._joint_indices], dim=-1)

#        if self._frame_count % 60 == 0:
#            print(f"[DEBUG] Torques aplicados (Nm): {torq[0].cpu().numpy()}")
#            print(dir(self.robot.data))
            
        quat = self.robot.data.root_quat_w
        euler = self.quaternion_to_euler_xyz(quat)
        ang_vel = self.robot.data.root_ang_vel_w
        lin_vel = self.robot.data.root_lin_vel_w
        dt = self.cfg.sim.dt
        lin_acc = (lin_vel - self._prev_root_vel) / dt
        self._prev_root_vel = lin_vel.clone()
        imu = torch.cat([euler, ang_vel, lin_acc], dim=-1)

#        if self._frame_count % 10 == 0:
#            print(f"[DEBUG] Euler: {euler[0].cpu().numpy()}")
#            print(f"[DEBUG] rotation actuator: {self.joint_pos[0, self._joint_indices[1]]}")
#            print(f"[DEBUG] tilt actuator: {self.joint_pos[0, self._joint_indices[2]]}")

        obs = torch.cat((pos, vel, torq, imu), dim=-1)
        if self.cfg.obs_noise_std > 0.0:
            noise = torch.randn_like(obs) * self.cfg.obs_noise_std
            obs = obs + noise
    
        return {"policy": obs}
    

    def _get_rewards(self) -> torch.Tensor:
        tilt_vel = self.robot.data.joint_vel[:, self._joint_indices[2]]  # actuador 2: tilt
        yaw_vel = self.robot.data.joint_vel[:, self._joint_indices[1]]   # actuador 1: yaw
        self._prev_tilt_vel = tilt_vel.clone()
        self._prev_yaw_vel = yaw_vel.clone()

        return self.reward_strategy.compute()

    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        quat = self.robot.data.root_quat_w
        euler = self.quaternion_to_euler_xyz(quat)
        roll = euler[:, 0]
        pitch = euler[:, 1]
        tilt = torch.maximum(torch.abs(roll), torch.abs(pitch))
        out_of_bounds = tilt > self.cfg.max_base_orientation_angle

        # detectar colisión física
        upper_joint_z = self.robot.data.body_pos_w[:, 3, 2]
        contact = upper_joint_z < self.cfg.min_upper_joint_height  

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        done = out_of_bounds | contact 

        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        for idx in self._joint_indices:
            joint_pos[:, idx] += sample_uniform(
                self.cfg.initial_joint_pos_range[0],
                self.cfg.initial_joint_pos_range[1],
                (len(env_ids),),
                joint_pos.device
            )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)



    def apply_random_impulses(self):
        """Aplica impulsos aleatorios a los upper_joint como fuerzas en el timestep."""
        prob = 1.0 / self.cfg.perturbation_interval
        mask = torch.rand(self.num_envs, device=self.device) < prob
        if not mask.any():
            return

        dt = self.cfg.sim.dt
        batch_size = self.num_envs

        angles = torch.rand(batch_size, device=self.device) * 2 * torch.pi
        inclinations = (torch.rand(batch_size, device=self.device) - 0.5) * torch.deg2rad(torch.tensor(40.0))
        directions = torch.stack([
            torch.cos(inclinations) * torch.cos(angles),
            torch.cos(inclinations) * torch.sin(angles),
            torch.sin(inclinations)
        ], dim=1)

        impulse_mag = self.cfg.perturbation_impulse_mag
        impulses = impulse_mag * directions / dt

        # Fuerzas externas: shape (num_envs, num_bodies, 3)
        self._perturb_forces.zero_()
        self._perturb_torques.zero_()
        forces = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
        torques = torch.zeros_like(forces)

        # Aplicar solo a los entornos activados
        for i in range(self.num_envs):
            if mask[i]:
                body_id = self._top_link_id[i]
                forces[i, body_id] = impulses[i]


        self.robot.set_external_force_and_torque(forces, torques)
        self.robot.write_data_to_sim()