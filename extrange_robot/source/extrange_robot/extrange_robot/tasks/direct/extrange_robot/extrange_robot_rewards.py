import torch
from abc import ABC, abstractmethod
from isaaclab.utils.math import quat_apply


class RewardStrategy(ABC):
    def __init__(self, env):
        self.env = env
        self.cfg = env.cfg

    @abstractmethod
    def compute(self) -> torch.Tensor:
        pass


class CoMBalanceReward(RewardStrategy):

    def compute(self) -> torch.Tensor:
        robot = self.env.robot
        cfg = self.cfg
        dt = cfg.sim.dt

        # === ESTADO ===
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        torque = robot.data.applied_torque
        quat = robot.data.root_quat_w
        pos = robot.data.root_pos_w
        com = robot.data.root_com_pos_w
        ang_vel = robot.data.root_ang_vel_w

        euler = self.env.quaternion_to_euler_xyz(quat)
        roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

        joint_ids = self.env._joint_indices
        tilt_vel = joint_vel[:, joint_ids[2]]
        yaw_vel = joint_vel[:, joint_ids[1]]
        torque_tilt = torque[:, joint_ids[2]]
        torque_yaw = torque[:, joint_ids[1]]
        torque_wheel = torque[:, joint_ids[0]]

        # === CENTRO DE MASA ===
        com_proj = com[:, :2]
        base_xy = pos[:, :2]
        com_offset = com_proj - base_xy            # [Δx, Δy]
        com_velocity = (com - self.env._prev_com_pos) / dt
        com_y_acc = (com_velocity[:, 1] - self.env._prev_com_vel[:, 1]) / dt

        # === COMPONENTES DE REWARD ===

        # 1. recompensa por estar vivo
        alive_reward = cfg.rew_alive

        # 2. mantener verticalidad general (roll + pitch)
        base_tilt = roll**2 + pitch**2
        upright_reward = cfg.rew_upright_bonus * torch.exp(-5.0 * base_tilt)
        base_tilt_penalty = cfg.rew_base_tilt_penalty * base_tilt ** 2

        # 3. corrección con rueda usando error de CoM en X
        com_x_error = com_offset[:, 0]
        desired_torque = -cfg.rew_com_x_balance_gain * com_x_error
        wheel_reward = cfg.rew_wheel_balance_correction * desired_torque * torque_wheel

        # 4. corrección con tilt para compensar movimiento del CoM en Y
        com_y_vel = com_velocity[:, 1]
        tilt_effect = tilt_vel * torch.sign(com_y_vel)
        tilt_reward = -cfg.rew_tilt_balance_correction * tilt_effect

        # 5. penalización por aceleración del CoM lateral (efecto de inercia)
        com_y_acc_penalty = cfg.rew_com_y_acc_penalty * com_y_acc ** 2

        # 6. penalización por velocidad de articulaciones
        velocity_penalty = cfg.rew_vel_penalty * torch.sum(joint_vel[:, 1:] ** 2, dim=-1)

        # 7. penalizaciones de torque y aceleración angular
        tilt_acc = (tilt_vel - self.env._prev_tilt_vel) / dt
        yaw_acc = (yaw_vel - self.env._prev_yaw_vel) / dt

        tilt_acc_penalty = cfg.rew_tilt_acc_penalty * tilt_acc**2
        yaw_acc_penalty = cfg.rew_yaw_acc_penalty * yaw_acc**2
        tilt_torque_penalty = cfg.rew_tilt_torque_penalty * torque_tilt**2
        yaw_torque_penalty = cfg.rew_yaw_torque_penalty * torque_yaw**2

        # 8. mantener orientación target en yaw y tilt (suave)
        yaw_error = yaw - cfg.target_rotation_angle
        yaw_target_penalty = cfg.rew_rotation_target_penalty * yaw_error**2
        yaw_velocity_penalty = cfg.rew_yaw_velocity_penalty * yaw_vel**2
        tilt_target_penalty = cfg.rew_tilt_target_penalty * (joint_pos[:, joint_ids[2]] - cfg.target_tilt_angle)**2

        # === SUMA TOTAL ===
        total = (
            alive_reward
            + upright_reward
            + wheel_reward
            + tilt_reward
            - base_tilt_penalty
            - com_y_acc_penalty
            - velocity_penalty
            - tilt_acc_penalty
            - yaw_acc_penalty
            - tilt_torque_penalty
            - yaw_torque_penalty
            - yaw_target_penalty
            - yaw_velocity_penalty
            - tilt_target_penalty
        )

        # Normalización batch
        mean = total.mean()
        std = total.std(unbiased=False) + 1e-4
        total_normalized = (total - mean) / std
        reward_total = total_normalized * 0.2

        # Actualizar variables internas
        self.env._prev_com_vel = com_velocity.clone()
        return reward_total


######################################################################
#                   basic controller
######################################################################
#
#        quat = self.robot.data.root_quat_w
#        pos = self.robot.data.root_pos_w
#        ang_vel = self.robot.data.root_ang_vel_w
#        euler = quaternion_to_euler_xyz(quat)
#
#        roll = euler[:, 0]
#        pitch = euler[:, 1]
#        yaw = euler[:, 2]
#
#        # magnitud de la inclinación
#        base_tilt = roll ** 2 + pitch ** 2
#        joint_1_pos = self.joint_pos[:, self._joint_indices[1]]
#        joint_2_pos = self.joint_pos[:, self._joint_indices[2]]
#        joint_vel = torch.stack([self.joint_vel[:, i] for i in self._joint_indices], dim=-1)
#        torque = torch.stack([self.robot.data.applied_torque[:, i] for i in self._joint_indices], dim=-1)
#        
#
#        # PRIORIDAD ALTA: verticalidad
#        upright_reward = self.cfg.rew_upright_bonus * torch.exp(-5.0 * base_tilt)
#        base_tilt_penalty = self.cfg.rew_base_tilt_penalty * base_tilt ** 2
#
#        # PRIORIDAD MEDIA: corrección activa con torque
#        torque_wheel = torque[:, 0]
#        desired_torque = - 2 * torch.sin(pitch)
#        torque_effect = desired_torque * torque_wheel
#        wheel_correction_reward = self.cfg.rew_wheel_balance_correction * torque_effect
#
#        torque_tilt = torque[:, 2]
#        tilt_vel = self.joint_vel[:, self._joint_indices[2]]
#        torque_effect = torch.sign(pitch) * torque_tilt * tilt_vel
#        tilt_correction_reward = self.cfg.rew_tilt_balance_correction * torch.tanh(roll) * torque_effect
#
#
#        # PRIORIDAD BAJA: mantenerse cerca de Y = 0
#        target_y = self.cfg.target_y_position
#        position_penalty = self.cfg.rew_position_penalty * ((pos[:, 1] - target_y) ** 2)
#
#        # penalización menor por velocidad en joints 1 y 2
#        velocity_penalty = self.cfg.rew_vel_penalty * torch.sum(joint_vel[:, 1:] ** 2, dim=-1)
#
#        # castigo si el joint 1 y 2 se mueve del drift
#        target_tilt = self.cfg.target_tilt_angle
#        target_rotation = self.cfg.target_rotation_angle
#        tilt_target_penalty = self.cfg.rew_tilt_target_penalty * (joint_2_pos - target_tilt) ** 2
#        rotation_target_penalty = self.cfg.rew_rotation_target_penalty * (joint_1_pos - target_rotation) ** 2
#        
#        alive_reward = self.cfg.rew_alive
#
#        total_reward = (
#            alive_reward
#            + upright_reward
#            + wheel_correction_reward
#            + tilt_correction_reward
#            - rotation_target_penalty
#            - tilt_target_penalty
#            - base_tilt_penalty
#            - position_penalty
#            - velocity_penalty
#        )