
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """
        作用：返回一个用于控制观测噪声的向量。
        核心逻辑：
            通过读取配置文件中的噪声参数(noise_scales 和 noise_level)以及观测的尺度因子(obs_scales)计算噪声强度。
            不同观测量（如角速度、重力、关节位置、速度）有不同的噪声强度。
            输出是一个形状与观测数据一致的张量，用于生成噪声。

        Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        """
        作用：初始化与足部相关的状态数据。
        核心逻辑：
            获取机器人刚体的状态数据，并提取足部位置和速度。
            足部信息通过 self.feet_indices 确定，用于定义哪些关节是足部。
        """
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        """
        作用：初始化一些必要的缓冲区。
        核心逻辑：
            调用父类的初始化方法。
            初始化与足部状态相关的缓冲区（_init_foot）。    
        """
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        """
        作用：更新足部状态（位置和速度）。
        核心逻辑：
            通过 Isaac Gym 的 API 刷新刚体状态。
            重新计算足部的位置和速度。
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        """
        作用：每步物理模拟后更新足部状态和步态相位。
        核心逻辑：
            更新步态周期（self.phase）和左右腿的相位偏移。
            调用父类的回调方法。
        """
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    # # 为跳跃重新设计
    # def _post_physics_step_callback(self):
    #     # 缩短周期并取消左右腿相位差，使所有腿同步发力
    #     period = 0.5 # 更短的周期
    #     self.phase = (self.episode_length_buf * self.dt) % period / period
    #     self.leg_phase = self.phase.unsqueeze(1).repeat(1, 4) # 假设4条腿同步
    #     return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        作用：计算机器人在当前环境下的观测数据。
        核心逻辑：
            观测数据包括角速度、重力、指令、关节位置、速度等，以及步态的正弦和余弦相位。
            提供特权观测（privileged_obs_buf），用于模拟学习时增强策略的训练。
            如果启用了噪声（add_noise），会对观测数据添加噪声。
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # 奖励足部在支撑相位内与地面接触，或在摆动相位内不接触地面。
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    # 奖励足部在摆动相位的期望高度，惩罚偏差。
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    # 奖励机器人保持存活
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    # 惩罚足部接触地面但没有速度的情况，鼓励动态接触。
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    # 惩罚髋关节位置偏离目标值的情况
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    
    '''设计跳跃专用奖励函数'''
    def _reward_jump_height(self):
        """奖励基座高度超过目标值"""
        target_height = 0.1 # 目标跳跃高度
        current_height = self.root_states[:, 2] # z轴位置
        return torch.clamp(current_height - target_height, min=0.0)
    
    def _reward_vertical_velocity(self):
        """奖励向上的垂直速度"""
        vertical_vel = self.root_states[:, 11] # z轴线速度
        return torch.clamp(vertical_vel, min=0.0)
    
    # 在_prepare_reward_function中激活新奖励


    
    
    