from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    """ 机器人实时控制核心类，负责：
        1、与实体机器人建立通信（DDS协议）
        2、加载训练好的策略网络（TorchScript模型）
        3、状态观测构建与动作生成
        4、安全状态切换（零力矩/默认位姿）
    """
    def __init__(self, config: Config) -> None:
        # 初始化通信接口（DDS），根据机器人类型（HG/GO）选择消息协议
        # 加载训练好的策略模型（policy.jit）
        # 初始化观测/动作缓冲区（qj, dqj, obs, action）
        # 绑定IMU、关节状态等传感器数据回调

        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0

        self.auto_walk = False  # 是否进入自动行走模式
        self.step_in_place = False  # 是否处于原地踏步模式
        self.step_start_time = 0  # 踏步开始时间

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self, msg: LowStateHG):
        """ HG型号机器人状态回调，解析：
            -电机状态（位置q、速度dq）
            -IMU数据（四元数、角速度）
            -遥控器信号（remote_controller）
        """
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        """ 发送控制命令到机器人，包含：
            -目标位置（q）
            -前馈力矩（tau）
            -PD参数（kp, kd）
            -CRC校验确保数据完整性
        """
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        """ 零力矩模式（安全启动）
            - 所有关节力矩设为0
            - 等待遥控器start按键触发后续动作
            防止上电时电机意外发力
        """
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        """ 安全移动到默认位姿（2秒缓动）：
            1、记录当前关节位置
            2、在num_steps步内插值到目标位置
            3、设置关节刚度(kp)/阻尼(kd)防止突变
        """
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        """ 
        默认位姿保持状态，核心功能：
        1. 通过高刚度PD控制维持预设关节角度
        2. 等待操作员按下遥控器A键确认启动策略
        3. 确保机器人进入策略控制前处于稳定状态
        """
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")

        # 条件判断：持续检测遥控器的A键是否按下（button[KeyMap.A] == 1）
        # 阻塞特性：未按下时循环将持续执行内部控制逻辑
        while self.remote_controller.button[KeyMap.A] != 1:
            # ----------腿部关节控制------------
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]                      # 获取实际电机ID
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i] # 目标位置
                self.low_cmd.motor_cmd[motor_idx].qd = 0                            # 目标速度（设为0）
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]           # 刚度系数（比例增益）
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]           # 阻尼系数（微分增益）
                self.low_cmd.motor_cmd[motor_idx].tau = 0                           # 前馈力矩（设为0）

            # 手臂/腰部关节控制（手臂/腰部关节可能具有不同的安全要求或运动范围，需独立配置参数）
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i] # 预设位置
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i] # 独立刚度
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i] # 独立阻尼
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            # 发送控制命令
            self.send_cmd(self.low_cmd)
            # 严格按照控制周期休眠
            time.sleep(self.config.control_dt) # 通常为0.002s（500Hz）

    def run(self):
        """ 主控制循环（每control_dt执行一次）：
            1. 获取实时状态：关节q/dq、IMU四元数/角速度
            2. 坐标变换：将IMU数据转换到骨盆坐标系（针对人形机器人）
            3. 构建观测向量：
               - 角速度（3维）
               - 重力方向投影（3维）
               - 遥控器指令（线速度x/y，角速度yaw）
               - 关节位置/速度 (标准化)
               - 历史动作 (动作延迟补偿)
               - 步态相位信号 (sin/cos)
            4. 策略网络推理生成动作
            5. 动作后处理：转换为目标关节角度
            6. 发送控制命令到机器人
        """

        # # 下面是自己改的代码，想要实现机器人原地踏步以后，向前走
        # self.counter += 1
        # # 检测遥控器启动信号
        # if self.remote_controller.button[KeyMap.start] == 1:
        #     # self.auto_walk = True  # 进入自动行走模式
        #     self.step_in_place = True  # 进入原地踏步模式
        #     self.step_start_time = time.time()  # 记录踏步开始时间
        
        # # 原地踏步阶段
        # if self.step_in_place:
        #     # 踏步持续时间检测（5秒）
        #     if time.time() - self.step_start_time < 5.0:
        #         self.cmd[0] = 0.0  # 前进速度为0（原地踏步）
        #         self.cmd[1] = 0.0  # 横向速度为0
        #         self.cmd[2] = 0.0  # 转向速度为0
        #     else:
        #         self.step_in_place = False  # 结束踏步
        #         self.auto_walk = True  # 进入自动行走模式

        # # 自动生成目标速度
        # if self.auto_walk:
        #     self.cmd[0] = 0.5  # 前进速度 0.5 m/s
        #     self.cmd[1] = 0.0  # 横向速度 0
        #     self.cmd[2] = 0.0  # 转向速度 0
        # else:
        #     self.cmd[0] = self.remote_controller.ly
        #     self.cmd[1] = self.remote_controller.lx * -1
        #     self.cmd[2] = self.remote_controller.rx * -1

        # # 检测停止信号
        # if self.auto_walk and self.remote_controller.button[KeyMap.select] == 1:
        #     self.auto_walk = False
        #     self.cmd = np.array([0.0, 0.0, 0.0])  # 停止运动

        # # 调整步态周期
        # if self.auto_walk:
        #     period = 0.6  # 较快速度对应较短周期
        # else:
        #     period = 0.8  # 默认周期 

        # # 更新相位信号
        # count = self.counter * self.config.control_dt
        # phase = count % period / period
        # sin_phase = np.sin(2 * np.pi * phase)
        # cos_phase = np.cos(2 * np.pi * phase)

        # # 构建观测向量
        # self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        # self.obs[9 + self.config.num_actions * 3] = sin_phase
        # self.obs[9 + self.config.num_actions * 3 + 1] = cos_phase

        # # 策略网络推理与动作执行
        # obs_tensor  = torch.from_numpy(self.obs).unsqueeze(0)
        # self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        # target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # # 发送控制命令
        # for i in range(len(self.config.leg_joint2motor_idx)):
        #     motor_idx = self.config.leg_joint2motor_idx[i]
        #     self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
        #     self.low_cmd.motor_cmd[motor_idx].qd = 0
        #     self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
        #     self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
        #     self.low_cmd.motor_cmd[motor_idx].tau = 0
        # self.send_cmd(self.low_cmd)

        # time.sleep(self.config.control_dt)

        # 计数器，记录当前控制循环的次数
        self.counter += 1
        # ==== 新增状态管理变量 ==== 
        if not hasattr(self, "step_in_place"):  # 初始化标志位
            self.step_in_place = False          # 原地踏步状态
            self.auto_walk = False              # 自动行走状态
            self.step_start_time = 0            # 踏步开始时间

        # ==== 检测遥控器启动信号 ====
        # 当按下Start键时，进入原地踏步模式
        if self.remote_controller.button[KeyMap.start] != 1 and not self.step_in_place and self.counter == 1:
            print("start按键值为：", self.remote_controller.button[KeyMap.start])
            self.step_in_place = True
            self.auto_walk = False
            self.step_start_time = time.time()  # 记录开始时间

        # ==== 状态机逻辑 ====
        if self.step_in_place:
            # 原地踏步阶段（持续3秒）
            print(f"[Step In Place] Remaining: {3.0 - (time.time()-self.step_start_time):.1f}s")
            if time.time() - self.step_start_time < 3.0:
                # 覆盖遥控器指令：强制速度为0（原地踏步）
                self.cmd[0] = 0.0  # 前进速度
                self.cmd[1] = 0.0  # 横向速度
                self.cmd[2] = 0.0  # 转向速度
            else:
                # 5秒后切换为自动行走
                self.step_in_place = False
                self.auto_walk = True

        elif self.auto_walk:
            # 自动向前行走阶段
            print("[Auto Walk] Forward speed: 0.2m/s")
            self.cmd[0] = 0.2  # 设置前进速度为0.2m/s
            self.cmd[1] = 0.0   # 横向速度
            self.cmd[2] = 0.0   # 转向速度

        else:
            print("按键值为：", self.remote_controller.button[KeyMap.start])
            # 默认状态：直接读取遥控器指令
            self.cmd[0] = self.remote_controller.ly
            self.cmd[1] = self.remote_controller.lx * -1
            self.cmd[2] = self.remote_controller.rx * -1

        # 下面的是官方源码（传感器数据读取与观测构建）

        # 获取当前关节的位置和速度
        for i in range(len(self.config.leg_joint2motor_idx)):
            # 读取关节位置（q）
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            # 读取关节速度（dq）
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z 获取IMU的四元数（姿态）和角速度
        quat = self.low_state.imu_state.quaternion  # 四元数（w, x, y, z）
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)  # 角速度

        # 如果IMU安装在躯干上（如H1机器人），需要将IMU数据转换到骨盆坐标系
        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            # 获取腰部关节的角度和角速度
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            # 转换IMU数据到骨盆坐标系
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation 构建观测向量
        gravity_orientation = get_gravity_orientation(quat) # 计算重力方向投影
        qj_obs = self.qj.copy() # 关节位置
        dqj_obs = self.dqj.copy() # 关节速度
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale # 标准化关节位置
        dqj_obs = dqj_obs * self.config.dof_vel_scale # 标准化关节速度
        ang_vel = ang_vel * self.config.ang_vel_scale # 标准化角速度
        # # 在观测构建后打印关键数据（放在这个位置有问题）
        # print(f"CMD: {self.cmd} | Phase: {phase:.2f} | Action: {self.action[:3]}")

        # 带==的为新增代码
        # ==== 动态调整步态周期 ====
        # 原地踏步和行走时使用更短的周期（0.6秒），默认周期0.8秒
        period = 0.6 if (self.step_in_place or self.auto_walk) else 0.8
        phase = (self.counter * self.config.control_dt) % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        # ==== 观测向量更新 ====
        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd  # 注意此处cmd可能被覆盖
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        self.obs[9 + num_actions * 3] = sin_phase
        self.obs[9 + num_actions * 3 + 1] = cos_phase

        # ==== 策略网络推理 ====
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # # 计算步态相位信号（用于周期性运动，如踏步）
        # period = 0.8  # 步态周期（秒）
        # count = self.counter * self.config.control_dt  # 当前时间
        # phase = count % period / period  # 相位（0到1之间）
        # sin_phase = np.sin(2 * np.pi * phase)  # 正弦相位信号
        # cos_phase = np.cos(2 * np.pi * phase)  # 余弦相位信号

        # # 从遥控器读取指令（前进/后退、横向移动、转向）
        # self.cmd[0] = self.remote_controller.ly  # 前进/后退
        # self.cmd[1] = self.remote_controller.lx * -1  # 横向移动
        # self.cmd[2] = self.remote_controller.rx * -1  # 转向

        # # 构建观测向量（observation）
        # num_actions = self.config.num_actions  # 动作维度
        # self.obs[:3] = ang_vel  # 角速度
        # self.obs[3:6] = gravity_orientation  # 重力方向投影
        # self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd  # 遥控器指令
        # self.obs[9 : 9 + num_actions] = qj_obs  # 标准化关节位置
        # self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs  # 标准化关节速度
        # self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action  # 上一时刻的动作
        # self.obs[9 + num_actions * 3] = sin_phase  # 正弦相位信号
        # self.obs[9 + num_actions * 3 + 1] = cos_phase  # 余弦相位信号

        # # Get the action from the policy network 使用策略网络生成动作
        # obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)  # 将观测转换为PyTorch张量
        # self.action = self.policy(obs_tensor).detach().numpy().squeeze()  # 推理生成动作
        
        # # transform action to target_dof_pos 将动作转换为目标关节角度
        # target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd 构建控制命令（low cmd）
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.obs[9 + num_actions * 3] = sin_phase  
            self.obs[9 + num_actions * 3 + 1] = cos_phase

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]  # 获取电机索引
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]  # 目标关节位置
            self.low_cmd.motor_cmd[motor_idx].qd = 0  # 目标关节速度（设为0）
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]  # 比例增益（刚度）
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]  # 微分增益（阻尼）
            self.low_cmd.motor_cmd[motor_idx].tau = 0  # 前馈力矩（设为0）

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command 发送控制命令到机器人
        self.send_cmd(self.low_cmd)

        # # ==== 安全停止检测 ====
        # if self.auto_walk and self.remote_controller.button[KeyMap.select] == 1:
        #     print("select的按键值为：", self.remote_controller.button[KeyMap.select])
        #     self.auto_walk = False
        #     self.cmd = np.array([0.0, 0.0, 0.0])  # 急停

        # 按控制周期休眠（通常为0.002秒，对应500Hz）
        time.sleep(self.config.control_dt)    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
