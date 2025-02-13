from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    """
    pos: 定义了机器人初始的位(位置）,其中 x、y 和 z 分别为机器人在三个方向上的位置。默认设置为 pos = [0.0, 0.0, 0.8],表示机器人初始位置位于地面上方 0.8 米处。
    default_joint_angles: 定义了机器人各个关节在初始状态下的角(以弧度为单位）。这些值表示机器人在静止状态下的关节角度。
    """
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,
           'torso_joint' : 0.,

        #    'waist_yaw_joint' : 0,
        #    'left_shoulder_pitch_joint':0,
        #    'left_shoulder_yaw_joint':0,
        #    'left_shoulder_roll_joint':0,
        #    'left_elbow_joint':0,
        #    'left_wrist_roll_joint':0,
        #    'right_shoulder_pitch_joint':0,
        #    'right_shoulder_yaw_joint':0,
        #    'right_shoulder_roll_joint':0,
        #    'right_elbow_joint':0,
        #    'right_wrist_roll_joint':0,
        }
    
    """
    num_observations: 观测空间的维度数,这里设置为 47。
    num_privileged_obs: 特权观测空间的维度数,这里为 50。
    num_actions: 机器人控制的动作数量,这里设置为 12,表示机器人的动作空间有 12 个维度。
    """
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12

    """
    randomize_friction: 随机化地面摩擦系数,默认为 True,这有助于训练中环境的多样性。
    friction_range: 摩擦系数的范围,范围从 0.1 到 1.25。
    randomize_base_mass: 随机化机器人的基础质量,默认为 True。
    added_mass_range: 机器人的附加质量范围,范围从 -1.0 到 3.0。
    push_robots: 是否推动机器人,默认为 True。
    push_interval_s: 推动机器人间隔时间,单位为秒,这里为 5 秒。
    max_push_vel_xy: 最大推动速度,单位为米/秒,设置为 1.5。
    """
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      
    """
    control_type: 设置为 'P',表示使用比例控(Proportional Control）。
    stiffness: 各关节的刚(N·m/rad）,包括髋部、膝部和踝部等。
    damping: 各关节的阻(N·m·s/rad）,包括髋部、膝部  和踝部等。
    action_scale: 动作缩放因子,这里为 0.25,表示动作的目标角度为 actionScale * action + defaultAngle。
    decimation: 控制动作更新的次数,每个仿真周期更新的控制动作次数为 4。
    """
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    """
    file: 机器人 URDF 文件路径,这里指向 g1_12dof.urdf。
    name: 机器人名称,这里是 "g1"。
    foot_name: 机器人脚部名称,这里是 "ankle_roll"。
    penalize_contacts_on: 设置哪些部位与环境的接触会受到惩罚,这里设置为 "hip" 和 "knee"。
    terminate_after_contacts_on: tra设置哪个部位接触后会终止仿真,这里为 "pelvis"。
    self_collisions: 是否启用自碰撞检测,这里设置为 0,表示启用。
    flip_visual_attachments: 是否翻转视觉附件,默认为 False。
    dof 23 比 12 多下面两个：
    waist_yaw_joint waist_support_joint
    """
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    """
    soft_dof_pos_limit: 关节角度的软限制,设置为 0.9。
    base_height_target: 机器人基座的目标高度,设置为 0.78 米。
    scales: 各种奖励项的权重比例,包括：
        tracking_lin_vel: 平移速度的奖励比例,设为 1.0。
        tracking_ang_vel: 角速度的奖励比例,设为 0.5。
        lin_vel_z: Z轴方向上的线性速度奖励,设为 -2.0。
        ang_vel_xy: XY平面上的角速度奖励,设为 -0.05。
        orientation: 机器人朝向的奖励,设为 -1.0。
        其他与脚部、接触、动作频率等相关的奖励项。
    """
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

"""
这个类定义了与 PP(Proximal Policy Optimization）相关的配置,继承自 LeggedRobotCfgPPO。

policy 类
    init_noise_std: 初始化噪声标准差,设为 0.8,用于初始化策略网络的权重。
    actor_hidden_dims: Actor 网络的隐藏层维度,这里设置为 [32]。
    critic_hidden_dims: Critic 网络的隐藏层维度,这里设置为 [32]。
    activation: 激活函数,设置为 'elu'。
    rnn_type: RNN 类型,选择了 'lstm'。
    rnn_hidden_size: LSTM 隐藏层的大小,设置为 64。
    rnn_num_layers: LSTM 的层数,设置为 1。
algorithm 类
    entropy_coef: 熵系数,设置为 0.01,用于平衡探索与利用。
runner 类
    policy_class_name: 策略类的名称,这里是 "ActorCriticRecurrent"。
    max_iterations: 最大迭代次数,设为 10000。
    run_name: 运行名称,设置为空字符串 ''。
    experiment_name: 实验名称,设置为 'g1'。
"""
class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        # max_iterations = 10000
        max_iterations = 1500
        run_name = ''
        experiment_name = 'g1'

  
