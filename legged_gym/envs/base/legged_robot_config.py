from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    # 基础环境配置
    class env:
        num_envs = 4096             #robots number 并行环境数量（4096个机器人同时训练）
        # num_observations = 48       # 观测空间维度
        num_observations = 56       # 增加观测空间维度
        num_privileged_obs = None   # 是否使用特权观测（None表示不使用）（if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise） 
        # num_actions = 12            # 动作空间维度（对应12个关节）
        num_actions = 22            # 动作空间维度（对应12个关节），增加10个手臂自由度
        env_spacing = 3.            # 环境之间的最小间隙（平面地形时有效）not used with heightfields/trimeshes 
        send_timeouts = True        # send time out information to the algorithm
        episode_length_s = 20       # 每个回合的最大时长（秒）episode length in seconds
        test = False

    # 地形生成配置
    class terrain:
        mesh_type = 'plane' #地形网格类型 "heightfield" # none, plane（平面）, heightfield（高度场） or trimesh（三角网格）
        horizontal_scale = 0.1 #水平缩放比例 [m] 水平分辨率（每个网格0.1米）
        vertical_scale = 0.005 #垂直缩放比例 [m] 垂直分辨率（每个高度单位0.005米）
        border_size = 25 #边界大小 [m]
        curriculum = True #是否应用课程学习方法（逐步增加难度）
        static_friction = 1.0 # 地面静态摩擦系数
        dynamic_friction = 1.0 # 地面动态摩擦系数
        restitution = 0. # 弹性恢复系数
        # rough terrain only:仅粗糙地形相关
        measure_heights = True# 是否测量高度
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 在1mx1.6m矩形范围内测量的点的x坐标（不包括中心线）
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]# 测量点的y坐标
        selected = False # select a unique terrain type and pass all arguments 是否选择唯一的地形类型并传递所有参数
        terrain_kwargs = None # 为所选地形类型指定的参数字典
        max_init_terrain_level = 5 # 课程学习开始的最大初始地形等级
        terrain_length = 8. # 地形长度，单位：米
        terrain_width = 8. # 地形宽度，单位：米
        num_rows= 20 # 地形行数（等级）
        num_cols = 20 # 地形列数（类型）
        # 地形类型：[平滑斜坡，粗糙斜坡，上楼梯，下楼梯，离散]
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        # trimesh（三角形网格）仅适用：
        slope_treshold = 0.75 # 斜坡阈值，高于此阈值的斜坡将被修正为垂直表面

    # 运动指令配置
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # 指令维度：[线速度x, 线速度y, 角速度yaw, 朝向] default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # 指令更新间隔（秒）time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # 前/后速度范围[-1,1]m/s min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # 横向速度范围 min max [m/s]
            ang_vel_yaw = [-1, 1]    # 偏航角速度范围 min max [rad/s]
            heading = [-3.14, 3.14]

    # 初始状态配置
    class init_state:
        pos = [0.0, 0.0, 1.] # 初始位置（x,y,z）单位：米 x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # 初始四元数姿态 x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # 关节初始角度（动作=0时的目标角度）target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    # 关节控制参数
    class control:
        control_type = 'P' # 控制类型：P(位置)/V(速度)/T(扭矩) P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]

        # # 尝试让机器人跳起来
        # stiffness = {'joint_a': 100.0, 'joint_b': 100.0}  # 关节刚度（单位：N·m/rad）[N*m/rad]
        # damping = {'joint_a': 10.0, 'joint_b': 10.0}     # 关节阻尼（单位：N·m·s/rad）[N*m*s/rad] 

        # 动作缩放系数：目标角度 = action * scale + 默认角度 action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # 控制频率降采样因子（策略更新频率=仿真频率/4）decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    # 机器人模型参数
    class asset:
        file = ""               # 机器人URDF/MJCF模型文件路径
        name = "legged_robot"  # actor name
        foot_name = "None" # 足端刚体名称（用于接触检测）name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # 合并固定连接的刚体 merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # 是否固定基座（调试时可用）fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 是否启用自碰撞检测 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01 # 碰撞体厚度（单位：米）

    # 域随机化配置
    class domain_rand:
        randomize_friction = True # 是否随机化地面摩擦
        friction_range = [0.5, 1.25] # 摩擦系数随机范围
        randomize_base_mass = False 
        added_mass_range = [-1., 1.]
        push_robots = True # 是否随机推动机器人
        push_interval_s = 15
        max_push_vel_xy = 1. # 最大随机推动速度（m/s）

    # 奖励函数配置
    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0  # 线速度跟踪奖励权重
            tracking_ang_vel = 0.5  # 角速度跟踪奖励权重
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0    # 足端空中时间奖励权重
            collision = -1.         # 碰撞惩罚权重
            feet_stumble = -0.0     
            action_rate = -0.01     # 动作变化率惩罚
            stand_still = -0.

        only_positive_rewards = True # 是否裁剪负奖励为0 if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # # 速度跟踪奖励的高斯方差参数 tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    # 观测标准化配置
    class normalization:
        class obs_scales:
            lin_vel = 2.0               # 线速度观测的缩放因子
            ang_vel = 0.25              # 角速度观测的缩放因子
            dof_pos = 1.0               # 关节位置观测的缩放因子
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.        # 观测值裁剪范围 [-100,100]
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

# PPO算法参数
class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]         # 策略网络隐藏层维度
        critic_hidden_dims = [512, 256, 128]        # 价值网络隐藏层维度
        activation = 'elu' # 激活函数类型 can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2        # PPO裁剪参数
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 # 学习率 5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99            # 折扣因子
        lam = 0.95              # GAE参数
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # # 每个环境采样的步数 per iteration
        max_iterations = 1500 # # 最大训练迭代次数 number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

"""
# create a directory to clone
mkdir ~/git && cd ~/git
# clone a repository with URDF files
git clone git@github.com:isaac-orbit/anymal_d_simple_description.git

cd IsaacLab
conda activate isaaclab
python source/standalone/tools/convert_urdf.py \
  ~/git/anymal_d_simple_description/urdf/anymal.urdf \
  source/extensions/omni.isaac.lab_assets/data/Robots/ANYbotics/anymal_d.usd \
  --merge-joints \
  --make-instanceable

"""