import Foundation
import MuJoCo
import NNC
import NNCMuJoCoConversion
import Numerics

/// ### Description
/// This environment builds on the hopper environment based on the work done by Erez, Tassa, and Todorov
/// in ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf)
/// by adding another set of legs making it possible for the robot to walker forward instead of
/// hop. Like other Mujoco environments, this environment aims to increase the number of independent state
/// and control variables as compared to the classic control environments. The walker is a
/// two-dimensional two-legged figure that consist of four main body parts - a single torso at the top
/// (with the two legs splitting after the torso), two thighs in the middle below the torso, two legs
/// in the bottom below the thighs, and two feet attached to the legs on which the entire body rests.
/// The goal is to make coordinate both sets of feet, legs, and thighs to move in the forward (right)
/// direction by applying torques on the six hinges connecting the six body parts.
/// ### Action Space
/// The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
/// | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
/// |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
/// | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
/// | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
/// | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
/// | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
/// | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
/// | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |
/// ### Observation Space
/// Observations consist of positional values of different body parts of the walker,
/// followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.
/// By default, observations do not include the x-coordinate of the top. It may
/// be included by passing `exclude_current_positions_from_observation=False` during construction.
/// In that case, the observation space will have 18 dimensions where the first dimension
/// represent the x-coordinates of the top of the walker.
/// Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
/// of the top will be returned in `info` with key `"x_position"`.
/// By default, observation is a `ndarray` with shape `(17,)` where the elements correspond to the following:
/// | Num | Observation                                      | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
/// | --- | ------------------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
/// | 0   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz (torso)                    | slide | position (m)             |
/// | 1   | angle of the top                                 | -Inf | Inf | rooty (torso)                    | hinge | angle (rad)              |
/// | 2   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
/// | 3   | angle of the leg joint                           | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
/// | 4   | angle of the foot joint                          | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
/// | 5   | angle of the left thigh joint                    | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
/// | 6   | angle of the left leg joint                      | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
/// | 7   | angle of the left foot joint                     | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
/// | 8   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
/// | 9   | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
/// | 10  | angular velocity of the angle of the top         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
/// | 11  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
/// | 12  | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
/// | 13  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
/// | 14  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
/// | 15  | angular velocity of the leg hinge                | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
/// | 16  | angular velocity of the foot hinge               | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |
/// ### Rewards
/// The reward consists of three parts:
/// - *healthy_reward*: Every timestep that the walker is alive, it receives a fixed reward of value `healthy_reward`,
/// - *forward_reward*: A reward of walking forward which is measured as
/// *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*.
/// *dt* is the time between actions and is dependeent on the frame_skip parameter
/// (default is 4), where the frametime is 0.002 - making the default
/// *dt = 4 * 0.002 = 0.008*. This reward would be positive if the walker walks forward (right) desired.
/// - *ctrl_cost*: A cost for penalising the walker if it
/// takes actions that are too large. It is measured as
/// *`ctrl_cost_weight` * sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is
/// a parameter set for the control and has a default value of 0.001
/// The total reward returned is ***reward*** *=* *healthy_reward bonus + forward_reward - ctrl_cost* and `info` will also contain the individual reward terms
/// ### Starting State
/// All observations start in state
/// (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
/// with a uniform noise in the range of [-`reset_noise_scale`, `reset_noise_scale`] added to the values for stochasticity.
/// ### Episode End
/// The walker is said to be unhealthy if any of the following happens:
/// 1. Any of the state space values is no longer finite
/// 2. The height of the walker is ***not*** in the closed interval specified by `healthy_z_range`
/// 3. The absolute value of the angle (`observation[1]` if `exclude_current_positions_from_observation=False`, else `observation[2]`) is ***not*** in the closed interval specified by `healthy_angle_range`
/// If `terminate_when_unhealthy=True` is passed during construction (which is the default),
/// the episode ends when any of the following happens:
/// 1. Truncation: The episode duration reaches a 1000 timesteps
/// 2. Termination: The walker is unhealthy
/// If `terminate_when_unhealthy=False` is passed, the episode is ended only when 1000 timesteps are exceeded.
/// ### Arguments
/// No additional arguments are currently supported in v2 and lower.
/// ```
/// env = gym.make('Walker2d-v4')
/// ```
/// v3 and beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.
/// ```
/// env = gym.make('Walker2d-v4', ctrl_cost_weight=0.1, ....)
/// ```
/// | Parameter                                    | Type      | Default          | Description                                                                                                                                                       |
/// | -------------------------------------------- | --------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
/// | `xml_file`                                   | **str**   | `"walker2d.xml"` | Path to a MuJoCo model                                                                                                                                            |
/// | `forward_reward_weight`                      | **float** | `1.0`            | Weight for _forward_reward_ term (see section on reward)                                                                                                          |
/// | `ctrl_cost_weight`                           | **float** | `1e-3`           | Weight for _ctr_cost_ term (see section on reward)                                                                                                                |
/// | `healthy_reward`                             | **float** | `1.0`            | Constant reward given if the ant is "healthy" after timestep                                                                                                      |
/// | `terminate_when_unhealthy`                   | **bool**  | `True`           | If true, issue a done signal if the z-coordinate of the walker is no longer healthy                                                                               |
/// | `healthy_z_range`                            | **tuple** | `(0.8, 2)`       | The z-coordinate of the top of the walker must be in this range to be considered healthy                                                                          |
/// | `healthy_angle_range`                        | **tuple** | `(-1, 1)`        | The angle must be in this range to be considered healthy                                                                                                          |
/// | `reset_noise_scale`                          | **float** | `5e-3`           | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                    |
/// | `exclude_current_positions_from_observation` | **bool**  | `True`           | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |
/// ### Version History
/// * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
/// * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
/// * v2: All continuous control environments now use mujoco_py >= 1.50
/// * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
/// * v0: Initial versions release (1.0.0)
public final class Walker2D: MuJoCoEnv {
  public let model: MjModel
  public var data: MjData

  private let initData: MjData
  private let forwardRewardWeight: Double
  private let ctrlCostWeight: Double
  private let healthyReward: Double
  private let terminateWhenUnhealthy: Bool
  private let healthyZRange: ClosedRange<Double>
  private let healthyAngleRange: ClosedRange<Double>
  private let resetNoiseScale: Double

  private var sfmt: SFMT

  public init(
    forwardRewardWeight: Double = 1.0, ctrlCostWeight: Double = 1e-3, healthyReward: Double = 1.0,
    terminateWhenUnhealthy: Bool = true, healthyZRange: ClosedRange<Double> = 0.8...2,
    healthyAngleRange: ClosedRange<Double> = -1...1, resetNoiseScale: Double = 5e-3
  ) throws {
    if let runfilesDir = ProcessInfo.processInfo.environment["RUNFILES_DIR"] {
      model = try MjModel(
        fromXMLPath: runfilesDir + "/s4nnc/gym/assets/walker2d.xml")
    } else {
      model = try MjModel(fromXMLPath: "../s4nnc/gym/assets/walker2d.xml")
    }
    data = model.makeData()
    initData = data.copied(model: model)
    var g = SystemRandomNumberGenerator()
    sfmt = SFMT(seed: g.next())
    self.forwardRewardWeight = forwardRewardWeight
    self.ctrlCostWeight = ctrlCostWeight
    self.healthyReward = healthyReward
    self.terminateWhenUnhealthy = terminateWhenUnhealthy
    self.healthyZRange = healthyZRange
    self.healthyAngleRange = healthyAngleRange
    self.resetNoiseScale = resetNoiseScale
  }
}

extension Walker2D: Env {
  public typealias ActType = Tensor<Float64>
  public typealias ObsType = Tensor<Float64>
  public typealias RewardType = Float
  public typealias TerminatedType = Bool

  private var isHealthy: Bool {
    let qpos = data.qpos
    let z = qpos[1]
    let angle = qpos[2]
    return healthyZRange.contains(z) && healthyAngleRange.contains(angle)
  }

  private var terminated: Bool {
    return !isHealthy ? terminateWhenUnhealthy : false
  }

  private func observations() -> Tensor<Float64> {
    let qpos = data.qpos
    let qvel = data.qvel
    var tensor = Tensor<Float64>(.CPU, .C(17))
    tensor[0..<8] = qpos[1...]
    for i in 0..<qvel.count {
      tensor[8 + i] = max(min(qvel[i], 10), -10)
    }
    return tensor
  }

  public func step(action: ActType) -> (ObsType, RewardType, TerminatedType, [String: Any]) {
    data.ctrl[...] = action
    let xPositionBefore = data.qpos[0]
    for _ in 0..<4 {
      model.step(data: &data)
    }
    // As of MuJoCo 2.0, force-related quantities like cacc are not computed
    // unless there's a force sensor in the model.
    // See https://github.com/openai/gym/issues/1541
    model.rnePostConstraint(data: &data)
    let xPositionAfter = data.qpos[0]
    let dt = model.opt.timestep * 4
    let xVelocity = (xPositionAfter - xPositionBefore) / dt
    var ctrlCost: Double = 0
    for i in 0..<6 {
      ctrlCost += Double(action[i] * action[i])
    }
    ctrlCost *= ctrlCostWeight
    let forwardReward = forwardRewardWeight * xVelocity
    let healthyReward = terminateWhenUnhealthy || isHealthy ? self.healthyReward : 0
    let rewards = forwardReward + healthyReward
    let costs = ctrlCost
    let obs = observations()
    let reward = Float(rewards - costs)
    let info: [String: Any] = [
      "x_position": xPositionAfter,
      "x_velocity": xVelocity,
    ]
    return (obs, reward, terminated, info)
  }

  public func reset(seed: Int?) -> (ObsType, [String: Any]) {
    let initQpos = initData.qpos
    let initQvel = initData.qvel
    var qpos = data.qpos
    var qvel = data.qvel
    if let seed = seed {
      sfmt = SFMT(seed: UInt64(bitPattern: Int64(seed)))
    }
    for i in 0..<qpos.count {
      qpos[i] = initQpos[i] + Double.random(in: -resetNoiseScale...resetNoiseScale, using: &sfmt)
    }
    for i in 0..<qvel.count {
      qvel[i] = initQvel[i] + Double.random(in: -resetNoiseScale...resetNoiseScale, using: &sfmt)
    }
    // After this, forward data to finish reset.
    model.forward(data: &data)
    let obs = observations()
    return (obs, [:])
  }

  public static var rewardThreshold: Float { 4_000 }
  public static var actionSpace: [ClosedRange<Float>] { Array(repeating: -1...1, count: 6) }
  public static var stateSize: Int { 17 }
}
