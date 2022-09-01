import Foundation
import MuJoCo
import NNC
import NNCMuJoCoConversion
import Numerics

/// ### Description
/// This environment is based on the work done by Erez, Tassa, and Todorov in
/// ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf). The environment aims to
/// increase the number of independent state and control variables as compared to
/// the classic control environments. The hopper is a two-dimensional
/// one-legged figure that consist of four main body parts - the torso at the
/// top, the thigh in the middle, the leg in the bottom, and a single foot on
/// which the entire body rests. The goal is to make hops that move in the
/// forward (right) direction by applying torques on the three hinges
/// connecting the four body parts.
/// ### Action Space
/// The action space is a `Box(-1, 1, (3,), float32)`. An action represents the torques applied between *links*
/// | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
/// |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
/// | 0   | Torque applied on the thigh rotor  | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
/// | 1   | Torque applied on the leg rotor    | -1          | 1           | leg_joint                        | hinge | torque (N m) |
/// | 3   | Torque applied on the foot rotor   | -1          | 1           | foot_joint                       | hinge | torque (N m) |
/// ### Observation Space
/// Observations consist of positional values of different body parts of the
/// hopper, followed by the velocities of those individual parts
/// (their derivatives) with all the positions ordered before all the velocities.
/// By default, observations do not include the x-coordinate of the hopper. It may
/// be included by passing `exclude_current_positions_from_observation=False` during construction.
/// In that case, the observation space will have 12 dimensions where the first dimension
/// represents the x-coordinate of the hopper.
/// Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
/// will be returned in `info` with key `"x_position"`.
/// However, by default, the observation is a `ndarray` with shape `(11,)` where the elements
/// correspond to the following:
/// | Num | Observation                                      | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
/// | --- | ------------------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
/// | 0   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz                            | slide | position (m)             |
/// | 1   | angle of the top                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
/// | 2   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
/// | 3   | angle of the leg joint                           | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
/// | 4   | angle of the foot joint                          | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
/// | 5   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
/// | 6   | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
/// | 7   | angular velocity of the angle of the top         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
/// | 8   | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
/// | 9   | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
/// | 10  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
/// ### Rewards
/// The reward consists of three parts:
/// - *healthy_reward*: Every timestep that the hopper is healthy (see definition in section "Episode Termination"), it gets a reward of fixed value `healthy_reward`.
/// - *forward_reward*: A reward of hopping forward which is measured
/// as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*. *dt* is
/// the time between actions and is dependent on the frame_skip parameter
/// (fixed to 4), where the frametime is 0.002 - making the
/// default *dt = 4 * 0.002 = 0.008*. This reward would be positive if the hopper
/// hops forward (positive x direction).
/// - *ctrl_cost*: A cost for penalising the hopper if it takes
/// actions that are too large. It is measured as *`ctrl_cost_weight` *
/// sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is a parameter set for the
/// control and has a default value of 0.001
/// The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost* and `info` will also contain the individual reward terms
/// ### Starting State
/// All observations start in state
/// (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise
///  in the range of [-`reset_noise_scale`, `reset_noise_scale`] added to the values for stochasticity.
/// ### Episode End
/// The hopper is said to be unhealthy if any of the following happens:
/// 1. An element of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, else `observation[2:]`) is no longer contained in the closed interval specified by the argument `healthy_state_range`
/// 2. The height of the hopper (`observation[0]` if  `exclude_current_positions_from_observation=True`, else `observation[1]`) is no longer contained in the closed interval specified by the argument `healthy_z_range` (usually meaning that it has fallen)
/// 3. The angle (`observation[1]` if  `exclude_current_positions_from_observation=True`, else `observation[2]`) is no longer contained in the closed interval specified by the argument `healthy_angle_range`
/// If `terminate_when_unhealthy=True` is passed during construction (which is the default),
/// the episode ends when any of the following happens:
/// 1. Truncation: The episode duration reaches a 1000 timesteps
/// 2. Termination: The hopper is unhealthy
/// If `terminate_when_unhealthy=False` is passed, the episode is ended only when 1000 timesteps are exceeded.
/// ### Arguments
/// No additional arguments are currently supported in v2 and lower.
/// ```
/// env = gym.make('Hopper-v2')
/// ```
/// v3 and v4 take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.
/// ```
/// env = gym.make('Hopper-v4', ctrl_cost_weight=0.1, ....)
/// ```
/// | Parameter                                    | Type      | Default               | Description                                                                                                                                                                     |
/// | -------------------------------------------- | --------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
/// | `xml_file`                                   | **str**   | `"hopper.xml"`        | Path to a MuJoCo model                                                                                                                                                          |
/// | `forward_reward_weight`                      | **float** | `1.0`                 | Weight for _forward_reward_ term (see section on reward)                                                                                                                        |
/// | `ctrl_cost_weight`                           | **float** | `0.001`               | Weight for _ctrl_cost_ reward (see section on reward)                                                                                                                           |
/// | `healthy_reward`                             | **float** | `1`                   | Constant reward given if the ant is "healthy" after timestep                                                                                                                    |
/// | `terminate_when_unhealthy`                   | **bool**  | `True`                | If true, issue a done signal if the hopper is no longer healthy                                                                                                                 |
/// | `healthy_state_range`                        | **tuple** | `(-100, 100)`         | The elements of `observation[1:]` (if `exclude_current_positions_from_observation=True`, else `observation[2:]`) must be in this range for the hopper to be considered healthy  |
/// | `healthy_z_range`                            | **tuple** | `(0.7, float("inf"))` | The z-coordinate must be in this range for the hopper to be considered healthy                                                                                                  |
/// | `healthy_angle_range`                        | **tuple** | `(-0.2, 0.2)`         | The angle given by `observation[1]` (if `exclude_current_positions_from_observation=True`, else `observation[2]`) must be in this range for the hopper to be considered healthy |
/// | `reset_noise_scale`                          | **float** | `5e-3`                | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                                  |
/// | `exclude_current_positions_from_observation` | **bool**  | `True`                | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies               |
/// ### Version History
/// * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
/// * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
/// * v2: All continuous control environments now use mujoco_py >= 1.50
/// * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
/// * v0: Initial versions release (1.0.0)
public final class Hopper: MuJoCoEnv {
  public let model: MjModel
  public var data: MjData

  private let initData: MjData
  private let forwardRewardWeight: Double
  private let ctrlCostWeight: Double
  private let healthyReward: Double
  private let terminateWhenUnhealthy: Bool
  private let healthyStateRange: ClosedRange<Double>
  private let healthyZRange: ClosedRange<Double>
  private let healthyAngleRange: ClosedRange<Double>
  private let resetNoiseScale: Double

  private var sfmt: SFMT

  public init(
    forwardRewardWeight: Double = 1.0, ctrlCostWeight: Double = 1e-3, healthyReward: Double = 1.0,
    terminateWhenUnhealthy: Bool = true, healthyStateRange: ClosedRange<Double> = -100...100,
    healthyZRange: ClosedRange<Double> = 0.7...Double.infinity,
    healthyAngleRange: ClosedRange<Double> = -0.2...0.2, resetNoiseScale: Double = 5e-3
  ) throws {
    if let runfilesDir = ProcessInfo.processInfo.environment["RUNFILES_DIR"] {
      model = try MjModel(
        fromXMLPath: runfilesDir + "/s4nnc/gym/assets/hopper.xml")
    } else {
      model = try MjModel(fromXMLPath: "../s4nnc/gym/assets/hopper.xml")
    }
    data = model.makeData()
    initData = data.copied(model: model)
    var g = SystemRandomNumberGenerator()
    sfmt = SFMT(seed: g.next())
    self.forwardRewardWeight = forwardRewardWeight
    self.ctrlCostWeight = ctrlCostWeight
    self.healthyReward = healthyReward
    self.terminateWhenUnhealthy = terminateWhenUnhealthy
    self.healthyStateRange = healthyStateRange
    self.healthyZRange = healthyZRange
    self.healthyAngleRange = healthyAngleRange
    self.resetNoiseScale = resetNoiseScale
  }
}

extension Hopper: Env {
  public typealias ActType = Tensor<Float64>
  public typealias ObsType = Tensor<Float64>
  public typealias RewardType = Float
  public typealias TerminatedType = Bool

  private var isHealthy: Bool {
    let qpos = data.qpos
    let z = qpos[1]
    let angle = qpos[2]
    if !healthyZRange.contains(z) || !healthyAngleRange.contains(angle) {
      return false
    }
    for i in 3..<qpos.count {
      if !healthyStateRange.contains(qpos[i]) {
        return false
      }
    }
    let qvel = data.qvel
    for i in 0..<qvel.count {
      if !healthyStateRange.contains(qvel[i]) {
        return false
      }
    }
    return true
  }

  private var terminated: Bool {
    return !isHealthy ? terminateWhenUnhealthy : false
  }

  private func observations() -> Tensor<Float64> {
    let qpos = data.qpos
    let qvel = data.qvel
    var tensor = Tensor<Float64>(.CPU, .C(11))
    tensor[0..<5] = qpos[1...]
    for i in 0..<qvel.count {
      tensor[5 + i] = max(min(qvel[i], 10), -10)
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
    for i in 0..<3 {
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

  public static var rewardThreshold: Float { 3_800 }
  public static var actionSpace: [ClosedRange<Float>] { Array(repeating: -1...1, count: 3) }
  public static var stateSize: Int { 11 }
}
