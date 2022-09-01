import Foundation
import MuJoCo
import NNC
import NNCMuJoCoConversion
import Numerics

/// ### Description
/// This environment corresponds to the Swimmer environment described in Rémi Coulom's PhD thesis
/// ["Reinforcement Learning Using Neural Networks, with Applications to Motor Control"](https://tel.archives-ouvertes.fr/tel-00003985/document).
/// The environment aims to increase the number of independent state and control
/// variables as compared to the classic control environments. The swimmers
/// consist of three or more segments ('***links***') and one less articulation
/// joints ('***rotors***') - one rotor joint connecting exactly two links to
/// form a linear chain. The swimmer is suspended in a two dimensional pool and
/// always starts in the same position (subject to some deviation drawn from an
/// uniform distribution), and the goal is to move as fast as possible towards
/// the right by applying torque on the rotors and using the fluids friction.
/// ### Notes
/// The problem parameters are:
/// Problem parameters:
/// * *n*: number of body parts
/// * *m<sub>i</sub>*: mass of part *i* (*i* ∈ {1...n})
/// * *l<sub>i</sub>*: length of part *i* (*i* ∈ {1...n})
/// * *k*: viscous-friction coefficient
/// While the default environment has *n* = 3, *l<sub>i</sub>* = 0.1,
/// and *k* = 0.1. It is possible to pass a custom MuJoCo XML file during construction to increase the
/// number of links, or to tweak any of the parameters.
/// ### Action Space
/// The action space is a `Box(-1, 1, (2,), float32)`. An action represents the torques applied between *links*
/// | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
/// |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
/// | 0   | Torque applied on the first rotor  | -1          | 1           | motor1_rot                       | hinge | torque (N m) |
/// | 1   | Torque applied on the second rotor | -1          | 1           | motor2_rot                       | hinge | torque (N m) |
/// ### Observation Space
/// By default, observations consists of:
/// * θ<sub>i</sub>: angle of part *i* with respect to the *x* axis
/// * θ<sub>i</sub>': its derivative with respect to time (angular velocity)
/// In the default case, observations do not include the x- and y-coordinates of the front tip. These may
/// be included by passing `exclude_current_positions_from_observation=False` during construction.
/// Then, the observation space will have 10 dimensions where the first two dimensions
/// represent the x- and y-coordinates of the front tip.
/// Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x- and y-coordinates
/// will be returned in `info` with keys `"x_position"` and `"y_position"`, respectively.
/// By default, the observation is a `ndarray` with shape `(8,)` where the elements correspond to the following:
/// | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
/// | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
/// | 0   | angle of the front tip               | -Inf | Inf | free_body_rot                    | hinge | angle (rad)              |
/// | 1   | angle of the first rotor             | -Inf | Inf | motor1_rot                       | hinge | angle (rad)              |
/// | 2   | angle of the second rotor            | -Inf | Inf | motor2_rot                       | hinge | angle (rad)              |
/// | 3   | velocity of the tip along the x-axis | -Inf | Inf | slider1                          | slide | velocity (m/s)           |
/// | 4   | velocity of the tip along the y-axis | -Inf | Inf | slider2                          | slide | velocity (m/s)           |
/// | 5   | angular velocity of front tip        | -Inf | Inf | free_body_rot                    | hinge | angular velocity (rad/s) |
/// | 6   | angular velocity of first rotor      | -Inf | Inf | motor1_rot                       | hinge | angular velocity (rad/s) |
/// | 7   | angular velocity of second rotor     | -Inf | Inf | motor2_rot                             | hinge | angular velocity (rad/s) |
/// ### Rewards
/// The reward consists of two parts:
/// - *forward_reward*: A reward of moving forward which is measured
/// as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*. *dt* is
/// the time between actions and is dependent on the frame_skip parameter
/// (default is 4), where the frametime is 0.01 - making the
/// default *dt = 4 * 0.01 = 0.04*. This reward would be positive if the swimmer
/// swims right as desired.
/// - *ctrl_cost*: A cost for penalising the swimmer if it takes
/// actions that are too large. It is measured as *`ctrl_cost_weight` *
/// sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is a parameter set for the
/// control and has a default value of 1e-4
/// The total reward returned is ***reward*** *=* *forward_reward - ctrl_cost* and `info` will also contain the individual reward terms
/// ### Starting State
/// All observations start in state (0,0,0,0,0,0,0,0) with a Uniform noise in the range of [-`reset_noise_scale`, `reset_noise_scale`] is added to the initial state for stochasticity.
/// ### Episode End
/// The episode truncates when the episode length is greater than 1000.
/// ### Arguments
/// No additional arguments are currently supported in v2 and lower.
/// ```
/// gym.make('Swimmer-v4')
/// ```
/// v3 and v4 take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.
/// ```
/// env = gym.make('Swimmer-v4', ctrl_cost_weight=0.1, ....)
/// ```
/// | Parameter                                    | Type      | Default         | Description                                                                                                                                                               |
/// | -------------------------------------------- | --------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
/// | `xml_file`                                   | **str**   | `"swimmer.xml"` | Path to a MuJoCo model                                                                                                                                                    |
/// | `forward_reward_weight`                      | **float** | `1.0`           | Weight for _forward_reward_ term (see section on reward)                                                                                                                  |
/// | `ctrl_cost_weight`                           | **float** | `1e-4`          | Weight for _ctrl_cost_ term (see section on reward)                                                                                                                       |
/// | `reset_noise_scale`                          | **float** | `0.1`           | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                            |
/// | `exclude_current_positions_from_observation` | **bool**  | `True`          | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |
/// ### Version History
/// * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
/// * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
/// * v2: All continuous control environments now use mujoco_py >= 1.50
/// * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
/// * v0: Initial versions release (1.0.0)
public final class Swimmer: MuJoCoEnv {
  public let model: MjModel
  public var data: MjData

  private let initData: MjData
  private let forwardRewardWeight: Double
  private let ctrlCostWeight: Double
  private let resetNoiseScale: Double

  private var sfmt: SFMT

  public init(
    forwardRewardWeight: Double = 1.0, ctrlCostWeight: Double = 1e-4, resetNoiseScale: Double = 0.1
  ) throws {
    if let runfilesDir = ProcessInfo.processInfo.environment["RUNFILES_DIR"] {
      model = try MjModel(fromXMLPath: runfilesDir + "/s4nnc/gym/assets/swimmer.xml")
    } else {
      model = try MjModel(fromXMLPath: "../s4nnc/gym/assets/swimmer.xml")
    }
    data = model.makeData()
    initData = data.copied(model: model)
    var g = SystemRandomNumberGenerator()
    sfmt = SFMT(seed: g.next())
    self.forwardRewardWeight = forwardRewardWeight
    self.ctrlCostWeight = ctrlCostWeight
    self.resetNoiseScale = resetNoiseScale
  }
}

extension Swimmer: Env {
  public typealias ActType = Tensor<Float64>
  public typealias ObsType = Tensor<Float64>
  public typealias RewardType = Float
  public typealias TerminatedType = Bool

  private func observations() -> Tensor<Float64> {
    let qpos = data.qpos
    let qvel = data.qvel
    var tensor = Tensor<Float64>(.CPU, .C(8))
    tensor[0..<3] = qpos[2...]
    tensor[3..<8] = qvel[...]
    return tensor
  }

  public func step(action: ActType) -> (ObsType, RewardType, TerminatedType, [String: Any]) {
    let qpos = data.qpos
    let xyPositionBefore = (qpos[0], qpos[1])
    data.ctrl[...] = action
    for _ in 0..<4 {
      model.step(data: &data)
    }
    // As of MuJoCo 2.0, force-related quantities like cacc are not computed
    // unless there's a force sensor in the model.
    // See https://github.com/openai/gym/issues/1541
    model.rnePostConstraint(data: &data)
    let xyPositionAfter = (qpos[0], qpos[1])
    let dt = model.opt.timestep * 4
    let xyVelocity = (
      (xyPositionAfter.0 - xyPositionBefore.0) / dt, (xyPositionAfter.1 - xyPositionBefore.1) / dt
    )
    let forwardReward = forwardRewardWeight * xyVelocity.0
    var ctrlCost: Double = 0
    for i in 0..<2 {
      ctrlCost += Double(action[i] * action[i])
    }
    ctrlCost *= ctrlCostWeight
    let obs = observations()
    let reward = Float(forwardReward - ctrlCost)
    let info: [String: Any] = [
      "reward_forward": forwardReward,
      "reward_ctrl": -ctrlCost,
      "x_position": xyPositionAfter.0,
      "y_position": xyPositionAfter.1,
      "distance_from_origin":
        (xyPositionAfter.0 * xyPositionAfter.0 + xyPositionAfter.1 * xyPositionAfter.1)
        .squareRoot(),
      "x_velocity": xyVelocity.0,
      "y_velocity": xyVelocity.1,
    ]
    return (obs, reward, false, info)
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
      qvel[i] = initQvel[i] + noise(resetNoiseScale, using: &sfmt)
    }
    // After this, forward data to finish reset.
    model.forward(data: &data)
    let obs = observations()
    return (obs, [:])
  }

  public static var rewardThreshold: Float { 360 }
  public static var actionSpace: [ClosedRange<Float>] { Array(repeating: -1...1, count: 2) }
  public static var stateSize: Int { 8 }
}
