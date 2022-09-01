import Foundation
import MuJoCo
import NNC
import NNCMuJoCoConversion
import Numerics

/// ### Description
/// This environment is the cartpole environment based on the work done by
/// Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can
/// solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
/// just like in the classic environments but now powered by the Mujoco physics simulator -
/// allowing for more complex experiments (such as varying the effects of gravity).
/// This environment involves a cart that can moved linearly, with a pole fixed on it
/// at one end and having another end free. The cart can be pushed left or right, and the
/// goal is to balance the pole on the top of the cart by applying forces on the cart.
/// ### Action Space
/// The agent take a 1-element vector for actions.
/// The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
/// the numerical force applied to the cart (with magnitude representing the amount of
/// force and sign representing the direction)
/// | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
/// |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
/// | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |
/// ### Observation Space
/// The state space consists of positional values of different body parts of
/// the pendulum system, followed by the velocities of those individual parts (their derivatives)
/// with all the positions ordered before all the velocities.
/// The observation is a `ndarray` with shape `(4,)` where the elements correspond to the following:
/// | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Unit                      |
/// | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
/// | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
/// | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
/// | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
/// | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | anglular velocity (rad/s) |
/// ### Rewards
/// The goal is to make the inverted pendulum stand upright (within a certain angle limit)
/// as long as possible - as such a reward of +1 is awarded for each timestep that
/// the pole is upright.
/// ### Starting State
/// All observations start in state
/// (0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
/// of [-0.01, 0.01] added to the values for stochasticity.
/// ### Episode End
/// The episode ends when any of the following happens:
/// 1. Truncation: The episode duration reaches 1000 timesteps.
/// 2. Termination: Any of the state space values is no longer finite.
/// 3. Termination: The absolutely value of the vertical angle between the pole and the cart is greater than 0.2 radian.
/// ### Arguments
/// No additional arguments are currently supported.
/// ```
/// env = gym.make('InvertedPendulum-v4')
/// ```
/// There is no v3 for InvertedPendulum, unlike the robot environments where a
/// v3 and beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.
/// ### Version History
/// * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
/// * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
/// * v2: All continuous control environments now use mujoco_py >= 1.50
/// * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum)
/// * v0: Initial versions release (1.0.0)
public final class InvertedPendulum: MuJoCoEnv {
  public let model: MjModel
  public var data: MjData

  private let initData: MjData

  private var sfmt: SFMT

  public init() throws {
    if let runfilesDir = ProcessInfo.processInfo.environment["RUNFILES_DIR"] {
      model = try MjModel(fromXMLPath: runfilesDir + "/s4nnc/gym/assets/inverted_pendulum.xml")
    } else {
      model = try MjModel(fromXMLPath: "../s4nnc/gym/assets/inverted_pendulum.xml")
    }
    data = model.makeData()
    initData = data.copied(model: model)
    var g = SystemRandomNumberGenerator()
    sfmt = SFMT(seed: g.next())
  }
}

extension InvertedPendulum: Env {
  public typealias ActType = Tensor<Float64>
  public typealias ObsType = Tensor<Float64>
  public typealias RewardType = Float
  public typealias TerminatedType = Bool

  private var isHealthy: Bool {
    let qpos = data.qpos
    let y = qpos[1]
    for i in 0..<qpos.count {
      if qpos[i].isInfinite {
        return false
      }
    }
    let qvel = data.qvel
    for i in 0..<qvel.count {
      if qvel[i].isInfinite {
        return false
      }
    }
    return abs(y) <= 0.2
  }

  private var terminated: Bool {
    return !isHealthy
  }

  private func observations() -> Tensor<Float64> {
    let qpos = data.qpos
    let qvel = data.qvel
    var tensor = Tensor<Float64>(.CPU, .C(4))
    tensor[0..<2] = qpos[...]
    tensor[2..<4] = qvel[...]
    return tensor
  }

  public func step(action: ActType) -> (ObsType, RewardType, TerminatedType, [String: Any]) {
    data.ctrl[...] = action
    for _ in 0..<2 {
      model.step(data: &data)
    }
    // As of MuJoCo 2.0, force-related quantities like cacc are not computed
    // unless there's a force sensor in the model.
    // See https://github.com/openai/gym/issues/1541
    model.rnePostConstraint(data: &data)
    let obs = observations()
    let reward: Float = 1.0
    return (obs, reward, terminated, [:])
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
      qpos[i] = initQpos[i] + Double.random(in: -0.01...0.01, using: &sfmt)
    }
    for i in 0..<qvel.count {
      qvel[i] = initQvel[i] + Double.random(in: -0.01...0.01, using: &sfmt)
    }
    // After this, forward data to finish reset.
    model.forward(data: &data)
    let obs = observations()
    return (obs, [:])
  }

  public static var rewardThreshold: Float { 950 }
  public static var actionSpace: [ClosedRange<Float>] { Array(repeating: -3...3, count: 1) }
  public static var stateSize: Int { 4 }
}
