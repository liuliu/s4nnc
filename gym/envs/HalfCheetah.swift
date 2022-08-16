import Foundation
import MuJoCo
import NNC
import NNCMuJoCoConversion
import Numerics

public final class HalfCheetah: MuJoCoEnv {
  public let model: MjModel
  public var data: MjData

  private let initData: MjData
  private let forwardRewardWeight: Double
  private let ctrlCostWeight: Double
  private let resetNoiseScale: Double

  private var sfmt: SFMT

  public init(
    forwardRewardWeight: Double = 1.0, ctrlCostWeight: Double = 0.1, resetNoiseScale: Double = 0.1
  ) throws {
    if let runfilesDir = ProcessInfo.processInfo.environment["RUNFILES_DIR"] {
      model = try MjModel(
        fromXMLPath: runfilesDir + "/s4nnc/gym/assets/half_cheetah.xml")
    } else {
      model = try MjModel(fromXMLPath: "../s4nnc/gym/assets/half_cheetah.xml")
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

extension HalfCheetah: Env {
  public typealias ActType = Tensor<Float64>
  public typealias ObsType = Tensor<Float64>
  public typealias RewardType = Float
  public typealias TerminatedType = Bool

  private func observations() -> Tensor<Float64> {
    let qpos = data.qpos
    let qvel = data.qvel
    var tensor = Tensor<Float64>(.CPU, .C(17))
    tensor[0..<8] = qpos[1...]
    tensor[8..<17] = qvel[...]
    return tensor
  }

  public func step(action: ActType) -> (ObsType, RewardType, TerminatedType, [String: Any]) {
    data.ctrl[...] = action
    let xPositionBefore = data.qpos[0]
    for _ in 0..<5 {
      model.step(data: &data)
    }
    // As of MuJoCo 2.0, force-related quantities like cacc are not computed
    // unless there's a force sensor in the model.
    // See https://github.com/openai/gym/issues/1541
    model.rnePostConstraint(data: &data)
    let xPositionAfter = data.qpos[0]
    let dt = model.opt.timestep * 5
    let xVelocity = (xPositionAfter - xPositionBefore) / dt
    var ctrlCost: Double = 0
    for i in 0..<6 {
      ctrlCost += Double(action[i] * action[i])
    }
    ctrlCost *= ctrlCostWeight
    let forwardReward = forwardRewardWeight * xVelocity
    let obs = observations()
    let reward = Float(forwardReward - ctrlCost)
    let info: [String: Any] = [
      "x_position": xPositionAfter,
      "x_velocity": xVelocity,
      "reward_run": forwardReward,
      "reward_ctrl": -ctrlCost,
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

  public static var rewardThreshold: Float { 4_800 }
  public static var actionSpace: [ClosedRange<Float>] { Array(repeating: -1...1, count: 6) }
  public static var stateSize: Int { 17 }
}
