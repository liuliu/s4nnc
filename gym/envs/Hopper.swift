import Foundation
import MuJoCo
import NNC
import NNCMuJoCoConversion
import Numerics

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

  public static var rewardThreshold: Float { 3_800 }
  public static var actionSpace: [ClosedRange<Float>] { Array(repeating: -1...1, count: 3) }
  public static var stateSize: Int { 11 }
}
