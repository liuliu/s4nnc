import Foundation
import MuJoCo
import NNC
import NNCMuJoCoConversion
import Numerics

public final class Humanoid: MuJoCoEnv {
  public let model: MjModel
  public var data: MjData

  private let initData: MjData
  private let forwardRewardWeight: Double
  private let ctrlCostWeight: Double
  private let healthyReward: Double
  private let terminateWhenUnhealthy: Bool
  private let healthyZRange: ClosedRange<Double>
  private let resetNoiseScale: Double

  private var sfmt: SFMT

  public init(
    forwardRewardWeight: Double = 1.25,
    ctrlCostWeight: Double = 0.1, healthyReward: Double = 5.0, terminateWhenUnhealthy: Bool = true,
    healthyZRange: ClosedRange<Double> = 1.0...2.0, resetNoiseScale: Double = 0.01
  ) throws {
    if let runfilesDir = ProcessInfo.processInfo.environment["RUNFILES_DIR"] {
      model = try MjModel(fromXMLPath: runfilesDir + "/s4nnc/gym/assets/humanoid.xml")
    } else {
      model = try MjModel(fromXMLPath: "../s4nnc/gym/assets/humanoid.xml")
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
    self.resetNoiseScale = resetNoiseScale
  }
}

extension Humanoid: Env {
  public typealias ActType = Tensor<Float64>
  public typealias ObsType = Tensor<Float64>
  public typealias RewardType = Float
  public typealias TerminatedType = Bool

  private var isHealthy: Bool {
    let qpos = data.qpos
    let z = qpos[2]
    return healthyZRange.contains(z)
  }

  private var terminated: Bool {
    return !isHealthy ? terminateWhenUnhealthy : false
  }

  private func observations() -> Tensor<Float64> {
    var tensor = Tensor<Float64>(.CPU, .C(376))
    tensor[0..<22] = data.qpos[2...]
    tensor[22..<45] = data.qvel[...]
    tensor[45..<185] = data.cinert[...]
    tensor[185..<269] = data.cvel[...]
    tensor[269..<292] = data.qfrcActuator[...]
    tensor[292..<376] = data.cfrcExt[...]
    return tensor
  }

  private func massCenter() -> (Double, Double) {
    let bodyMass = model.bodyMass
    let xipos = data.xipos
    var xSum: Double = 0
    var ySum: Double = 0
    var sum: Double = 0
    for i in 0..<Int(model.nbody) {
      xSum += xipos[i * 3] * bodyMass[i]
      ySum += xipos[i * 3 + 1] * bodyMass[i]
      sum += bodyMass[i]
    }
    return (xSum / sum, ySum / sum)
  }

  public func step(action: ActType) -> (ObsType, RewardType, TerminatedType, [String: Any]) {
    let id = model.name2id(type: .body, name: "torso")
    precondition(id >= 0)
    let xyPositionBefore = massCenter()
    data.ctrl[...] = action
    for _ in 0..<5 {
      model.step(data: &data)
    }
    // As of MuJoCo 2.0, force-related quantities like cacc are not computed
    // unless there's a force sensor in the model.
    // See https://github.com/openai/gym/issues/1541
    model.rnePostConstraint(data: &data)
    let xyPositionAfter = massCenter()
    let dt = model.opt.timestep * 5
    let xyVelocity = (
      (xyPositionAfter.0 - xyPositionBefore.0) / dt, (xyPositionAfter.1 - xyPositionBefore.1) / dt
    )
    let forwardReward = self.forwardRewardWeight * xyVelocity.0
    let healthyReward = terminateWhenUnhealthy || isHealthy ? self.healthyReward : 0
    var ctrlCost: Double = 0
    for i in 0..<17 {
      ctrlCost += Double(action[i] * action[i])
    }
    ctrlCost *= ctrlCostWeight
    let rewards = forwardReward + healthyReward
    let costs = ctrlCost
    let obs = observations()
    let reward = Float(rewards - costs)
    let info: [String: Any] = [
      "reward_forward": forwardReward,
      "reward_ctrl": -ctrlCost,
      "reward_alive": healthyReward,
      "x_position": xyPositionAfter.0,
      "y_position": xyPositionAfter.1,
      "distance_from_origin":
        (xyPositionAfter.0 * xyPositionAfter.0 + xyPositionAfter.1 * xyPositionAfter.1)
        .squareRoot(),
      "x_velocity": xyVelocity.0,
      "y_velocity": xyVelocity.1,
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

  public static var rewardThreshold: Float { 1_000 }
  public static var actionSpace: [ClosedRange<Float>] { Array(repeating: -0.4...0.4, count: 17) }
  public static var stateSize: Int { 376 }
}
