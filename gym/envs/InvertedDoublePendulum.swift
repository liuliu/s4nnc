import Foundation
import MuJoCo
import NNC
import NNCMuJoCoConversion
import Numerics

public final class InvertedDoublePendulum: MuJoCoEnv {
  public let model: MjModel
  public var data: MjData

  private let initData: MjData

  private var sfmt: SFMT

  public init() throws {
    if let runfilesDir = ProcessInfo.processInfo.environment["RUNFILES_DIR"] {
      model = try MjModel(
        fromXMLPath: runfilesDir + "/s4nnc/gym/assets/inverted_double_pendulum.xml")
    } else {
      model = try MjModel(fromXMLPath: "../s4nnc/gym/assets/inverted_double_pendulum.xml")
    }
    data = model.makeData()
    initData = data.copied(model: model)
    var g = SystemRandomNumberGenerator()
    sfmt = SFMT(seed: g.next())
  }
}

extension InvertedDoublePendulum: Env {
  public typealias ActType = Tensor<Float64>
  public typealias ObsType = Tensor<Float64>
  public typealias RewardType = Float
  public typealias TerminatedType = Bool

  private var isHealthy: Bool {
    let y = data.siteXpos[2]
    return y > 1
  }

  private var terminated: Bool {
    return !isHealthy
  }

  private func observations() -> Tensor<Float64> {
    let qpos = data.qpos
    let qvel = data.qvel
    let qfrcConstraint = data.qfrcConstraint
    var tensor = Tensor<Float64>(.CPU, .C(11))
    tensor[0] = qpos[0]
    tensor[1] = sin(qpos[1])
    tensor[2] = sin(qpos[2])
    tensor[3] = cos(qpos[2])
    tensor[4] = cos(qpos[2])
    tensor[5] = min(max(qvel[0], -10), 10)
    tensor[6] = min(max(qvel[1], -10), 10)
    tensor[7] = min(max(qvel[2], -10), 10)
    tensor[8] = min(max(qfrcConstraint[0], -10), 10)
    tensor[9] = min(max(qfrcConstraint[1], -10), 10)
    tensor[10] = min(max(qfrcConstraint[2], -10), 10)
    return tensor
  }

  public func step(action: ActType) -> (ObsType, RewardType, TerminatedType, [String: Any]) {
    data.ctrl[...] = action
    for _ in 0..<5 {
      model.step(data: &data)
    }
    // As of MuJoCo 2.0, force-related quantities like cacc are not computed
    // unless there's a force sensor in the model.
    // See https://github.com/openai/gym/issues/1541
    model.rnePostConstraint(data: &data)
    let obs = observations()
    let x = data.siteXpos[0]
    let y = data.siteXpos[2]
    let distPenality = 0.01 * x * x + (y - 2) * (y - 2)
    let v1 = data.qvel[1]
    let v2 = data.qvel[2]
    let velPenality = 1e-3 * v1 * v1 + 5e-3 * v2 * v2
    let reward = Float(10 - distPenality - velPenality)
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
      qpos[i] = initQpos[i] + Double.random(in: -0.1...0.1, using: &sfmt)
    }
    for i in 0..<qvel.count {
      qvel[i] = initQvel[i] + noise(0.1, using: &sfmt)
    }
    // After this, forward data to finish reset.
    model.forward(data: &data)
    let obs = observations()
    return (obs, [:])
  }

  public static var rewardThreshold: Float { 9_100 }
  public static var actionSpace: [ClosedRange<Float>] { Array(repeating: -1...1, count: 1) }
  public static var stateSize: Int { 11 }
}
