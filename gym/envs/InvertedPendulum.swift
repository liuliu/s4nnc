import Foundation
import MuJoCo
import NNC
import NNCMuJoCoConversion
import Numerics

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
  public typealias DoneType = Bool

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

  private var done: Bool {
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

  public func step(action: ActType) -> (ObsType, RewardType, DoneType, [String: Any]) {
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
    return (obs, reward, done, [:])
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

  public var rewardThreshold: Float { 950 }
}
