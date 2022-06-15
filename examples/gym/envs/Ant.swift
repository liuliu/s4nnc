import C_sfmt
import MuJoCo
import NNC
import NNCMuJoCoConversion
import Numerics

public final class Ant {
  private let model: MjModel
  private let initData: MjData
  private let ctrlCostWeight: Double
  private let healthyReward: Double
  private let terminateWhenUnhealthy: Bool
  private let healthyZRange: ClosedRange<Double>
  private let resetNoiseScale: Double

  private var sfmt = SFMT(seed: 0)
  private var data: MjData

  public init(
    ctrlCostWeight: Double = 0.5, healthyReward: Double = 0.1, terminateWhenUnhealthy: Bool = true,
    healthyZRange: ClosedRange<Double> = 0.2...1.0, resetNoiseScale: Double = 0.1
  ) throws {
    model = try MjModel(fromXMLPath: "examples/gym/assets/ant.xml")
    data = model.makeData()
    initData = data.copied(model: model)
    self.ctrlCostWeight = ctrlCostWeight
    self.healthyReward = healthyReward
    self.terminateWhenUnhealthy = terminateWhenUnhealthy
    self.healthyZRange = healthyZRange
    self.resetNoiseScale = resetNoiseScale
  }
}

struct SFMT: RandomNumberGenerator {
  private var state: sfmt_t
  init(seed: UInt64) {
    state = sfmt_t()
    sfmt_init_gen_rand(&state, UInt32(truncatingIfNeeded: seed))
  }
  mutating func next() -> UInt64 {
    return sfmt_genrand_uint64(&state)
  }
}

func noise<T: RandomNumberGenerator>(_ std: Double, using: inout T) -> Double {
  let u1 = Double.random(in: 0...1, using: &using)
  let u2 = Double.random(in: 0...1, using: &using)
  let mag = std * (-2.0 * .log(u1)).squareRoot()
  return mag * .cos(.pi * 2 * u2)
}

extension Ant: Env {
  public typealias ActType = Tensor<Float32>
  public typealias ObsType = Tensor<Float32>

  private var isHealthy: Bool {
    let qpos = data.qpos
    let z = qpos[2]
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
    return healthyZRange.contains(z)
  }

  private var done: Bool {
    return !isHealthy ? terminateWhenUnhealthy : false
  }

  private func observations() -> Tensor<Float32> {
    let qpos = data.qpos
    let qvel = data.qvel
    var tensor = Tensor<Float32>(.CPU, .C(27))
    tensor[0..<13] = Tensor(from: qpos[2...])
    tensor[13..<27] = Tensor(from: qvel[...])
    return tensor
  }

  public func step(action: ActType) -> (ObsType, Float, Bool, [String: Any]) {
    let id = model.name2id(type: .body, name: "torso")
    precondition(id >= 0)
    let xyPositionBefore = (data.xpos[Int(id) * 3], data.xpos[Int(id) * 3 + 1])
    data.ctrl[...] = Tensor(from: action)
    model.step(data: &data)
    // As of MuJoCo 2.0, force-related quantities like cacc are not computed
    // unless there's a force sensor in the model.
    // See https://github.com/openai/gym/issues/1541
    model.rnePostConstraint(data: &data)
    let xyPositionAfter = (data.xpos[Int(id) * 3], data.xpos[Int(id) * 3 + 1])
    let dt = model.opt.timestep
    let xyVelocity = (
      (xyPositionAfter.0 - xyPositionBefore.0) / dt, (xyPositionAfter.1 - xyPositionBefore.1) / dt
    )
    let forwardReward = xyVelocity.0
    let healthyReward = isHealthy || terminateWhenUnhealthy ? self.healthyReward : 0
    var ctrlCost: Double = 0
    for i in 0..<8 {
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
      "reward_survive": healthyReward,
      "x_position": xyPositionAfter.0,
      "y_position": xyPositionAfter.1,
      "distance_from_origin":
        (xyPositionAfter.0 * xyPositionAfter.0 + xyPositionAfter.1 * xyPositionAfter.1)
        .squareRoot(),
      "x_velocity": xyVelocity.0,
      "y_velocity": xyVelocity.1,
    ]
    return (obs, reward, done, info)
  }

  public func reset(seed: Int?) -> (ObsType, [String: Any]) {
    let initQpos = initData.qpos
    let initQvel = initData.qvel
    var qpos = data.qpos
    var qvel = data.qvel
    if let seed = seed {
      sfmt = SFMT(seed: UInt64(bitPattern: Int64(seed)))
    } else {
      var g = SystemRandomNumberGenerator()
      sfmt = SFMT(seed: g.next())
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

  public func render() {
    // Need to think through this.
  }
}
