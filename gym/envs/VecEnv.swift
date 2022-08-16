import Dispatch
import NNC

public final class VecEnv<EnvType: Env, Element: TensorNumeric>
where EnvType.ActType == Tensor<Element>, EnvType.ObsType == Tensor<Element> {
  private var envs = [EnvType]()
  private var terminated = [Bool]()
  private var obs = [EnvType.ObsType]()
  private var rewards = [EnvType.RewardType]()
  public init(count: Int, _ closure: (_: Int) throws -> EnvType) rethrows {
    precondition(count > 0)
    envs = []
    terminated = []
    for i in 0..<count {
      envs.append(try closure(i))
      terminated.append(false)
    }
  }
}

extension VecEnv: Env where EnvType.TerminatedType == Bool {
  public typealias ActType = EnvType.ActType
  public typealias ObsType = EnvType.ObsType
  public typealias RewardType = [EnvType.RewardType]
  public typealias TerminatedType = [Bool]
  public func step(action: ActType) -> (ObsType, RewardType, TerminatedType, [String: Any]) {
    if obs.count == 0 || rewards.count == 0 {  // If we never done obs, we need to build up the array, do it serially. The reason because I cannot construct the array with optional types easily.
      obs = []
      rewards = []
      for i in 0..<envs.count {
        assert(!self.terminated[i])
        let (obs, reward, terminated, _) = envs[i].step(action: action[i, ...])
        self.obs.append(obs)
        self.rewards.append(reward)
        self.terminated[i] = terminated
      }
    } else {  // Once we built up, we can do it concurrently.
      DispatchQueue.concurrentPerform(iterations: envs.count) { [self] i in
        let (obs, reward, terminated, _) = envs[i].step(action: action[i, ...])
        self.obs[i] = obs
        self.rewards[i] = reward
        self.terminated[i] = terminated
      }
    }
    var obs = Tensor<Element>(
      self.obs[0].kind, format: self.obs[0].format,
      shape: [envs.count, self.obs[0].shape[0]])
    for i in 0..<envs.count {
      obs[i, ...] = self.obs[i]
    }
    return (obs, rewards, terminated, [:])
  }

  public func reset(seed: Int?) -> (ObsType, [String: Any]) {
    if let seed = seed {
      var sfmt = SFMT(seed: UInt64(bitPattern: Int64(seed)))
      if obs.count == 0 {
        for i in 0..<envs.count {
          let (obs, _) = envs[i].reset(seed: Int(bitPattern: sfmt.next()))
          self.obs.append(obs)
        }
      } else {
        let seeds = (0..<envs.count).map { _ in Int(bitPattern: sfmt.next()) }
        DispatchQueue.concurrentPerform(iterations: envs.count) { [self] i in
          let (obs, _) = envs[i].reset(seed: seeds[i])
          self.obs[i] = obs
        }
      }
    } else {
      if obs.count == 0 {
        for i in 0..<envs.count {
          let (obs, _) = envs[i].reset(seed: nil)
          self.obs.append(obs)
        }
      } else {
        DispatchQueue.concurrentPerform(iterations: envs.count) { [self] i in
          let (obs, _) = envs[i].reset(seed: nil)
          self.obs[i] = obs
        }
      }
    }
    var obs = Tensor<Element>(
      self.obs[0].kind, format: self.obs[0].format,
      shape: [envs.count, self.obs[0].shape[0]])
    for i in 0..<envs.count {
      obs[i, ...] = self.obs[i]
      terminated[i] = false
    }
    return (obs, [:])
  }

  public static var rewardThreshold: Float { EnvType.rewardThreshold }
  public static var actionSpace: [ClosedRange<Float>] { EnvType.actionSpace }
  public static var stateSize: Int { EnvType.stateSize }
}
