import MuJoCo

public final class TimeLimit<EnvType: Env> {
  private var env: EnvType
  private let maxEpisodeSteps: Int
  private var elapsedSteps: Int
  public init(env: EnvType, maxEpisodeSteps: Int) {
    self.env = env
    self.maxEpisodeSteps = maxEpisodeSteps
    self.elapsedSteps = 0
  }
}

extension TimeLimit: Env where EnvType.TerminatedType == Bool {
  public typealias ActType = EnvType.ActType
  public typealias ObsType = EnvType.ObsType
  public typealias RewardType = EnvType.RewardType
  public typealias TerminatedType = EnvType.TerminatedType

  public func step(action: ActType) -> (ObsType, RewardType, TerminatedType, [String: Any]) {
    let result = env.step(action: action)
    var (_, _, terminated, info) = result
    elapsedSteps += 1
    if elapsedSteps >= maxEpisodeSteps {
      // TimeLimit.truncated key may have been already set by the environment
      // do not overwrite it
      let episodeTruncated = !terminated || (info["TimeLimit.truncated", default: false] as! Bool)
      info["TimeLimit.truncated"] = episodeTruncated
      terminated = true
    }
    return (result.0, result.1, terminated, info)
  }

  public func reset(seed: Int?) -> (ObsType, [String: Any]) {
    elapsedSteps = 0
    return env.reset(seed: seed)
  }

  public static var rewardThreshold: Float { EnvType.rewardThreshold }
  public static var actionSpace: [ClosedRange<Float>] { EnvType.actionSpace }
  public static var stateSize: Int { EnvType.stateSize }
}

extension TimeLimit: MuJoCoEnv where EnvType: MuJoCoEnv {
  public var model: MjModel { env.model }
  public var data: MjData {
    get { env.data }
    set { env.data = newValue }
  }
}
