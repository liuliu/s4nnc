import NNC

/// The helper to collect data from the given Envs.
///
/// The policy is intentional to be flexible as a simple closure. We collect both the action and state
/// data the closure provides. There may be some type conversions for what the Env expects and what
/// the policy provides, thus, need to specify both. To make this easier to use, the placeholder
/// type would be useful: `Collector<Float, (), _, _>(envs: envs) { ... }`
public struct Collector<Element: TensorNumeric, StateType, EnvType: Env, EnvElement: TensorNumeric>
where EnvType.ObsType == Tensor<EnvElement>, EnvType.ActType == Tensor<EnvElement> {
  public typealias ObsType = Tensor<Element>
  public typealias ActType = Tensor<Element>
  var envs: [EnvType]
  var batch: [CollectedData<Element, StateType>]
  var finalizedBatch: [CollectedData<Element, StateType>]
  let policy: (_: ObsType) -> (ActType, StateType)
  public init(envs: [EnvType], policy: @escaping (_: ObsType) -> (ActType, StateType)) {
    self.envs = envs
    self.policy = policy
    batch = []
    for i in 0..<envs.count {
      let (obs, _) = self.envs[i].reset(seed: i)
      batch.append(CollectedData(lastObservation: ObsType(from: obs)))
    }
    finalizedBatch = []
  }
}

public struct CollectedData<Element: TensorNumeric, StateType> {
  public typealias ObsType = Tensor<Element>
  public typealias ActType = Tensor<Element>
  public var lastObservation: ObsType?
  public var rewards: [Float]
  public var states: [StateType]
  public var episodeReward: Float
  public var episodeLength: Int
  public init(lastObservation: ObsType?) {
    self.lastObservation = lastObservation
    rewards = []
    states = []
    episodeReward = 0
    episodeLength = 0
  }
  mutating func reset() {
    rewards.removeAll()
    states.removeAll()
  }
}

extension Collector {
  public struct Statistics {
    public var episodeCount: Int
    public var stepCount: Int
    public var episodeReward: NumericalStatistics
    public var episodeLength: NumericalStatistics
    init(
      episodeCount: Int, stepCount: Int, episodeReward: NumericalStatistics,
      episodeLength: NumericalStatistics
    ) {
      self.episodeCount = episodeCount
      self.stepCount = stepCount
      self.episodeReward = episodeReward
      self.episodeLength = episodeLength
    }
  }

  public mutating func resetData() {
    for i in 0..<batch.count {
      batch[i].reset()
    }
    finalizedBatch.removeAll()
  }

  public mutating func reset() {
    for i in 0..<envs.count {
      batch[i].reset()
      let (newObs, _) = envs[i].reset()
      batch[i].lastObservation = ObsType(from: newObs)
    }
  }

  public var data: [CollectedData<Element, StateType>] {
    finalizedBatch + batch.filter { $0.rewards.count > 0 }
  }
}

extension Collector where EnvType.TerminatedType == Bool, EnvType.RewardType == Float {
  public mutating func collect(nStep: Int) -> Statistics {
    var episodeCount = 0
    var stepCount = 0
    var episodeRewards = [Float]()
    var episodeLengths = [Float]()
    while stepCount < nStep {
      for i in 0..<envs.count {
        let obs = batch[i].lastObservation!
        let (action, state) = policy(obs)
        let (newObs, reward, done, info) = envs[i].step(action: EnvType.ActType(from: action))
        batch[i].states.append(state)
        batch[i].rewards.append(reward)
        batch[i].lastObservation = ObsType(from: newObs)
        batch[i].episodeReward += reward
        batch[i].episodeLength += 1
        if done {
          if info["TimeLimit.truncated"] as? Bool? != true {
            batch[i].lastObservation = nil
          }
          episodeRewards.append(batch[i].episodeReward)
          episodeLengths.append(Float(batch[i].episodeLength))
          finalizedBatch.append(batch[i])
          let (newObs, _) = envs[i].reset()
          batch[i].reset()
          batch[i].lastObservation = ObsType(from: newObs)
          batch[i].episodeReward = 0
          batch[i].episodeLength = 0
          episodeCount += 1
        }
      }
      stepCount += envs.count
    }
    return Statistics(
      episodeCount: episodeCount, stepCount: stepCount,
      episodeReward: NumericalStatistics(episodeRewards),
      episodeLength: NumericalStatistics(episodeLengths))
  }
}
