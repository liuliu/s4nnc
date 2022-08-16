import NNC

/// The helper to collect data from the given Envs.
///
/// The policy is intentional to be flexible as a simple closure. We collect both the action and other
/// data the closure provides. There may be some type conversions for what the Env expects and what
/// the policy provides, thus, need to specify both. To make this easier to use, the placeholder
/// type would be useful: `Collector<Float, (), _, _>(envs: envs) { ... }`
public struct Collector<Element: TensorNumeric, OtherType, EnvType: Env, EnvElement: TensorNumeric>
where EnvType.ObsType == Tensor<EnvElement>, EnvType.ActType == Tensor<EnvElement> {
  public typealias ObsType = Tensor<Element>
  public typealias ActType = Tensor<Element>
  var envs: [EnvType]
  var batch: [CollectedData<Element, OtherType>]
  var finalizedBatch: [CollectedData<Element, OtherType>]
  let policy: (_: ObsType) -> (ActType, OtherType)
  public init(envs: [EnvType], policy: @escaping (_: ObsType) -> (ActType, OtherType)) {
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

public struct CollectedData<Element: TensorNumeric, OtherType> {
  public typealias ObsType = Tensor<Element>
  public typealias ActType = Tensor<Element>
  public var lastObservation: ObsType?
  public var actions: [ActType]
  public var rewards: [Float]
  public var others: [OtherType]
  public var episodeReward: Float
  public var episodeLength: Int
  public init(lastObservation: ObsType?) {
    self.lastObservation = lastObservation
    actions = []
    rewards = []
    others = []
    episodeReward = 0
    episodeLength = 0
  }
  mutating func reset() {
    actions.removeAll()
    rewards.removeAll()
    others.removeAll()
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

  public var data: [CollectedData<Element, OtherType>] {
    finalizedBatch + batch.filter { $0.actions.count > 0 }
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
        let (action, other) = policy(obs)
        let (newObs, reward, done, info) = envs[i].step(action: EnvType.ActType(from: action))
        batch[i].actions.append(action)
        batch[i].others.append(other)
        batch[i].rewards.append(reward)
        batch[i].lastObservation = ObsType(from: newObs)
        batch[i].episodeReward += reward
        batch[i].episodeLength += 1
        if done {
          episodeRewards.append(batch[i].episodeReward)
          if info["TimeLimit.truncated"] as? Bool? != true {
            batch[i].lastObservation = nil
          }
          episodeLengths.append(Float(batch[i].episodeLength))
          let (newObs, _) = envs[i].reset()
          finalizedBatch.append(batch[i])
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
