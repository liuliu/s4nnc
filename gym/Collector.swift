import NNC

public struct Collector<ObsType: AnyTensor, ActType: AnyTensor, OtherType, EnvType: Env>
where EnvType.ObsType: AnyTensor, EnvType.ActType: AnyTensor {
  var envs: [EnvType]
  var batch: [Collector.Data]
  var finalizedBatch: [Collector.Data]
  let policy: (_: ObsType) -> (ActType, OtherType)
  public init(envs: [EnvType], policy: @escaping (_: ObsType) -> (ActType, OtherType)) {
    self.envs = envs
    self.policy = policy
    batch = []
    for i in 0..<envs.count {
      let (obs, _) = self.envs[i].reset(seed: i)
      batch.append(Data(lastObservation: ObsType(from: obs)))
    }
    finalizedBatch = []
  }
}

extension Collector {
  public struct Statistics {
    public var episodeCount: Int
    public var stepCount: Int
    public var episodeReward: NumericalStatistics
    public var episodeLength: NumericalStatistics
    public init(
      episodeCount: Int, stepCount: Int, episodeReward: NumericalStatistics,
      episodeLength: NumericalStatistics
    ) {
      self.episodeCount = episodeCount
      self.stepCount = stepCount
      self.episodeReward = episodeReward
      self.episodeLength = episodeLength
    }
  }

  public struct Data {
    public var lastObservation: ObsType
    public var actions: [ActType]
    public var rewards: [Float]
    public var others: [OtherType]
    public init(lastObservation: ObsType) {
      self.lastObservation = lastObservation
      actions = []
      rewards = []
      others = []
    }
    mutating func reset() {
      actions.removeAll()
      rewards.removeAll()
      others.removeAll()
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

  public var data: [Collector.Data] {
    finalizedBatch + batch.filter { $0.actions.count > 0 }
  }
}

extension Collector where EnvType.DoneType == Bool, EnvType.RewardType == Float {
  public mutating func collect(nStep: Int) -> Statistics {
    var episodeCount = 0
    var stepCount = 0
    var episodeRewards = [Float]()
    var episodeLengths = [Float]()
    while stepCount < nStep {
      for i in 0..<envs.count {
        let obs = batch[i].lastObservation
        let (action, other) = policy(obs)
        let (newObs, reward, done, _) = envs[i].step(action: EnvType.ActType(from: action))
        batch[i].actions.append(action)
        batch[i].others.append(other)
        batch[i].rewards.append(reward)
        batch[i].lastObservation = ObsType(from: newObs)
        if done {
          episodeRewards.append(batch[i].rewards.reduce(0) { $0 + $1 })
          episodeLengths.append(Float(batch[i].rewards.count))
          let (newObs, _) = envs[i].reset()
          finalizedBatch.append(batch[i])
          batch[i].reset()
          batch[i].lastObservation = ObsType(from: newObs)
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
