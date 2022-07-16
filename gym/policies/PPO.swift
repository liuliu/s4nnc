import NNC

public struct PPO {

  static func computeEpisodicGAEReturns(
    values: [Float], rewards: [Float], rewStd: Float, gamma: Float = 0.99, gaeGamma: Float = 0.95
  ) -> (
    advantages: [Float], returns: [Float]
  ) {
    let delta: [Float] = rewards.enumerated().map { (i: Int, rew: Float) -> Float in
      rew + (values[i + 1] * gamma - values[i]) * rewStd
    }
    var gae: Float = 0
    let advantages: [Float] = delta.reversed().map({ (delta: Float) -> Float in
      gae = delta + gamma * gaeGamma * gae
      return gae
    }).reversed()
    let unnormalizedReturns = advantages.enumerated().map { i, adv in
      values[i] * rewStd + adv
    }
    return (advantages, unnormalizedReturns)
  }

  public struct ContinuousActionSpace {
    public var centroid: Tensor<Float>
    public var observation: Tensor<Float>
    public init(centroid: Tensor<Float>, observation: Tensor<Float>) {
      self.centroid = centroid
      self.observation = observation
    }
  }

  private var rewMean: Double = 0
  private var rewVar: Double = 1
  private var rewTotal: Int = 0
  private let graph: DynamicGraph
  private let critic: (_: Tensor<Float>) -> Tensor<Float>

  public init(graph: DynamicGraph, critic: @escaping (_: Tensor<Float>) -> Tensor<Float>) {
    self.graph = graph
    self.critic = critic
  }
}

extension PPO {
  public struct Statistics {
    public var rewardsNormalization: NumericalStatistics
    init(rewardsNormalization: NumericalStatistics) {
      self.rewardsNormalization = rewardsNormalization
    }
  }

  public var statistics: Statistics {
    return Statistics(
      rewardsNormalization: NumericalStatistics(
        mean: Float(rewMean), std: Float(rewVar.squareRoot())))
  }
}

extension PPO {
  public func distributions(
    scale: Tensor<Float>, from batch: [CollectedData<Float, ContinuousActionSpace>]
  ) -> [[Tensor<Float>]] {
    let scaleVar = graph.constant(scale)
    let expScale = Functional.exp(scaleVar)
    let var2 = 1 / (2 * (expScale .* expScale))
    var resultDistributions = [[Tensor<Float>]]()
    for data in batch {
      var distributions = [Tensor<Float>]()
      for (i, other) in data.others.enumerated() {
        let mu = graph.constant(other.centroid)
        let action = graph.constant(data.actions[i])
        let distOld = ((mu - action) .* (mu - action) .* var2 + scaleVar).rawValue.copied()
        distributions.append(distOld)
      }
      resultDistributions.append(distributions)
    }
    return resultDistributions
  }

  public mutating func computeReturns(from batch: [CollectedData<Float, ContinuousActionSpace>])
    -> (returns: [[Float]], advantages: [[Float]])
  {
    var resultReturns = [[Float]]()
    var resultAdvatanges = [[Float]]()
    for data in batch {
      var rewStd: Float = 1
      var invStd: Float = 1
      if rewTotal > 0 {
        rewStd = Float(rewVar.squareRoot()) + 1e-5
        invStd = 1.0 / rewStd
      }
      // Recompute value with critics.
      var values = [Float]()
      for other in data.others {
        let value = critic(other.observation)
        values.append(value[0])
      }
      let value = critic(data.lastObservation)
      values.append(value[0])
      let (advantages, unnormalizedReturns) = Self.computeEpisodicGAEReturns(
        values: values, rewards: data.rewards, rewStd: rewStd)
      var returns = [Float]()
      var batchMean: Double = 0
      for unnormalizedReturn in unnormalizedReturns {
        returns.append(invStd * unnormalizedReturn)
        batchMean += Double(unnormalizedReturn)
      }
      resultReturns.append(returns)
      resultAdvatanges.append(advantages)
      var batchVar: Double = 0
      batchMean = batchMean / Double(unnormalizedReturns.count)
      for rew in unnormalizedReturns {
        batchVar += (Double(rew) - batchMean) * (Double(rew) - batchMean)
      }
      batchVar = batchVar / Double(unnormalizedReturns.count)
      let delta = batchMean - rewMean
      let totalCount = unnormalizedReturns.count + rewTotal
      rewMean = rewMean + delta * Double(unnormalizedReturns.count) / Double(totalCount)
      let mA = rewVar * Double(rewTotal)
      let mB = batchVar * Double(unnormalizedReturns.count)
      let m2 =
        mA + mB + delta * delta * Double(rewTotal) * Double(unnormalizedReturns.count)
        / Double(totalCount)
      rewVar = m2 / Double(totalCount)
      rewTotal = totalCount
    }
    return (returns: resultReturns, advantages: resultAdvatanges)
  }

}
