import Algorithms
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

  public struct ContinuousState {
    public var centroid: Tensor<Float>
    public var action: Tensor<Float>
    public var observation: Tensor<Float>
    public init(centroid: Tensor<Float>, action: Tensor<Float>, observation: Tensor<Float>) {
      self.centroid = centroid
      self.action = action
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
    scale: Tensor<Float>, from batch: [CollectedData<Float, ContinuousState>]
  ) -> [[Tensor<Float>]] {
    let scaleVar = graph.constant(scale)
    let expScale = Functional.exp(scaleVar)
    let var2 = 1 / (2 * (expScale .* expScale))
    var resultDistributions = [[Tensor<Float>]]()
    for data in batch {
      var distributions = [Tensor<Float>]()
      for state in data.states {
        let mu = graph.constant(state.centroid)
        let action = graph.constant(state.action)
        let distOld = ((mu - action) .* (mu - action) .* var2 + scaleVar).rawValue.copied()
        distributions.append(distOld)
      }
      resultDistributions.append(distributions)
    }
    return resultDistributions
  }

  public mutating func computeReturns(from batch: [CollectedData<Float, ContinuousState>])
    -> (returns: [[Float]], advantages: [[Float]])
  {
    var resultReturns = [[Float]]()
    var resultAdvatanges = [[Float]]()
    var resultUnnormalizedReturns = [[Float]]()
    var batchMean: Double = 0
    var batchCount: Int = 0
    for data in batch {
      var rewStd: Float = 1
      var invStd: Float = 1
      if rewTotal > 0 {
        rewStd = Float(rewVar.squareRoot()) + 1e-8
        invStd = 1.0 / rewStd
      }
      // Recompute value with critics.
      var values = [Float]()
      for state in data.states {
        let value = critic(state.observation)
        values.append(value[0])
      }
      if let lastObservation = data.lastObservation {
        let value = critic(lastObservation)
        values.append(value[0])
      } else {
        values.append(0)
      }
      let (advantages, unnormalizedReturns) = Self.computeEpisodicGAEReturns(
        values: values, rewards: data.rewards, rewStd: rewStd)
      var returns = [Float]()
      for unnormalizedReturn in unnormalizedReturns {
        returns.append(invStd * unnormalizedReturn)
        batchMean += Double(unnormalizedReturn)
      }
      resultReturns.append(returns)
      resultAdvatanges.append(advantages)
      resultUnnormalizedReturns.append(unnormalizedReturns)
      batchCount += unnormalizedReturns.count
    }
    batchMean = batchMean / Double(batchCount)
    var batchVar: Double = 0
    for unnormalizedReturns in resultUnnormalizedReturns {
      for rew in unnormalizedReturns {
        batchVar += (Double(rew) - batchMean) * (Double(rew) - batchMean)
      }
    }
    batchVar = batchVar / Double(batchCount)
    let delta = batchMean - rewMean
    let totalCount = batchCount + rewTotal
    rewMean = rewMean + delta * Double(batchCount) / Double(totalCount)
    let mA = rewVar * Double(rewTotal)
    let mB = batchVar * Double(batchCount)
    let m2 = mA + mB + delta * delta * Double(rewTotal) * Double(batchCount) / Double(totalCount)
    rewVar = m2 / Double(totalCount)
    rewTotal = totalCount
    return (returns: resultReturns, advantages: resultAdvatanges)
  }

  private struct Sample {
    var observation: Tensor<Float32>
    var action: Tensor<Float32>
    var advantage: Tensor<Float32>
    var `return`: Tensor<Float32>
    var oldDistribution: Tensor<Float32>
  }

  public static func samples<T: RandomNumberGenerator>(
    from collectedData: [CollectedData<Float, ContinuousState>], episodeCount: Int,
    using generator: inout T, returns: [[Float]], advantages: [[Float]],
    oldDistributions: [[Tensor<Float>]]
  ) -> DataFrame {
    var samples = [Sample]()
    let batch = (0..<collectedData.count).randomSample(count: episodeCount, using: &generator)
    for i in batch {
      let bufferReturns = returns[i]
      let bufferAdvantages = advantages[i]
      let bufferOldDistributions = oldDistributions[i]
      let bufferStates = collectedData[i].states
      for j in 0..<bufferStates.count {
        samples.append(
          Sample(
            observation: collectedData[i].states[j].observation, action: bufferStates[j].action,
            advantage: Tensor([bufferAdvantages[j]], .CPU, .C(1)),
            return: Tensor([bufferReturns[j]], .CPU, .C(1)),
            oldDistribution: bufferOldDistributions[j]))
      }
    }
    var df = DataFrame(from: samples, name: "data")
    df["observations"] = df["data", Sample.self].map(\.observation)
    df["actions"] = df["data", Sample.self].map(\.action)
    df["advantages"] = df["data", Sample.self].map(\.advantage)
    df["returns"] = df["data", Sample.self].map(\.return)
    df["oldDistributions"] = df["data", Sample.self].map(\.oldDistribution)
    return df
  }

  public struct ClipLoss {
    public var epsilon: Float
    public var entropyCoefficient: Float
    public init(epsilon: Float, entropyCoefficient: Float) {
      self.epsilon = epsilon
      self.entropyCoefficient = entropyCoefficient
    }
    public func callAsFunction<T: DynamicGraph.TensorGroup>(
      _ mu: T, oldAction: T, oldDistribution: T, advantages: T, scale: T
    ) -> (T, T, T, T) {
      let expScale = Functional.exp(scale)
      let var2 = 1 / (2 * (expScale .* expScale))
      let dist = ((mu - oldAction) .* (mu - oldAction) .* var2 + scale)
      let ratio = Functional.exp(oldDistribution - dist)
      let surr1 = advantages .* ratio
      let surr2 = advantages .* ratio.clamped((1.0 - epsilon)...(1.0 + epsilon))
      let clipLoss =
        entropyCoefficient * scale.reduced(.mean, axis: [0])
        + Functional.min(surr1, surr2).reduced(.mean, axis: [1])
      return (clipLoss, surr1, surr2, ratio)
    }
  }

}
