import NNC

public struct PPO {

  static func computeEpisodicReturn(
    vs: [Float], rewards: [Float], rewStd: Float, gamma: Float = 0.99, gaeGamma: Float = 0.95
  ) -> (
    advantages: [Float], returns: [Float]
  ) {
    let delta: [Float] = rewards.enumerated().map { (i: Int, rew: Float) -> Float in
      rew + (vs[i + 1] * gamma - vs[i]) * rewStd
    }
    var gae: Float = 0
    let advantages: [Float] = delta.reversed().map({ (delta: Float) -> Float in
      gae = delta + gamma * gaeGamma * gae
      return gae
    }).reversed()
    let unnormalizedReturns = advantages.enumerated().map { i, adv in
      vs[i] * rewStd + adv
    }
    return (advantages, unnormalizedReturns)
  }

  public init() {
  }

  public func dataframe(from data: CollectedData<Float, ()>) -> DataFrame {
    let dataframe = DataFrame(from: data.actions)
    return dataframe
  }

}
