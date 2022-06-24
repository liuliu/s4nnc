public protocol Env {
  associatedtype ObsType
  associatedtype ActType
  associatedtype RewardType
  associatedtype DoneType
  mutating func step(action: ActType) -> (ObsType, RewardType, DoneType, [String: Any])
  mutating func reset(seed: Int?) -> (ObsType, [String: Any])
  var rewardThreshold: Float { get }
}

extension Env {
  public mutating func reset() -> (ObsType, [String: Any]) {
    return reset(seed: nil)
  }
}
