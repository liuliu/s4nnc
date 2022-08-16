public protocol Env {
  associatedtype ObsType
  associatedtype ActType
  associatedtype RewardType
  associatedtype TerminatedType
  mutating func step(action: ActType) -> (ObsType, RewardType, TerminatedType, [String: Any])
  mutating func reset(seed: Int?) -> (ObsType, [String: Any])
  static var rewardThreshold: Float { get }
  static var stateSize: Int { get }
  static var actionSpace: [ClosedRange<Float>] { get }
}

extension Env {
  public mutating func reset() -> (ObsType, [String: Any]) {
    return reset(seed: nil)
  }
}
