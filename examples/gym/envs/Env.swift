public protocol Env {
  associatedtype ObsType
  associatedtype ActType
  func step(action: ActType) -> (ObsType, Float, Bool, [String: Any]?)
  func reset(seed: Int?) -> (ObsType, [String: Any]?)
  func render()
}

extension Env {
  public func reset() -> (ObsType, [String: Any]?) {
    return reset(seed: nil)
  }
}
