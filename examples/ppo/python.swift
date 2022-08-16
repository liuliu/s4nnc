import PythonKit

struct PythonEnv: PythonConvertible {
  public private(set) var pythonObject: PythonObject
  public init(pythonObject: PythonObject) {
    self.pythonObject = pythonObject
  }
}

extension PythonEnv: Env {
  public typealias ActType = Tensor<Float64>
  public typealias ObsType = Tensor<Float64>
  public typealias RewardType = Float
  public typealias DoneType = Bool
  public func step(action: ActType) -> (ObsType, RewardType, DoneType, [String: Any]) {
    let (obs, reward, done, info) = pythonObject.step(action).tuple4
    var newInfo = [String: Any]()
    newInfo["TimeLimit.truncated"] = info.checking["TimeLimit.truncated"].flatMap { Bool($0) }
    return (try! Tensor<Float64>(numpy: obs), Float(reward)!, Bool(done)!, newInfo)
  }
  public func reset(seed: Int?) -> (ObsType, [String: Any]) {
    let obs = pythonObject.reset(seed: seed)
    return (try! Tensor<Float64>(numpy: obs), [:])
  }
  public static var rewardThreshold: Float { 6_000 }
  public static var actionSpace: [ClosedRange<Float>] { fatalError() }
  public static var stateSize: Int { fatalError() }
}
