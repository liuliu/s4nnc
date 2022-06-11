import MuJoCo
import NNC

final class MuJoCoEnv {
  let model: MjModel
  public init(modelPath: String) throws {
    model = try MjModel(fromXMLPath: modelPath)
  }
}

extension MuJoCoEnv: Env {
  public typealias ActType = Tensor<Float64>
  public typealias ObsType = Tensor<Float64>

  public func step(action: ActType) -> (ObsType, Float, Bool, [String: Any]?)? {
    return nil
  }

  public func reset(seed: Int?) -> (ObsType, [String: Any]?)? {
    return nil
  }

  public func render() {
  }
}
