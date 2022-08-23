import Foundation
import MuJoCo

public final class MuJoCoVideo<EnvType: MuJoCoEnv> {
  var env: EnvType
  var task: Task<Void, Never>? = nil
  var cpugenesis: Double = 0
  var simgenesis: Double = 0
  var framesPerSecond: Int
  var simulate: Simulate
  public init(
    env: EnvType, width: Int = 1280, height: Int = 720, framesPerSecond: Int = 30,
    title: String = "video"
  ) {
    self.env = env
    self.framesPerSecond = framesPerSecond
    simulate = Simulate(width: width, height: height, title: title)
    simulate.use(model: self.env.model, data: &self.env.data)
    simulate.ui0 = false
    simulate.ui1 = false
  }
}

extension MuJoCoVideo: Renderable {
  public func render() {
    if cpugenesis == 0 {
      cpugenesis = GLContext.time
      simgenesis = env.data.time
    }
    let simsync = env.data.time
    simulate.yield()
    var cpusync = GLContext.time
    while simsync - simgenesis >= cpusync - cpugenesis {  // wait until reality catches up with simulation.
      simulate.yield()
      cpusync = GLContext.time
    }
  }
}

extension MuJoCoVideo {
  public func close() {
  }
}
