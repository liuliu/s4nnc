import Foundation
import MuJoCo

public protocol MuJoCoEnv {
  var model: MjModel { get }
  var data: MjData { get set }
}

public final class MuJoCoViewer<EnvType: MuJoCoEnv> {
  var env: EnvType
  var task: Task<Void, Never>? = nil
  var cpugenesis: Double = 0
  var simgenesis: Double = 0
  var simulate: Simulate
  public init(env: EnvType, width: Int = 1280, height: Int = 720, title: String = "viewer") {
    self.env = env
    simulate = Simulate(width: width, height: height, title: title)
    simulate.use(model: self.env.model, data: &self.env.data)
    simulate.ui0 = false
    simulate.ui1 = false
  }
}

extension MuJoCoViewer: Renderable {
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
