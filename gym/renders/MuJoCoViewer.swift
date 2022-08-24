import Foundation
import JupyterDisplay
import MuJoCo
import NIOCore

public protocol MuJoCoEnv {
  var model: MjModel { get }
  var data: MjData { get set }
}

public final class MuJoCoViewer<EnvType: MuJoCoEnv> {
  var env: EnvType
  var cpugenesis: Double = 0
  var simgenesis: Double = 0
  var simulate: Simulate
  let renderServer: HTTPRenderServer
  var httpChannel: Channel? = nil
  var renderCell: Int? = nil
  public init(env: EnvType, width: Int = 1280, height: Int = 720, title: String = "viewer") {
    self.env = env
    simulate = Simulate(width: width, height: height, title: title)
    simulate.use(model: self.env.model, data: &self.env.data)
    simulate.ui0 = false
    simulate.ui1 = false
    renderServer = HTTPRenderServer(simulate, maxWidth: width, maxHeight: height, canResize: false)
  }
}

extension MuJoCoViewer: Renderable {
  public func render() {
    if JupyterDisplay.isEnabled {
      // Check to see if we launched the render server yet.
      if httpChannel == nil {
        httpChannel = try? renderServer.bind(host: "0.0.0.0", port: .random(in: 10_000..<20_000))
          .wait()
      }
      if JupyterDisplay.executionCount != renderCell {
        JupyterDisplay.display(html: renderServer.html)
        JupyterDisplay.flush()
        renderCell = JupyterDisplay.executionCount
      }
    }
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
