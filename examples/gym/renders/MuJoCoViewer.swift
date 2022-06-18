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
  var lock = os_unfair_lock()
  public init(env: EnvType) {
    self.env = env
    os_unfair_lock_lock(&lock)
  }

  deinit {
    task?.cancel()
  }

  public func runDetachedLoop(width: Int, height: Int) {
    // Run the render loop in a separate detached task.
    guard task == nil else { return }
    task = Task.detached { [self] in
      let glContext = GLContext(width: width, height: height, title: "Viewer")
      glContext.makeCurrent {
        var camera = MjvCamera()
        let vOption = MjvOption()
        os_unfair_lock_lock(&lock)
        var scene = MjvScene(model: env.model, maxgeom: 5_000)
        let context = MjrContext(model: env.model, fontScale: ._100)
        os_unfair_lock_unlock(&lock)
        glContext.runLoop(swapInterval: 1) { width, height in
          let viewport = MjrRect(left: 0, bottom: 0, width: width, height: height)
          os_unfair_lock_lock(&lock)
          scene.updateScene(
            model: env.model, data: &env.data, option: vOption, perturb: nil, camera: &camera)
          os_unfair_lock_unlock(&lock)
          context.render(viewport: viewport, scene: &scene)
          return true
        }
      }
    }
  }
}

extension MuJoCoViewer {
  // The necessary context for the detached task.
  final class ViewerContext {
  }
}

extension MuJoCoViewer: Renderable {
  public func render(width: Int, height: Int) {
    if cpugenesis == 0 {
      cpugenesis = GLContext.time
      simgenesis = env.data.time
    }
    let simsync = env.data.time
    // If haven't, run the render loop in a separate detached task.
    runDetachedLoop(width: width, height: height)
    // Check with last time, and see if we need to wait this much until next time.
    let cpusync = GLContext.time
    if simsync - simgenesis < cpusync - cpugenesis {  // The simulation hasn't caught up with reality.
      // Do nothing, let it run again until simulation catches up.
    } else {
      let cpuwait = max((simsync - simgenesis) - (cpusync - cpugenesis), 0)
      // Unlock, sleep.
      os_unfair_lock_unlock(&lock)
      Thread.sleep(forTimeInterval: cpuwait)
      os_unfair_lock_lock(&lock)
    }
  }
}
