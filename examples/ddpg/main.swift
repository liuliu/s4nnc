import NNC
import NNCPythonConversion
import PythonKit

func Net() -> Model {
  return Model([
    Dense(count: 128), RELU(),
    Dense(count: 128), RELU(),
    Dense(count: 128), RELU(),
    Dense(count: 128), RELU(),
    Dense(count: 1),
  ])
}

let name = "Pendulum-v0"

let gym = Python.import("gym")

let env = gym.make(name)

env.seed(0)

let action_space = env.action_space

let graph = DynamicGraph()

struct Replay {
  var obs: Tensor<Float32>  // The state before action.
  var obs_next: Tensor<Float32>  // The state n_step ahead.
  var rewards: [Float32]  // Rewards for 0..<n_step - 1
  var act: Float  // The act taken in the episode.
  var step: Int  // The step in the episode.
  var step_count: Int  // How many steps til the end, step < step_count.
}

let actor = Net()
let critic = Net()
let criticOld = critic.copy()

var actorOptim = AdamOptimizer(graph, rate: 1e-4)
actorOptim.parameters = [actor.parameters]
var criticOptim = AdamOptimizer(graph, rate: 1e-3)
criticOptim.parameters = [critic.parameters]

let obs = env.reset()
var buffer = [(obs: Tensor<Float32>, reward: Float32, act: Int)]()
var last_obs: Tensor<Float32> = Tensor(from: try! Tensor<Float64>(numpy: obs))
while true {
  let act = Tensor<Float64>([Float64.random(in: -2...2)], .C(1))
  let (obs, reward, done, _) = env.step(act).tuple4
  last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
  print("obs \(obs), reward \(reward)")
  if Bool(done)! {
    let obs = env.reset()
    last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
    break
  }
}

env.close()
