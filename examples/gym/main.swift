import NNC
import NNCPythonConversion
import PythonKit

func Net() -> Model {
  return Model([
    Dense(count: 128), RELU(),
    Dense(count: 128), RELU(),
    Dense(count: 128), RELU(),
    Dense(count: 128), RELU(),
    Dense(count: 2),
  ])
}

let gym = Python.import("gym")

let env = gym.make("CartPole-v0")

env.seed(0)

let action_space = env.action_space

let graph = DynamicGraph()

let net = Net()

let eps_test = 0.05
let eps_train = 0.1
let buffer_size = 20_000
let lr = 1e-3
let gamma = 0.9
let n_step = 3
let target_update_freq = 320
let max_epoch = 10
let step_per_epoch = 1000
let collect_per_step = 10
let batch_size = 64
let alpha = 0.6
let beta = 0.4

struct Replay {
  var obs: Tensor<Float32>
  var obs_next: Tensor<Float32>
}

let lastNet = net.copy()
lastNet.parameters.copy(from: net.parameters)

env.reset()
for epoch in 0..<max_epoch {
  for t in 0..<step_per_epoch {
    var episodes = 0
    var env_step_count = 0
    var buffer = [(obs: Tensor<Float32>, reward: PythonObject, act: Int)]()
    var last_obs = Tensor<Float32>([0, 0, 0, 0], .C(4))
    while env_step_count < collect_per_step {
      let variable = graph.variable(last_obs)
      let output = DynamicGraph.Tensor<Float32>(net(inputs: variable)[0])
      let act = output[0] > output[1] ? 0 : 1
      let (obs, reward, done, _) = env.step(act).tuple4
      last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
      buffer.append((obs: last_obs, reward: reward, act: act))
      if Bool(done)! {
        env.reset()
        episodes += 1
        env_step_count += buffer.count
        // Organizing data into ReplayBuffer.
        buffer.removeAll()
        last_obs = Tensor<Float32>([0, 0, 0, 0], .C(4))
      }
    }
  }
}

env.close()
