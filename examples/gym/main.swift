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
  var obs: Tensor<Float32>  // The state before action.
  var obs_next: Tensor<Float32>  // The state n_step ahead.
  var step: Int  // The step in the episode.
  var step_count: Int  // How many steps til the end, step < step_count.
}

let lastNet = net.copy()
var replays = [Replay]()
var netIter = 0

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
        last_obs = Tensor<Float32>([0, 0, 0, 0], .C(4))
        // Organizing data into ReplayBuffer.
        for (i, _) in buffer.enumerated() {
          let obs: Tensor<Float32> = i > 0 ? buffer[i - 1].obs : last_obs
          let replay = Replay(
            obs: obs, obs_next: buffer[min(i + n_step - 1, buffer.count - 1)].obs, step: i,
            step_count: buffer.count)
          replays.append(replay)
        }
        buffer.removeAll()
      }
    }
    // Only update target network at intervals.
    if netIter % target_update_freq == 0 {
      lastNet.parameters.copy(from: net.parameters)
    }
    // Now update the model. First, get some samples out of replay buffer.
    replays.shuffled()
    var obs = Tensor<Float32>(.CPU, .NC(64, 4))
    var obs_next = Tensor<Float32>(.CPU, .NC(64, 4))
    for i in 0..<batch_size {
      let replay = replays[i % replays.count]
      obs[i, ...] = replay.obs[...]
      obs_next[i, ...] = replay.obs[...]
    }
    // Compute the q.
    let obs_next_v = graph.constant(obs_next)
    let act = net(inputs: obs_next_v)
    let target_q = lastNet(inputs: obs_next_v)
    print(target_q)
    print(act)
    netIter += 1
  }
}

env.close()
