import NNC
import NNCPythonConversion
import Numerics
import PythonKit

@Sequential
func Net() -> Model {
  Dense(count: 128)
  RELU()
  Dense(count: 128)
  RELU()
  Dense(count: 1)
}

let name = "Pendulum-v0"

let gym = Python.import("gym")

let env = gym.make(name)

env.seed(0)
env.spec.reward_threshold = -250

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

let buffer_size = 20_000
let actor_lr: Float = 1e-4
let critic_lr: Float = 1e-3
let gamma = 0.99
let tau = 0.005
let exploration_noise: Float = 0.1
let epoch = 20
let step_per_epoch = 2400
let collect_per_step = 4
let batch_size = 128
let training_num = 8
let testing_num = 100
let rew_norm = 1
let n_step = 1

func noise(_ std: Float) -> Float {
  let u1 = Float.random(in: 0...1)
  let u2 = Float.random(in: 0...1)
  let mag = std * (-2.0 * .log(u1)).squareRoot()
  return mag * .cos(.pi * 2 * u2)
}

let actor = Net()
let critic = Net()
let criticOld = critic.copy()

var actorOptim = AdamOptimizer(graph, rate: actor_lr)
actorOptim.parameters = [actor.parameters]
var criticOptim = AdamOptimizer(graph, rate: critic_lr)
criticOptim.parameters = [critic.parameters]

let obs = env.reset()
var buffer = [(obs: Tensor<Float32>, reward: Float32, act: Int)]()
var last_obs: Tensor<Float32> = Tensor(from: try! Tensor<Float64>(numpy: obs))
while true {
  let variable = graph.variable(last_obs)
  let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
  var act_v = act.rawValue
  act_v[0] += noise(exploration_noise)
  let (obs, reward, done, _) = env.step(act_v).tuple4
  buffer.append((obs: last_obs, reward: Float32(reward)!, act: act))
  last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
  print("obs \(obs), reward \(reward)")
  if Bool(done)! {
    let obs = env.reset()
    last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
    break
  }
}

env.close()
