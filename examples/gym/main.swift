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

let name = "CartPole-v0"

let gym = Python.import("gym")

let env = gym.make(name)

env.seed(0)

let action_space = env.action_space

let graph = DynamicGraph()

let net = Net()

let eps_test: Float = 0.05
let eps_train: Float = 0.1
let buffer_size = 50_000
let lr: Float32 = 1e-3
let gamma: Float32 = 0.95
let n_step = 5
let target_update_freq = 160
let max_epoch = 10000
let step_per_epoch = 1
let collect_per_step = 10
let update_per_step = 1
let training_num = 8
let testing_num = 100
let batch_size = 64
let alpha = 0.6
let beta = 0.4

struct Replay {
  var obs: Tensor<Float32>  // The state before action.
  var obs_next: Tensor<Float32>  // The state n_step ahead.
  var rewards: [Float32]  // Rewards for 0..<n_step - 1
  var act: Int  // The act taken in the episode.
  var step: Int  // The step in the episode.
  var step_count: Int  // How many steps til the end, step < step_count.
}

let lastNet = net.copy()
var replays = [Replay]()

let obs = env.reset()
var adamOptimizer = AdamOptimizer(graph, rate: lr)
adamOptimizer.parameters = [net.parameters]
var buffer = [(obs: Tensor<Float32>, reward: Float32, act: Int)]()
var last_obs: Tensor<Float32> = Tensor(from: try! Tensor<Float64>(numpy: obs))
var env_step = 0
var step_count = 0
for epoch in 0..<max_epoch {
  for _ in 0..<step_per_epoch {
    var episodes = 0
    var env_step_count = 0
    let eps: Float = eps_train
    while env_step_count < collect_per_step {
      for _ in 0..<training_num {
        while true {
          let variable = graph.variable(last_obs)
          let output = DynamicGraph.Tensor<Float32>(net(inputs: variable)[0])
          let act: Int
          if Float.random(in: 0..<1) < eps {
            act = Int.random(in: 0...1)
          } else {
            act = output[0] > output[1] ? 0 : 1
          }
          let (obs, reward, done, _) = env.step(act).tuple4
          buffer.append((obs: last_obs, reward: Float32(reward)!, act: act))
          last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
          if Bool(done)! {
            let obs = env.reset()
            episodes += 1
            env_step_count += buffer.count
            last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
            // Organizing data into ReplayBuffer.
            for (i, _) in buffer.enumerated() {
              var rewards = [Float32]()
              for j in 0..<n_step {
                rewards.append(i + j < buffer.count ? buffer[i + j].reward : 0)
              }
              let replay = Replay(
                obs: buffer[i].obs, obs_next: buffer[min(i + n_step, buffer.count - 1)].obs,
                rewards: rewards, act: act, step: i, step_count: buffer.count)
              replays.append(replay)
            }
            buffer.removeAll()
            break
          }
        }
      }
    }
    env_step += env_step_count
    // Only update target network at intervals.
    if step_count % target_update_freq == 0 {
      lastNet.parameters.copy(from: net.parameters)
    }
    if replays.count > buffer_size {  // Only keep the most recent ones.
      replays.removeFirst(replays.count - buffer_size)
    }
    // Now update the model. First, get some samples out of replay buffer.
    for _ in 0..<update_per_step {
      var batch = [Replay]()
      for _ in 0..<batch_size {
        let i = Int.random(in: 0..<replays.count)
        batch.append(replays[i])
      }
      var obs = Tensor<Float32>(.CPU, .NC(batch_size, 4))
      var obs_next = Tensor<Float32>(.CPU, .NC(batch_size, 4))
      var act = Tensor<Int32>(.CPU, .C(batch_size))
      var r = Tensor<Float32>(.CPU, .NC(batch_size, 1))
      var d = Tensor<Float32>(.CPU, .NC(batch_size, 1))
      for (i, replay) in batch.enumerated() {
        obs[i, ...] = replay.obs[...]
        obs_next[i, ...] = replay.obs_next[...]
        var rew: Float32 = 0
        var discount: Float32 = 1
        for j in 0..<n_step {
          if replay.step + j < replay.step_count {
            rew += discount * replay.rewards[j]  // reward is always 1 in CartPole
          }
          discount *= gamma
        }
        act[i] = replay.act == 0 ? Int32(i) * 2 : Int32(i) * 2 + 1
        r[i, 0] = rew
        d[i, 0] = replay.step + n_step < replay.step_count ? discount : 0
      }
      // Compute the q.
      let obs_next_v = graph.variable(obs_next)
      let act_q = DynamicGraph.Tensor<Float32>(net(inputs: obs_next_v)[0])
      let target_q = DynamicGraph.Tensor<Float32>(lastNet(inputs: obs_next_v)[0])
      var act_next = Tensor<Int32>(.CPU, .C(batch_size))
      for i in 0..<batch_size {
        act_next[i] = act_q[i, 0] > act_q[i, 1] ? Int32(i) * 2 : Int32(i) * 2 + 1
      }
      let act_next_v = graph.constant(act_next)
      let q_q = Functional.indexSelect(
        input: target_q.reshape(.NC(batch_size * 2, 1)), index: act_next_v)
      let r_q = graph.constant(r) .+ graph.constant(d) .* q_q
      let obs_v = graph.variable(obs)
      let act_v = graph.constant(act)
      let pred_q = DynamicGraph.Tensor<Float32>(net(inputs: obs_v)[0])
      let y_q = Functional.indexSelect(input: pred_q.reshape(.NC(batch_size * 2, 1)), index: act_v)
      let td_q = y_q - r_q
      let mse = td_q .* td_q
      mse.backward(to: obs_v)
      adamOptimizer.step()
    }
    step_count += 1
  }
  // Running test and print how many steps we can perform in an episode before it fails.
  var env_step_count = 0
  for _ in 0..<testing_num {
    while true {
      let variable = graph.variable(last_obs)
      let output = DynamicGraph.Tensor<Float32>(net(inputs: variable)[0])
      let act = output[0] > output[1] ? 0 : 1
      let (obs, _, done, _) = env.step(act).tuple4
      last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
      env_step_count += 1
      if Bool(done)! {
        let obs = env.reset()
        last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
        break
      }
    }
  }
  let avg_step_count = Float(env_step_count) / Float(testing_num)
  print("epoch \(epoch), testing env step \(avg_step_count)")
  if avg_step_count >= Float(env.spec.reward_threshold)! {
    print("Stop criteria met. Saving mode to \(name).ckpt.")
    graph.openStore("\(name).ckpt") { store in
      store.write("dqn", model: net)
    }
    break
  }
}

env.close()
