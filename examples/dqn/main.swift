import Algorithms
import Foundation
import NNC
import NNCPythonConversion
import PythonKit

@Sequential
func Net() -> Model {
  Dense(count: 128)
  ReLU()
  Dense(count: 128)
  ReLU()
  Dense(count: 128)
  ReLU()
  Dense(count: 128)
  ReLU()
  Dense(count: 2)
}

let name = "CartPole-v0"

let gym = Python.import("gym")

let env = gym.make(name)

let action_space = env.action_space

let graph = DynamicGraph()

let net = Net()

let eps_test: Float = 0.05
let eps_train: Float = 0.1
let buffer_size = 50_000
let lr: Float32 = 0.001
let gamma: Float32 = 0.9
let n_step = 5
let target_update_freq = 160
let max_epoch = 100
let step_per_epoch = 100
let collect_per_step = 10
let update_per_step = 1
let training_num = 8
let testing_num = 100
let batch_size = 64

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
  var step_in_epoch = 0
  while step_in_epoch < step_per_epoch {
    var episodes = 0
    var env_step_count = 0
    let eps: Float
    if env_step < 10_000 {
      eps = eps_train
    } else if env_step < 50_000 {
      eps = eps_train - Float(env_step - 10_000) / Float(40_000) * 0.9 * eps_train
    } else {
      eps = 0.1 * eps_train
    }
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
            for (i, play) in buffer.enumerated() {
              var rewards = [Float32]()
              for j in 0..<n_step {
                if i + j == buffer.count - 1 {  // For the end, we penalize it.
                  rewards.append(-10)
                } else {
                  rewards.append(i + j < buffer.count ? buffer[i + j].reward : 0)
                }
              }
              let replay = Replay(
                obs: play.obs, obs_next: buffer[min(i + n_step, buffer.count - 1)].obs,
                rewards: rewards, act: play.act, step: i, step_count: buffer.count)
              replays.append(replay)
            }
            buffer.removeAll()
            break
          }
        }
      }
    }
    env_step += env_step_count
    if replays.count > buffer_size {  // Only keep the most recent ones.
      replays.removeFirst(replays.count - buffer_size)
    }
    // Now update the model. First, get some samples out of replay buffer.
    let update_steps = min(
      update_per_step * (env_step_count / collect_per_step), step_per_epoch - step_in_epoch)
    var totalLoss: Float = 0
    for _ in 0..<update_steps {
      // Only update target network at intervals.
      if step_count % target_update_freq == 0 {
        lastNet.parameters.copy(from: net.parameters)
      }
      let batch = replays.randomSample(count: batch_size)
      var obs = Tensor<Float32>(.CPU, .NC(batch_size, 4))
      var obs_next = Tensor<Float32>(.CPU, .NC(batch_size, 4))
      var act = Tensor<Int32>(.CPU, .C(batch_size))
      var r = Tensor<Float32>(.CPU, .NC(batch_size, 1))
      var d = Tensor<Float32>(.CPU, .NC(batch_size, 1))
      for i in 0..<batch_size {
        let replay = batch[i % batch.count]
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
      var q_v = Tensor<Float32>(.CPU, .NC(batch_size, 1))
      for i in 0..<batch_size {
        q_v[i, 0] = act_q[i, 0] > act_q[i, 1] ? target_q[i, 0] : target_q[i, 1]
      }
      let q_q = graph.constant(q_v)
      let r_q = graph.constant(r) .+ graph.constant(d) .* q_q
      let obs_v = graph.variable(obs)
      let act_v = graph.constant(act)
      let pred_q = DynamicGraph.Tensor<Float32>(net(inputs: obs_v)[0])
      let y_q = Functional.indexSelect(input: pred_q.reshaped(.NC(batch_size * 2, 1)), index: act_v)
      // Use Huber loss.
      let loss = DynamicGraph.Tensor<Float32>(SmoothL1Loss()(y_q, target: r_q)[0])
      // Use MSE loss.
      /*
      let td_q = y_q - r_q
      let loss = td_q .* td_q
      */
      let grad: DynamicGraph.Tensor<Float32> = graph.variable(.CPU, .NC(batch_size, 1))
      grad.full(1.0 / Float(batch_size))
      loss.grad = grad
      var total: Float = 0
      for i in 0..<batch_size {
        total += loss[i, 0]
      }
      totalLoss += total
      loss.backward(to: obs_v)
      adamOptimizer.step()
      step_count += 1
      step_in_epoch += 1
    }
    print(
      "Epoch \(epoch), step \(step_in_epoch), loss \(Float(totalLoss) / Float(batch_size * update_steps)), reward \(Float(env_step_count) / Float(episodes))"
    )
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
  print("Epoch \(epoch), testing reward \(avg_step_count)")
  if avg_step_count >= Float(env.spec.reward_threshold)! {
    print("Stop criteria met. Saving mode to \(name).ckpt.")
    graph.openStore("\(name).ckpt") { store in
      store.write("dqn", model: net)
    }
    break
  }
}

var episodes = 0
while episodes < 10 {
  let variable = graph.variable(last_obs)
  let output = DynamicGraph.Tensor<Float32>(net(inputs: variable)[0])
  let act = output[0] > output[1] ? 0 : 1
  let (obs, _, done, _) = env.step(act).tuple4
  last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
  if Bool(done)! {
    let obs = env.reset()
    last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
    episodes += 1
  }
  env.render()
  Thread.sleep(forTimeInterval: 0.0166667)
}

env.close()
