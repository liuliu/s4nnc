import Algorithms
import Foundation
import NNC
import NNCPythonConversion
import Numerics
import PythonKit

@Sequential
func Net() -> Model {
  Dense(count: 128)
  ReLU()
  Dense(count: 128)
  ReLU()
  Dense(count: 1)
}

let name = "Pendulum-v1"

let gym = Python.import("gym")

let env = gym.make(name)

env.reset(seed: 0)
env.spec.reward_threshold = -250

let action_space = env.action_space

let graph = DynamicGraph()

struct Replay {
  var obs: Tensor<Float32>  // The state before action.
  var obs_next: Tensor<Float32>  // The state n_step ahead.
  var rewards: [Float32]  // Rewards for 0..<n_step - 1
  var act: Tensor<Float32>  // The act taken in the episode.
  var act_next: Tensor<Float32>  // The act taken n_step ahead.
  var step: Int  // The step in the episode.
  var step_count: Int  // How many steps til the end, step < step_count.
}

let buffer_size = 20_000
let actor_lr: Float = 1e-4
let critic_lr: Float = 1e-3
let gamma: Float = 0.99
let tau: Float = 0.005
let exploration_noise: Float = 0.1
let max_epoch = 20
let step_per_epoch = 2400
let collect_per_step = 4
let update_per_step = 1
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
let actorOld = actor.copy()
let criticOld = critic.copy()

var actorOptim = AdamOptimizer(graph, rate: actor_lr)
actorOptim.parameters = [actor.parameters]
var criticOptim = AdamOptimizer(graph, rate: critic_lr)
criticOptim.parameters = [critic.parameters]

let actionLow: Float = Float(env.action_space.low[0])!
let actionHigh: Float = Float(env.action_space.high[0])!

var replays = [Replay]()
let obs = env.reset()
var buffer = [(obs: Tensor<Float32>, reward: Float32, act: Tensor<Float32>)]()
var last_obs: Tensor<Float32> = try! Tensor<Float32>(numpy: obs)
var env_step = 0
var step_count = 0
for epoch in 0..<max_epoch {
  var step_in_epoch = 0
  while step_in_epoch < step_per_epoch {
    var episodes = 0
    var env_step_count = 0
    var training_rewards: Float = 0
    while env_step_count < collect_per_step {
      for _ in 0..<training_num {
        while true {
          let variable = graph.variable(last_obs)
          let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
          let v = max(min(act[0] + noise(exploration_noise), actionHigh), actionLow)
          let act_v = Tensor<Float32>([v], .CPU, .C(1))
          let (obs, reward, done, _) = env.step(act_v).tuple4
          buffer.append((obs: last_obs, reward: Float32(reward)!, act: act_v))
          last_obs = try! Tensor<Float32>(numpy: obs)
          if Bool(done)! {
            let obs = env.reset()
            episodes += 1
            env_step_count += buffer.count
            last_obs = try! Tensor<Float32>(numpy: obs)
            // Organizing data into ReplayBuffer.
            for (i, play) in buffer.enumerated() {
              training_rewards += play.reward
              var rewards = [Float32]()
              for j in 0..<n_step {
                rewards.append(i + j < buffer.count ? buffer[i + j].reward : 0)
              }
              let replay = Replay(
                obs: play.obs,
                obs_next: buffer[min(i + n_step, buffer.count - 1)].obs,
                rewards: rewards,
                act: play.act,
                act_next: buffer[min(i + n_step, buffer.count - 1)].act,
                step: i, step_count: buffer.count)
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
    var criticLoss: Float = 0
    var actorLoss: Float = 0
    for update_step in 0..<update_steps {
      let batch = replays.randomSample(count: batch_size)
      var obs = Tensor<Float32>(.CPU, .NC(batch_size, 3))
      var obs_next = Tensor<Float32>(.CPU, .NC(batch_size, 3))
      var act = Tensor<Float32>(.CPU, .NC(batch_size, 1))
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
        act[i, ...] = replay.act[...]
        r[i, 0] = rew
        d[i, 0] = replay.step + n_step < replay.step_count ? discount : 0
      }
      // Compute the q.
      let obs_next_v = graph.constant(obs_next)
      if update_step == 0 && epoch == 0 && step_in_epoch == 0 {
        // Firs time use actorOld, copy its parameters from actor.
        actorOld.parameters.copy(from: actor.parameters)
      }
      let act_next_v = DynamicGraph.Tensor<Float32>(actorOld(inputs: obs_next_v)[0])
      act_next_v.clamp(actionLow...actionHigh)
      let obs_act_next_v: DynamicGraph.Tensor<Float32> = graph.constant(.CPU, .NC(batch_size, 4))
      obs_act_next_v[0..<batch_size, 0..<3] = obs_next_v
      obs_act_next_v[0..<batch_size, 3..<4] = act_next_v
      let target_q = DynamicGraph.Tensor<Float32>(criticOld(inputs: obs_act_next_v)[0])
      let r_q = graph.constant(r) .+ graph.constant(d) .* target_q
      let obs_v = graph.variable(obs)
      let act_v = graph.constant(act)
      let obs_act_v: DynamicGraph.Tensor<Float32> = graph.variable(.CPU, .NC(batch_size, 4))
      obs_act_v[0..<batch_size, 0..<3] = obs_v
      obs_act_v[0..<batch_size, 3..<4] = act_v
      if update_step == 0 && epoch == 0 && step_in_epoch == 0 {
        // First time use critic, copy its parameters from criticOld.
        critic.parameters.copy(from: criticOld.parameters)
      }
      let pred_q = DynamicGraph.Tensor<Float32>(critic(inputs: obs_act_v)[0])
      let loss = DynamicGraph.Tensor<Float32>(SmoothL1Loss()(pred_q, target: r_q)[0])
      for i in 0..<batch_size {
        criticLoss += loss[i, 0]
      }
      loss.backward(to: obs_act_v)
      criticOptim.step()
      let new_act_v = DynamicGraph.Tensor<Float32>(actor(inputs: obs_v)[0])
      new_act_v.clamp(actionLow...actionHigh)
      let new_obs_act_v: DynamicGraph.Tensor<Float32> = graph.variable(.CPU, .NC(batch_size, 4))
      new_obs_act_v[0..<batch_size, 0..<3] = obs_v
      new_obs_act_v[0..<batch_size, 3..<4] = new_act_v
      let actor_loss = DynamicGraph.Tensor<Float32>(critic(inputs: new_obs_act_v)[0])
      for i in 0..<batch_size {
        actorLoss += actor_loss[i, 0]
      }
      let grad: DynamicGraph.Tensor<Float32> = graph.variable(.CPU, .NC(batch_size, 1))
      // Run gradient ascent, therefore, the negative sign for the gradient. It is the same as:
      // actor_loss = -critic(inputs: new_obs_act_v)[0]
      grad.full(-1.0 / Float(batch_size))
      actor_loss.grad = grad
      actor_loss.backward(to: obs_v)
      actorOptim.step()
      actorOld.parameters.lerp(tau, to: actor.parameters)
      criticOld.parameters.lerp(tau, to: critic.parameters)
      step_count += 1
      step_in_epoch += 1
    }
    criticLoss = criticLoss / Float(batch_size * update_steps)
    actorLoss = -actorLoss / Float(batch_size * update_steps)
    print(
      "Epoch \(epoch), step \(step_in_epoch), critic loss \(criticLoss), actor loss \(actorLoss), reward \(Float(training_rewards) / Float(episodes))"
    )
  }
  // Running test and print how many steps we can perform in an episode before it fails.
  var testing_rewards: Float = 0
  for _ in 0..<testing_num {
    while true {
      let variable = graph.variable(last_obs)
      let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
      act.clamp(actionLow...actionHigh)
      let act_v = act.rawValue
      let (obs, reward, done, _) = env.step(act_v).tuple4
      last_obs = try! Tensor<Float32>(numpy: obs)
      testing_rewards += Float(reward)!
      if Bool(done)! {
        let obs = env.reset()
        last_obs = try! Tensor<Float32>(numpy: obs)
        break
      }
    }
  }
  let avg_testing_rewards = testing_rewards / Float(testing_num)
  print("Epoch \(epoch), testing reward \(avg_testing_rewards)")
  if avg_testing_rewards >= Float(env.spec.reward_threshold)! {
    print("Stop criteria met. Saving mode to \(name).ckpt.")
    graph.openStore("\(name).ckpt") { store in
      store.write("ddpg_actor", model: actor)
      store.write("ddpg_critic", model: critic)
    }
    break
  }
}

var episodes = 0
while episodes < 10 {
  let variable = graph.variable(last_obs)
  let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
  act.clamp(actionLow...actionHigh)
  let act_v = act.rawValue
  let (obs, _, done, _) = env.step(act_v).tuple4
  last_obs = try! Tensor<Float32>(numpy: obs)
  if Bool(done)! {
    let obs = env.reset()
    last_obs = try! Tensor<Float32>(numpy: obs)
    episodes += 1
  }
  env.render()
  Thread.sleep(forTimeInterval: 0.0166667)
}

env.close()
