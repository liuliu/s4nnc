import Algorithms
import Foundation
import NNC
import NNCPythonConversion
import Numerics
import PythonKit

func NetA() -> (Model, Model) {
  let lastLayer = Dense(count: 8)
  return (
    Model([
      Dense(count: 64),
      Tanh(),
      Dense(count: 64),
      Tanh(),
      lastLayer,
    ]), lastLayer
  )
}

@Sequential
func NetC() -> Model {
  Dense(count: 64)
  Tanh()
  Dense(count: 64)
  Tanh()
  Dense(count: 1)
  Tanh()
}

let graph = DynamicGraph()

// Note that for PPO, we record the whole episode of replay.
struct Replay {
  var obs: [Tensor<Float32>]  // The state before action.
  var rewards: [Float32]  // Rewards for each step.
  var mu: [Tensor<Float32>]  // The mean for the act in continuous space.
  var act: [Tensor<Float32>]  // The act taken in this step.
  var vs: [Float32]  // The estimated values from obs.
}

struct Data {
  var obs: Tensor<Float32>
  var mu: Tensor<Float32>
  var adv: Float
  var ret: Float
  var distOld: Tensor<Float32>
}

let buffer_size = 4_096
let actor_lr: Float = 3e-4
let critic_lr: Float = 3e-4
let max_epoch = 100
let gamma = 0.99
let step_per_epoch = 30_000
let step_per_collect = 2_048
let exploration_noise: Float = 0.001
let collect_per_step = 2_048
let update_per_step = 1
let batch_size = 64
let rew_norm = true
let vf_coef = 0.25
// let ent_coef = 0.0
let training_num = 64
let testing_num = 10
let gae_lambda = 0.95
let max_grad_norm = 0.5
let eps_clip: Float = 0.2
// let dual_clip = nil
// let value_clip = 0.0
// let norm_adv = 0
let recompute_adv = true

let name = "Ant-v3"

let gym = Python.import("gym")

let env = gym.make(name)

let action_space = env.action_space

let obs = env.reset(seed: 0)
var episodes = 0

let (actor, actorLastLayer) = NetA()
let critic = NetC()

var actorOptim = AdamOptimizer(graph, rate: actor_lr)
actorOptim.parameters = [actor.parameters]
var criticOptim = AdamOptimizer(graph, rate: critic_lr)
criticOptim.parameters = [critic.parameters]

func noise(_ std: Float) -> Float {
  let u1 = Float.random(in: 0...1)
  let u2 = Float.random(in: 0...1)
  let mag = std * (-2.0 * .log(u1)).squareRoot()
  return mag * .cos(.pi * 2 * u2)
}

func compute_episodic_return(replay: Replay, gamma: Float = 0.99, gae_gamma: Float = 0.95) -> (
  advantages: [Float32], returns: [Float32]
) {
  let vs = replay.vs
  let count = vs.count
  let delta: [Float32] = replay.rewards.enumerated().map { (i: Int, rew: Float) -> Float in
    rew + (i + 1 < count ? vs[i + 1] : 0) * gamma - vs[i]
  }
  var gae: Float = 0
  let advantages: [Float32] = delta.enumerated().reversed().map({ (_: Int, delta: Float) -> Float in
    gae = delta + gamma * gae_gamma * gae
    return gae
  }).reversed()
  let unnormalized_returns = advantages.enumerated().map { i, adv in
    vs[i] + adv
  }
  return (advantages, unnormalized_returns)
}

let actionLow: [Float] = env.action_space.low.map { Float($0)! }
let actionHigh: [Float] = env.action_space.high.map { Float($0)! }

var replays = [Replay]()
var buffer = [
  (obs: Tensor<Float32>, reward: Float32, mu: Tensor<Float32>, act: Tensor<Float32>, v: Float32)
]()
var last_obs: Tensor<Float32> = Tensor(from: try! Tensor<Float64>(numpy: obs))
var env_step = 0
var step_count = 0
var rew_var: Double = 0
var rew_mean: Double = 0
var rew_total = 0
var initActorLastLayer = false
for epoch in 0..<max_epoch {
  var step_in_epoch = 0
  while step_in_epoch < step_per_epoch {
    var episodes = 0
    var env_step_count = 0
    var training_rewards: Float = 0
    while env_step_count < collect_per_step {
      for _ in 0..<training_num {
        while true {
          let variable = graph.variable(last_obs.toGPU(0))
          let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0]).toCPU()
          if !initActorLastLayer {
            // Try to init actor's last layer with reduced weights.
            let bias = graph.variable(.CPU, .C(8), of: Float32.self)
            bias.full(0)
            actorLastLayer.parameters(for: .bias).copy(from: bias)
            let weight = graph.variable(.CPU, .NC(64, 8), of: Float32.self)
            actorLastLayer.parameters(for: .weight).copy(to: weight)
            let updatedWeight = 0.01 * weight
            actorLastLayer.parameters(for: .weight).copy(from: updatedWeight)
            initActorLastLayer = true
          }
          let v = DynamicGraph.Tensor<Float32>(critic(inputs: variable)[0]).toCPU()
          let f = (0..<8).map {
            max(min(act[$0] + noise(exploration_noise), actionHigh[$0]), actionLow[$0])
          }
          let act_f = Tensor<Float32>(f, .CPU, .C(8))
          let act_mu = act.rawValue.copied()
          let (obs, reward, done, _) = env.step(act_f).tuple4
          buffer.append((obs: last_obs, reward: Float32(reward)!, mu: act_mu, act: act_f, v: v[0]))
          last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
          if Bool(done)! {
            let obs = env.reset()
            episodes += 1
            env_step_count += buffer.count
            last_obs = Tensor(from: try! Tensor<Float64>(numpy: obs))
            var obss = [Tensor<Float32>]()
            var rewards = [Float32]()
            var mus = [Tensor<Float32>]()
            var acts = [Tensor<Float32>]()
            var vs = [Float32]()
            // Organizing data into ReplayBuffer.
            for play in buffer {
              training_rewards += play.reward
              rewards.append(play.reward)
              obss.append(play.obs)
              mus.append(play.mu)
              acts.append(play.act)
              vs.append(play.v)
            }
            let replay = Replay(
              obs: obss,
              rewards: rewards,
              mu: mus,
              act: acts,
              vs: vs)
            replays.append(replay)
            buffer.removeAll()
            break
          }
        }
      }
    }
    env_step += env_step_count
    // Now update the model. First, get some samples out of replay buffer.
    let update_steps = min(
      1,
      min(
        update_per_step * (env_step_count / collect_per_step), step_per_epoch - step_in_epoch))
    var criticLoss: Float = 0
    var actorLoss: Float = 0
    var update_count = 0
    for _ in 0..<update_steps {
      let replayBatch = replays.randomSample(count: batch_size)
      var data = [Data]()
      // Sample from these batches into smaller batch sizes and do the update.
      for replay in replayBatch {
        let (advantages, unnormalized_returns) = compute_episodic_return(replay: replay)
        let obs = replay.obs
        let mu = replay.mu
        let act = replay.act
        var inv_std: Float = 1
        if rew_total > 0 {
          let rew_std =
            (rew_var / Double(rew_total)
            - (rew_mean / Double(rew_total)) * (rew_mean / Double(rew_total))).squareRoot()
          inv_std = 1.0 / (Float(rew_std) + 1e-4)
        }
        for (i, adv) in advantages.enumerated() {
          let muv = graph.constant(mu[i])
          let actv = graph.constant(act[i])
          let distOld = ((muv - actv) .* (muv - actv)).rawValue.copied()
          data.append(
            Data(
              obs: obs[i], mu: mu[i], adv: adv, ret: inv_std * unnormalized_returns[i],
              distOld: distOld))
        }
        for rew in unnormalized_returns {
          rew_var += Double(rew * rew)
          rew_mean += Double(rew)
        }
        rew_total += unnormalized_returns.count
      }
      for _ in 0..<(data.count / batch_size) {
        let batch = data.randomSample(count: batch_size)
        var obs = Tensor<Float32>(.CPU, .NC(batch_size, 111))
        var mu = Tensor<Float32>(.CPU, .NC(batch_size, 8))
        var advantages = Tensor<Float32>(.CPU, .NC(batch_size, 1))
        var returns = Tensor<Float32>(.CPU, .NC(batch_size, 1))
        var distOld = Tensor<Float32>(.CPU, .NC(batch_size, 8))
        for i in 0..<batch_size {
          let data = batch[i % batch.count]
          obs[i, ...] = data.obs[...]
          mu[i, ...] = data.mu[...]
          advantages[i, 0] = data.adv
          returns[i, 0] = data.ret
          distOld[i, ...] = data.distOld[...]
        }
        let variable = graph.variable(obs.toGPU(0))
        let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
        let v = DynamicGraph.Tensor<Float32>(critic(inputs: variable)[0])
        let muv = graph.constant(mu.toGPU(0))
        let distOldv = graph.constant(distOld.toGPU(0))
        let dist = ((muv - act) .* (muv - act))
        let ratio = Functional.exp(dist - distOldv)
        let advantagesv = graph.constant(advantages.toGPU(0))
        let surr1 = advantagesv .* ratio
        let surr2 = advantagesv .* ratio.clamped((1.0 - eps_clip)...(1.0 + eps_clip))
        let clip_loss = Functional.min(surr1, surr2)
        let cpu_clip_loss = clip_loss.toCPU()
        var totalLoss: Float = 0
        for i in 0..<batch_size {
          for j in 0..<8 {
            totalLoss += cpu_clip_loss[i, j]
          }
        }
        actorLoss += totalLoss
        let grad: DynamicGraph.Tensor<Float32> = graph.variable(.GPU(0), .NC(batch_size, 8))
        grad.full(-1.0 / Float(batch_size * 8))
        clip_loss.grad = grad
        clip_loss.backward(to: variable)
        actor.parameters.clipGradNorm(maxNorm: 0.5)
        actorOptim.step()
        let returnsv = graph.constant(returns.toGPU(0))
        let vf_loss = DynamicGraph.Tensor<Float32>(SmoothL1Loss()(v, target: returnsv)[0])
        let cpu_vf_loss = vf_loss.toCPU()
        for i in 0..<batch_size {
          criticLoss += cpu_vf_loss[i, 0]
        }
        let vf_grad: DynamicGraph.Tensor<Float32> = graph.variable(.GPU(0), .NC(batch_size, 1))
        vf_grad.full(1.0 / Float(batch_size))
        vf_loss.grad = vf_grad
        vf_loss.backward(to: variable)
        critic.parameters.clipGradNorm(maxNorm: 0.5)
        criticOptim.step()
        update_count += 1
      }
    }
    step_in_epoch += update_count
    criticLoss = criticLoss / Float(batch_size * update_count)
    actorLoss = -actorLoss / Float(batch_size * 8 * update_count)
    replays.removeAll()
    print(
      "Epoch \(epoch), step \(step_in_epoch), critic loss \(criticLoss), actor loss \(actorLoss), reward \(Float(training_rewards) / Float(episodes))"
    )
  }
}

/*
while episodes < 10 {
  let act_v = action_space.sample()
  let (obs, _, done, _) = env.step(act_v).tuple4
  if Bool(done)! {
    let obs = env.reset()
    episodes += 1
  }
  env.render()
  Thread.sleep(forTimeInterval: 0.0166667)
}
*/
env.close()
