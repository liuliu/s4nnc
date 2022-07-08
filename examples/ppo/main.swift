import Algorithms
import Foundation
import Gym
import NNC
import NNCPythonConversion
import Numerics

let input_dim = 27
let output_dim = 8

func NetA() -> (Model, Model) {
  let lastLayer = Dense(count: output_dim)
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
}

let graph = DynamicGraph()
var sfmt = SFMT(seed: 10)

// Note that for PPO, we record the whole episode of replay.
struct Replay {
  var obs: [Tensor<Float32>]  // The state before action.
  var lastObs: Tensor<Float32>  // The state before action.
  var rewards: [Float32]  // Rewards for each step.
  var mu: [Tensor<Float32>]  // The mean for the act in continuous space.
  var act: [Tensor<Float32>]  // The act taken in this step.
  var vs: [Float32]  // The estimated values from obs.
}

struct Data {
  var obs: Tensor<Float32>
  var act: Tensor<Float32>
  var adv: Float
  var ret: Float
  var distOld: Tensor<Float32>
}

let actor_lr: Float = 3e-4
let critic_lr: Float = 3e-4
let max_epoch = 100
let step_per_epoch = 30_000
let collect_per_step = 20_000
let update_per_step = 10
let batch_size = 64
let vf_coef: Float = 0.25
let ent_coef: Float = 0.001
let training_num = 64
let testing_num = 10
let max_grad_norm: Float = 0.5
let eps_clip: Float = 0.2

var envs = [TimeLimit<Ant>]()
var obs = [Tensor<Float64>]()
for i in 0..<training_num {
  let env = TimeLimit(env: try Ant(), maxEpisodeSteps: 1_000)
  let result = env.reset(seed: i)
  envs.append(env)
  obs.append(result.0)
}
let viewer = MuJoCoViewer(env: envs[0])
var episodes = 0

let (actor, actorLastLayer) = NetA()
let critic = NetC()

var actorOptim = AdamOptimizer(graph, rate: actor_lr)
let scale = graph.variable(.GPU(0), .C(output_dim), of: Float32.self)
scale.full(0)
actorOptim.parameters = [actor.parameters, scale]
var criticOptim = AdamOptimizer(graph, rate: critic_lr)
criticOptim.parameters = [critic.parameters]

func compute_episodic_return(
  replay: Replay, rew_std: Float, gamma: Float = 0.99, gae_gamma: Float = 0.95
) -> (
  advantages: [Float32], returns: [Float32]
) {
  let vs = replay.vs
  let delta: [Float32] = replay.rewards.enumerated().map { (i: Int, rew: Float) -> Float in
    rew + (vs[i + 1] * gamma - vs[i]) * rew_std
  }
  var gae: Float = 0
  let advantages: [Float32] = delta.reversed().map({ (delta: Float) -> Float in
    gae = delta + gamma * gae_gamma * gae
    return gae
  }).reversed()
  let unnormalized_returns = advantages.enumerated().map { i, adv in
    vs[i] * rew_std + adv
  }
  return (advantages, unnormalized_returns)
}

var obsRms = RunningMeanStd(
  mean: ({ () -> DynamicGraph.Tensor<Float> in
    let mean = graph.constant(.GPU(0), .C(input_dim), of: Float32.self)
    mean.full(0)
    return mean
  })(),
  variance: ({ () -> DynamicGraph.Tensor<Float> in
    let variance = graph.constant(.GPU(0), .C(input_dim), of: Float32.self)
    variance.full(1)
    return variance
  })())
var replays = [Replay]()
var buffer = Array(
  repeating: [
    (obs: Tensor<Float32>, reward: Float32, mu: Tensor<Float32>, act: Tensor<Float32>)
  ](), count: training_num)
var last_obs: [Tensor<Float32>] = obs.map { Tensor(from: $0) }
var env_step = 0
var rew_var: Double = 1
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
      for i in 0..<training_num {
        var lastObs = graph.variable(last_obs[i].toGPU(0))
        let variable =
          (lastObs - obsRms.mean) ./ Functional.squareRoot(obsRms.variance).clamped(1e-5...)
        obsRms.update([lastObs])
        let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
        if !initActorLastLayer {
          // Try to init actor's last layer with reduced weights.
          let bias = graph.variable(.CPU, .C(output_dim), of: Float32.self)
          bias.full(0)
          actorLastLayer.parameters(for: .bias).copy(from: bias)
          let weight = graph.variable(
            actorLastLayer.parameters(for: .weight).copied(Float32.self))
          let updatedWeight = 0.01 * weight
          actorLastLayer.parameters(for: .weight).copy(from: updatedWeight)
          initActorLastLayer = true
        }
        let n = graph.variable(Tensor<Float32>(.GPU(0), .C(output_dim)))
        n.randn(std: 1, mean: 0)
        let act_f = (n .* Functional.exp(scale) + act).clamped(-1...1).toCPU().rawValue.copied()
        let act_mu = act.toCPU().rawValue.copied()
        let (obs, reward, done, _) = envs[i].step(action: Tensor(from: act_f))
        buffer[i].append(
          (obs: variable.rawValue.toCPU(), reward: reward, mu: act_mu, act: act_f)
        )
        last_obs[i] = Tensor(from: obs)
        if done {
          lastObs = graph.variable(last_obs[i].toGPU(0))
          let variable =
            (lastObs - obsRms.mean) ./ Functional.squareRoot(obsRms.variance).clamped(1e-5...)
          obsRms.update([lastObs])
          let (obs, _) = envs[i].reset()
          episodes += 1
          var obss = [Tensor<Float32>]()
          var rewards = [Float32]()
          var mus = [Tensor<Float32>]()
          var acts = [Tensor<Float32>]()
          // Organizing data into ReplayBuffer.
          for play in buffer[i] {
            training_rewards += play.reward
            rewards.append(play.reward)
            obss.append(play.obs)
            mus.append(play.mu)
            acts.append(play.act)
          }
          let vs: [Float32] = Array(repeating: 0, count: buffer[i].count + 1)
          let replay = Replay(
            obs: obss,
            lastObs: variable.rawValue.toCPU(),
            rewards: rewards,
            mu: mus,
            act: acts,
            vs: vs)
          replays.append(replay)
          last_obs[i] = Tensor(from: obs)
          buffer[i].removeAll()
        }
      }
      env_step_count += training_num
    }
    // If there are any partial replays left in the buffer, put these into the replay array and remove from buffer.
    for i in 0..<training_num where buffer[i].count > 0 {
      let lastObs = graph.variable(last_obs[i].toGPU(0))
      let variable =
        (lastObs - obsRms.mean) ./ Functional.squareRoot(obsRms.variance).clamped(1e-5...)
      obsRms.update([lastObs])
      var obss = [Tensor<Float32>]()
      var rewards = [Float32]()
      var mus = [Tensor<Float32>]()
      var acts = [Tensor<Float32>]()
      // Organizing data into ReplayBuffer.
      for play in buffer[i] {
        training_rewards += play.reward
        rewards.append(play.reward)
        obss.append(play.obs)
        mus.append(play.mu)
        acts.append(play.act)
      }
      let vs: [Float32] = Array(repeating: 0, count: buffer[i].count + 1)
      let replay = Replay(
        obs: obss,
        lastObs: variable.rawValue.toCPU(),
        rewards: rewards,
        mu: mus,
        act: acts,
        vs: vs)
      replays.append(replay)
      buffer[i].removeAll()
    }
    env_step += env_step_count
    let lr =
      1.0 - (min(Float(step_in_epoch) / Float(step_per_epoch), 1) + Float(epoch)) / Float(max_epoch)
    actorOptim.rate = actor_lr * lr
    criticOptim.rate = critic_lr * lr
    step_in_epoch += env_step_count
    // Now update the model. First, get some samples out of replay buffer.
    var criticLoss: Float = 0
    var actorLoss: Float = 0
    var update_count = 0
    let rawScale = scale.toCPU().rawValue.copied()
    for _ in 0..<update_per_step {
      let replayBatch = replays.randomSample(count: batch_size, using: &sfmt)  // System random generator is very slow, use custom one.
      var data = [Data]()
      // Sample from these batches into smaller batch sizes and do the update.
      for var replay in replayBatch {
        var rew_std: Float = 1
        var inv_std: Float = 1
        if rew_total > 0 {
          rew_std = Float(rew_var.squareRoot()) + 1e-5
          inv_std = 1.0 / rew_std
        }
        // Recompute value with critics.
        for (j, obs) in replay.obs.enumerated() {
          let variable = graph.variable(obs.toGPU(0))
          let v = DynamicGraph.Tensor<Float32>(critic(inputs: variable)[0]).toCPU()
          replay.vs[j] = v[0]
        }
        let variable = graph.variable(replay.lastObs.toGPU(0))
        let v = DynamicGraph.Tensor<Float32>(critic(inputs: variable)[0]).toCPU()
        replay.vs[replay.obs.count] = v[0]
        let (advantages, unnormalized_returns) = compute_episodic_return(
          replay: replay, rew_std: rew_std)
        let obs = replay.obs
        let mu = replay.mu
        let act = replay.act
        let scale = graph.constant(rawScale)
        let expScale = Functional.exp(scale)
        let var2 = 1 / (2 * (expScale .* expScale))
        for (i, adv) in advantages.enumerated() {
          let muv = graph.constant(mu[i])
          let actv = graph.constant(act[i])
          let distOld = ((muv - actv) .* (muv - actv) .* var2 + scale).rawValue.copied()
          data.append(
            Data(
              obs: obs[i], act: act[i], adv: adv, ret: inv_std * unnormalized_returns[i],
              distOld: distOld))
        }
        var batch_var: Double = 0
        var batch_mean: Double = 0
        for rew in unnormalized_returns {
          batch_mean += Double(rew)
        }
        batch_mean = batch_mean / Double(unnormalized_returns.count)
        for rew in unnormalized_returns {
          batch_var += (Double(rew) - batch_mean) * (Double(rew) - batch_mean)
        }
        batch_var = batch_var / Double(unnormalized_returns.count)
        let delta = batch_mean - rew_mean
        let total_count = unnormalized_returns.count + rew_total
        rew_mean = rew_mean + delta * Double(unnormalized_returns.count) / Double(total_count)
        let m_a = rew_var * Double(rew_total)
        let m_b = batch_var * Double(unnormalized_returns.count)
        let m_2 =
          m_a + m_b + delta * delta * Double(rew_total) * Double(unnormalized_returns.count)
          / Double(total_count)
        rew_var = m_2 / Double(total_count)
        rew_total = total_count
      }
      for _ in 0..<(data.count / batch_size) {
        let batch = data.randomSample(count: batch_size, using: &sfmt)  // System random generator is very slow, use custom one.
        var obs = Tensor<Float32>(.CPU, .NC(batch_size, input_dim))
        var act = Tensor<Float32>(.CPU, .NC(batch_size, output_dim))
        var advantages = Tensor<Float32>(.CPU, .NC(batch_size, 1))
        var returns = Tensor<Float32>(.CPU, .NC(batch_size, 1))
        var distOld = Tensor<Float32>(.CPU, .NC(batch_size, output_dim))
        for i in 0..<batch_size {
          let data = batch[i % batch.count]
          obs[i, ...] = data.obs[...]
          act[i, ...] = data.act[...]
          advantages[i, 0] = data.adv
          returns[i, 0] = data.ret
          distOld[i, ...] = data.distOld[...]
        }
        let variable = graph.variable(obs.toGPU(0))
        let mu = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
        let actv = graph.constant(act.toGPU(0))
        let distOldv = graph.constant(distOld.toGPU(0))
        let expScale = Functional.exp(scale)
        let var2 = 1 / (2 * (expScale .* expScale))
        let dist = ((mu - actv) .* (mu - actv) .* var2 + scale)
        let ratio = Functional.exp(distOldv - dist)
        let advantagesv = graph.constant(advantages.toGPU(0))
        let surr1 = advantagesv .* ratio
        let surr2 = advantagesv .* ratio.clamped((1.0 - eps_clip)...(1.0 + eps_clip))
        let clip_loss =
          ent_coef * scale.reduced(.sum, axis: [0])
          + Functional.min(surr1, surr2).reduced(.sum, axis: [1])
        let cpu_clip_loss = clip_loss.toCPU()
        var totalLoss: Float = 0
        for i in 0..<batch_size {
          totalLoss += cpu_clip_loss[i, 0]
        }
        actorLoss += totalLoss
        let grad: DynamicGraph.Tensor<Float32> = graph.variable(.GPU(0), .NC(batch_size, 1))
        grad.full(-1.0 / Float(batch_size))
        clip_loss.grad = grad
        clip_loss.backward(to: [variable, scale])
        actor.parameters.clipGradNorm(maxNorm: 0.5)
        actorOptim.step()
        let v = DynamicGraph.Tensor<Float32>(critic(inputs: variable)[0])
        let returnsv = graph.constant(returns.toGPU(0))
        let vf_loss = MSELoss()(v, target: returnsv)[0].as(of: Float32.self)
        let cpu_vf_loss = vf_loss.toCPU()
        for i in 0..<batch_size {
          criticLoss += cpu_vf_loss[i, 0]
        }
        let vf_grad: DynamicGraph.Tensor<Float32> = graph.variable(.GPU(0), .NC(batch_size, 1))
        vf_grad.full(vf_coef / Float(batch_size))
        vf_loss.grad = vf_grad
        vf_loss.backward(to: variable)
        critic.parameters.clipGradNorm(maxNorm: max_grad_norm)
        criticOptim.step()
        update_count += 1
      }
    }
    criticLoss = criticLoss / Float(batch_size * update_count)
    actorLoss = -actorLoss / Float(batch_size * update_count)
    let scaleCPU = scale.toCPU()
    print(
      "rew std \(rew_var.squareRoot()), log scale [\(scaleCPU[0]), \(scaleCPU[1]), \(scaleCPU[2]), \(scaleCPU[3]), \(scaleCPU[4]), \(scaleCPU[5])]"
    )
    replays.removeAll()
    print(
      "Epoch \(epoch), step \(env_step), critic loss \(criticLoss), actor loss \(actorLoss), reward \(Float(training_rewards) / Float(episodes))"
    )
  }
  // Running test and print how many steps we can perform in an episode before it fails.
  let result = envs[0].reset()
  last_obs[0] = Tensor(from: result.0)
  var testing_rewards: Float = 0
  for _ in 0..<testing_num {
    while true {
      let lastObs = graph.variable(last_obs[0].toGPU(0))
      let variable =
        (lastObs - obsRms.mean) ./ Functional.squareRoot(obsRms.variance).clamped(1e-5...)
      let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
      act.clamp(-1...1)
      let act_v = act.rawValue.toCPU()
      let (obs, reward, done, _) = envs[0].step(action: Tensor(from: act_v))
      last_obs[0] = Tensor(from: obs)
      testing_rewards += reward
      if done {
        let (obs, _) = envs[0].reset()
        last_obs[0] = Tensor(from: obs)
        break
      }
    }
  }
  let avg_testing_rewards = testing_rewards / Float(testing_num)
  print("Epoch \(epoch), testing reward \(avg_testing_rewards)")
  if avg_testing_rewards > envs[0].rewardThreshold {
    break
  }
}

let result = envs[0].reset()
last_obs[0] = Tensor(from: result.0)
while episodes < 10 {
  let lastObs = graph.variable(last_obs[0].toGPU(0))
  let variable =
    (lastObs - obsRms.mean) ./ Functional.squareRoot(obsRms.variance).clamped(1e-5...)
  let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
  act.clamp(-1...1)
  let act_v = act.rawValue.toCPU()
  let (obs, _, done, _) = envs[0].step(action: Tensor(from: act_v))
  last_obs[0] = Tensor(from: obs)
  if done {
    let (obs, _) = envs[0].reset()
    last_obs[0] = Tensor(from: obs)
    episodes += 1
  }
  viewer.render()
}
