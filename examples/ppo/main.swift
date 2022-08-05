import Algorithms
import Foundation
import Gym
import NNC
import NNCPythonConversion
import Numerics
import TensorBoard

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
let summary = SummaryWriter(logDirectory: "/tmp/ppo")

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
for i in 0..<training_num {
  let env = TimeLimit(env: try Ant(), maxEpisodeSteps: 1_000)
  let _ = env.reset(seed: i)
  envs.append(env)
}
DynamicGraph.setSeed(0)
var testEnv = TimeLimit(env: try Ant(), maxEpisodeSteps: 1_000)
let _ = testEnv.reset(seed: 180)
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
var env_step = 0
var initActorLastLayer = false
var training_collector = Collector<Float, PPO.ContinuousActionSpace, TimeLimit<Ant>, Double>(
  envs: envs
) {
  let obs = graph.variable(Tensor<Float>(from: $0).toGPU(0))
  let variable = obsRms.norm(obs)
  obsRms.update([obs])
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
  return (
    act_f, PPO.ContinuousActionSpace(centroid: act_mu, observation: variable.rawValue.toCPU())
  )
}
var ppo = PPO(graph: graph) {
  let variable = graph.variable($0.toGPU(0))
  return DynamicGraph.Tensor<Float32>(critic(inputs: variable)[0]).rawValue.toCPU()
}
for epoch in 0..<max_epoch {
  var step_in_epoch = 0
  while step_in_epoch < step_per_epoch {
    let stats = training_collector.collect(nStep: collect_per_step)
    let env_step_count = stats.stepCount
    var collectedData = training_collector.data
    training_collector.resetData()
    for (i, buffer) in collectedData.enumerated() {
      let obs = graph.variable(Tensor<Float>(from: buffer.lastObservation).toGPU(0))
      let variable = obsRms.norm(obs)
      obsRms.update([obs])
      collectedData[i].lastObservation = variable.rawValue.toCPU()
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
    let oldDistributions = ppo.distributions(scale: scale.toCPU().rawValue, from: collectedData)
    for _ in 0..<update_per_step {
      let (returns, advantages) = ppo.computeReturns(from: collectedData)
      var dataframe = PPO.samples(
        from: collectedData, episodeCount: batch_size, using: &sfmt, returns: returns,
        advantages: advantages, oldDistributions: oldDistributions)
      dataframe.shuffle()
      let batched = dataframe[
        "observations", "actions", "returns", "advantages", "oldDistributions"
      ].combine(size: batch_size)
      for batch in batched["observations", "actions", "returns", "advantages", "oldDistributions"] {
        let obs = batch[0] as! Tensor<Float32>
        let act = batch[1] as! Tensor<Float32>
        let returns = batch[2] as! Tensor<Float32>
        let advantages = batch[3] as! Tensor<Float32>
        let distOld = batch[4] as! Tensor<Float32>
        let variable = graph.variable(obs.toGPU(0))
        let mu = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
        let actv = graph.constant(act.toGPU(0))
        let distOldv = graph.constant(distOld.toGPU(0))
        let advantagesv = graph.constant(advantages.toGPU(0))
        let clip_loss = PPO.ClipLoss(epsilon: eps_clip, entropyCoefficient: ent_coef)(
          mu, oldAction: actv, oldDistribution: distOldv, advantages: advantagesv, scale: scale)
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
      "rew std \(ppo.statistics.rewardsNormalization.std), log scale [\(scaleCPU[0]), \(scaleCPU[1]), \(scaleCPU[2]), \(scaleCPU[3]), \(scaleCPU[4]), \(scaleCPU[5])]"
    )
    print(
      "Epoch \(epoch), step \(env_step), critic loss \(criticLoss), actor loss \(actorLoss), reward \(stats.episodeReward.mean) (±\(stats.episodeReward.std))"
    )
    summary.addGraph("actor", actor)
    summary.addGraph("critic", critic)
    summary.addScalar("critic_loss", criticLoss, step: epoch)
    summary.addScalar("actor_loss", actorLoss, step: epoch)
    summary.addScalar("avg_reward", stats.episodeReward.mean, step: epoch)
  }
  // Running test and print how many steps we can perform in an episode before it fails.
  let (obs, _) = testEnv.reset()
  var last_obs = Tensor<Float>(from: obs)
  summary.addHistogram("init_obs", last_obs, step: epoch)
  var testing_rewards = [Float]()
  for _ in 0..<testing_num {
    var rewards: Float = 0
    while true {
      let lastObs = graph.variable(last_obs.toGPU(0))
      let variable = obsRms.norm(lastObs)
      let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
      act.clamp(-1...1)
      let act_v = act.rawValue.toCPU()
      let (obs, reward, done, _) = testEnv.step(action: Tensor(from: act_v))
      last_obs = Tensor(from: obs)
      rewards += reward
      if done {
        testing_rewards.append(rewards)
        let (obs, _) = testEnv.reset()
        last_obs = Tensor(from: obs)
        break
      }
    }
  }
  let avg_testing_rewards = NumericalStatistics(testing_rewards)
  print("Epoch \(epoch), testing reward \(avg_testing_rewards.mean) (±\(avg_testing_rewards.std))")
  if avg_testing_rewards.mean > testEnv.rewardThreshold {
    break
  }
}

let (obs, _) = testEnv.reset()
var last_obs = Tensor<Float>(from: obs)
while episodes < 10 {
  let lastObs = graph.variable(last_obs.toGPU(0))
  let variable = obsRms.norm(lastObs)
  let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
  act.clamp(-1...1)
  let act_v = act.rawValue.toCPU()
  let (obs, _, done, _) = testEnv.step(action: Tensor(from: act_v))
  last_obs = Tensor(from: obs)
  if done {
    let (obs, _) = testEnv.reset()
    last_obs = Tensor(from: obs)
    episodes += 1
  }
  viewer.render()
}
