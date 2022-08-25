import Algorithms
import Foundation
import Gym
import NNC
import NNCPythonConversion
import Numerics
import PythonKit

typealias TargetEnv = Walker2D

let input_dim = TargetEnv.stateSize
let output_dim = TargetEnv.actionSpace.count
let action_range: Float = TargetEnv.actionSpace[0].upperBound

func NetA() -> Model {
  var x: Model.IO = Input()
  x = Dense(count: 64)(x)
  x = ReLU()(x)
  x = Dense(count: 64)(x)
  x = ReLU()(x)
  let action = Dense(count: output_dim)(x)
  let sigma = Dense(count: output_dim)(x)
  return Model([x], [action, sigma])
}

@Sequential
func NetC() -> Model {
  Dense(count: 64)
  ReLU()
  Dense(count: 64)
  ReLU()
  Dense(count: 1)
}

let graph = DynamicGraph()
var sfmt = SFMT(seed: 10)

let buffer_size = 1_000_000
let actor_lr: Float = 1e-3
let critic_lr: Float = 1e-3
let gamma: Float = 0.99
let tau: Float = 0.005
let alpha: Float = 0.2
let alpha_lr: Float = 3e-4
let max_epoch = 200
let step_per_epoch = 5000
let step_per_collect = 1
let update_per_step = 1
let n_step = 1
let batch_size = 256
let training_num = 1
let testing_num = 10

let actor = NetA()
let critic1 = NetC()
let critic2 = NetC()
let actorOld = actor.copied()
let critic1Old = critic1.copied()
let critic2Old = critic2.copied()

var actorOptim = AdamOptimizer(graph, rate: actor_lr)
actorOptim.parameters = [actor.parameters]
var critic1Optim = AdamOptimizer(graph, rate: critic_lr)
critic1Optim.parameters = [critic1.parameters]
var critic2Optim = AdamOptimizer(graph, rate: critic_lr)
critic2Optim.parameters = [critic2.parameters]

enum SAC {
  struct ContinuousState {
    public var centroid: Tensor<Float>
    public var action: Tensor<Float>
    public var observation: Tensor<Float>
    public init(centroid: Tensor<Float>, action: Tensor<Float>, observation: Tensor<Float>) {
      self.centroid = centroid
      self.action = action
      self.observation = observation
    }
  }
}

struct Replay {
  var reward: Float32  // Rewards for 0..<n_step - 1
  var observation: Tensor<Float32>  // The state before action.
  var action: Tensor<Float32>  // The act taken in the episode.
  var nextObservation: Tensor<Float32>  // The state after the action.
}

var envs = [TimeLimit<TargetEnv>]()
for i in 0..<training_num {
  let env = TimeLimit(env: try TargetEnv(), maxEpisodeSteps: 1_000)
  let _ = env.reset(seed: i)
  envs.append(env)
}
DynamicGraph.setSeed(0)
var testEnv = TimeLimit(env: try TargetEnv(), maxEpisodeSteps: 1_000)
let _ = testEnv.reset(seed: 180)
let viewer = MuJoCoViewer(env: testEnv)
var episodes = 0

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
var training_collector = Collector<Float, SAC.ContinuousState, TimeLimit<TargetEnv>, Double>(
  envs: envs
) {
  let obs = graph.variable(Tensor<Float>(from: $0).toGPU(0))
  let variable = obsRms.norm(obs)
  variable.clamp(-10...10)
  obsRms.update([obs])
  let ret = actor(inputs: variable)
  let act = ret[0].as(of: Float32.self)
  let sigma = ret[1].as(of: Float32.self)
  let n = graph.variable(Tensor<Float32>(.GPU(0), .C(output_dim)))
  n.randn(std: 1, mean: 0)
  let act_f = n .* Functional.exp(sigma) + act
  let action = act_f.rawValue.toCPU()
  // This is clamp + action scaling
  act_f.tanh()
  let map_action = (action_range * act_f).rawValue.toCPU()
  let act_mu = act.rawValue.toCPU()
  return (
    map_action,
    SAC.ContinuousState(centroid: act_mu, action: action, observation: variable.rawValue.toCPU())
  )
}

var replays = [Replay]()

for epoch in 0..<max_epoch {
  var step_in_epoch = 0
  while step_in_epoch < step_per_epoch {
    let stats = training_collector.collect(nStep: step_per_collect)
    let env_step_count = stats.stepCount
    var collectedData = training_collector.data
    training_collector.resetData(keepLastN: n_step)
    for (i, buffer) in collectedData.enumerated() {
      let obs = graph.variable(Tensor<Float>(from: buffer.lastObservation).toGPU(0))
      let variable = obsRms.norm(obs)
      variable.clamp(-10...10)
      obsRms.update([obs])
      collectedData[i].lastObservation = variable.rawValue.toCPU()
    }
    env_step += env_step_count
    for data in collectedData {
      // Ignore the last one if it is not terminated yet (thus, we always have obs / obs_next pair).
      let count = data.rewards.count - (data.terminated ? 0 : 1)
      for i in 0..<count {
        var discount: Float = 1
        var rew: Float = 0
        let rewards = data.rewards[i..<min(i + n_step, data.rewards.count)]
        for reward in rewards {
          rew += reward * discount
          discount *= gamma
        }
        let state = data.states[i]
        let replay = Replay(
          reward: rew, observation: state.observation, action: state.action,
          nextObservation: i < data.states.count - 1
            ? data.states[i + 1].observation : data.lastObservation)
        replays.append(replay)
      }
    }
    if replays.count > buffer_size {  // Only keep the most recent ones.
      replays.removeFirst(replays.count - buffer_size)
    }
    guard replays.count >= batch_size else {
      continue
    }
    var critic1Loss: Float = 0
    var critic2Loss: Float = 0
    var actorLoss: Float = 0
    let batch = replays.randomSample(count: batch_size, using: &sfmt)
    var obs = Tensor<Float32>(.CPU, .NC(batch_size, input_dim))
    var obs_next = Tensor<Float32>(.CPU, .NC(batch_size, input_dim))
    var act = Tensor<Float32>(.CPU, .NC(batch_size, output_dim))
    var r = Tensor<Float32>(.CPU, .NC(batch_size, 1))
    /*
    for i in 0..<batch_size {
      let replay = batch[i % batch.count]
      obs[i, ...] = replay.observations[0]
      obs_next[i, ...] = replay.observations[1]
      act[i, ...] = replay.actions[0]
      r[i, 0] = replay.reward
    }
    // Compute the q.
    let obs_next_v = graph.constant(obs_next.toGPU(0))
    if update_step == 0 && epoch == 0 && step_in_epoch == 0 {
      // Firs time use actorOld, copy its parameters from actor.
      actorOld.parameters.copy(from: actor.parameters)
    }
    var act_next_v = DynamicGraph.Tensor<Float32>(actorOld(inputs: obs_next_v)[0])
    let policy_noise_v: DynamicGraph.Tensor<Float32> = graph.constant(.GPU(0), .NC(batch_size, 1))
    policy_noise_v.randn(std: policy_noise)
    policy_noise_v.clamp(-noise_clip...noise_clip)
    act_next_v = act_next_v + policy_noise_v
    act_next_v.clamp(actionLow...actionHigh)
    var obs_act_next_v: DynamicGraph.Tensor<Float32> = graph.constant(.GPU(0), .NC(batch_size, 4))
    obs_act_next_v[0..<batch_size, 0..<3] = obs_next_v
    obs_act_next_v[0..<batch_size, 3..<4] = act_next_v
    let target1_q = DynamicGraph.Tensor<Float32>(critic1Old(inputs: obs_act_next_v)[0])
    let target2_q = DynamicGraph.Tensor<Float32>(critic2Old(inputs: obs_act_next_v)[0])
    let target_q = Functional.min(target1_q, target2_q)
    let r_q = graph.constant(r.toGPU(0)) .+ graph.constant(d.toGPU(0)) .* target_q
    let obs_v = graph.variable(obs.toGPU(0))
    let act_v = graph.constant(act.toGPU(0))
    var obs_act_v: DynamicGraph.Tensor<Float32> = graph.variable(.GPU(0), .NC(batch_size, 4))
    obs_act_v[0..<batch_size, 0..<3] = obs_v
    obs_act_v[0..<batch_size, 3..<4] = act_v
    if update_step == 0 && epoch == 0 && step_in_epoch == 0 {
      // First time use critic, copy its parameters from criticOld.
      critic1.parameters.copy(from: critic1Old.parameters)
      critic2.parameters.copy(from: critic2Old.parameters)
    }

    let pred1_q = DynamicGraph.Tensor<Float32>(critic1(inputs: obs_act_v)[0])
    let loss1 = DynamicGraph.Tensor<Float32>(SmoothL1Loss()(pred1_q, target: r_q)[0])
    let cpuLoss1 = loss1.toCPU()
    for i in 0..<batch_size {
      critic1Loss += cpuLoss1[i, 0]
    }
    loss1.backward(to: obs_act_v)
    critic1Optim.step()

    let pred2_q = DynamicGraph.Tensor<Float32>(critic2(inputs: obs_act_v)[0])
    let loss2 = DynamicGraph.Tensor<Float32>(SmoothL1Loss()(pred2_q, target: r_q)[0])
    let cpuLoss2 = loss2.toCPU()
    for i in 0..<batch_size {
      critic2Loss += cpuLoss2[i, 0]
    }
    loss2.backward(to: obs_act_v)
    critic2Optim.step()

    critic1Old.parameters.lerp(tau, to: critic1.parameters)
    critic2Old.parameters.lerp(tau, to: critic2.parameters)

    if step_count % update_actor_freq == 0 {
      let new_act_v = DynamicGraph.Tensor<Float32>(actor(inputs: obs_v)[0])
      new_act_v.clamp(actionLow...actionHigh)
      var new_obs_act_v: DynamicGraph.Tensor<Float32> = graph.variable(
        .GPU(0), .NC(batch_size, 4))
      new_obs_act_v[0..<batch_size, 0..<3] = obs_v
      new_obs_act_v[0..<batch_size, 3..<4] = new_act_v
      let actor_loss = DynamicGraph.Tensor<Float32>(critic1(inputs: new_obs_act_v)[0])
      let cpuActorLoss = actor_loss.toCPU()
      for i in 0..<batch_size {
        actorLoss += cpuActorLoss[i, 0]
      }
      let grad: DynamicGraph.Tensor<Float32> = graph.variable(.GPU(0), .NC(batch_size, 1))
      // Run gradient ascent, therefore, the negative sign for the gradient. It is the same as:
      // actor_loss = -critic(inputs: new_obs_act_v)[0]
      grad.full(-1.0 / Float(batch_size))
      actor_loss.grad = grad
      actor_loss.backward(to: obs_v)
      actorOptim.step()
      actorOld.parameters.lerp(tau, to: actor.parameters)
    }

    step_count += 1
    step_in_epoch += 1

    critic1Loss = critic1Loss / Float(batch_size * update_steps)
    critic2Loss = critic2Loss / Float(batch_size * update_steps)
    actorLoss = -actorLoss / Float(batch_size * update_steps)
    */
    print(
      "Epoch \(epoch), step \(step_in_epoch), critic1 loss \(critic1Loss), critic2 loss \(critic2Loss), actor loss \(actorLoss)"
    )
  }
  // Running test and print how many steps we can perform in an episode before it fails.
  var testing_rewards: Float = 0
  /*
  for _ in 0..<testing_num {
    while true {
      let variable = graph.variable(last_obs.toGPU(0))
      let act = DynamicGraph.Tensor<Float32>(actor(inputs: variable)[0])
      act.clamp(-1...1)
      let act_v = act.rawValue.toCPU()
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
      store.write("ddpg_critic1", model: critic1)
      store.write("ddpg_critic2", model: critic2)
    }
    break
  }
  */
}
