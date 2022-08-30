import Algorithms
import Foundation
import Gym
import NNC
import Numerics
import TensorBoard

typealias TargetEnv = Ant

let input_dim = TargetEnv.stateSize
let output_dim = TargetEnv.actionSpace.count
let action_range: Float = TargetEnv.actionSpace[0].upperBound

func NetA() -> Model {
  let x = Input()
  var y = Dense(count: 256)(x)
  y = ReLU()(y)
  y = Dense(count: 256)(y)
  y = ReLU()(y)
  let act = Dense(count: output_dim)(y)
  let sigma = Dense(count: output_dim)(y)
  return Model([x], [act, sigma])
}

@Sequential
func NetC() -> Model {
  Dense(count: 256)
  ReLU()
  Dense(count: 256)
  ReLU()
  Dense(count: 1)
}

let graph = DynamicGraph()
var sfmt = SFMT(seed: 10)
let summary = SummaryWriter(logDirectory: "/tmp/sac")

let buffer_size = 1_000_000
let actor_lr: Float = 1e-3
let critic_lr: Float = 1e-3
let gamma: Float = 0.99
let tau: Float = 0.005
let alpha: Float = 0.2
let alpha_lr: Float = 3e-4
let max_epoch = 200
let step_per_epoch = 5_000
let start_timestamps = 10_000
let step_per_collect = 1
let n_step = 1
let batch_size = 256
let training_num = 1
let testing_num = 10
let SIGMA_MIN: Float = -2
let SIGMA_MAX: Float = 2

let actor = NetA()
let critic1 = NetC()
let critic2 = NetC()
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
    public var action: Tensor<Float>
    public var observation: Tensor<Float>
    public init(action: Tensor<Float>, observation: Tensor<Float>) {
      self.action = action
      self.observation = observation
    }
  }
}

struct Replay {
  var reward: Float32  // Rewards for 0..<n_step - 1
  var terminated: Bool  // Whether this is terminated already.
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
var env_step = 0
var training_collector = Collector<Float, SAC.ContinuousState, TimeLimit<TargetEnv>, Double>(
  envs: envs
) {
  let obs = graph.variable(Tensor<Float>(from: $0).toGPU(0))
  let variable = obs
  let result = actor(inputs: variable)
  let act = result[0].as(of: Float32.self)
  let sigma = result[1].as(of: Float32.self)
  sigma.clamp(SIGMA_MIN...SIGMA_MAX)
  let n = graph.constant(Tensor<Float32>(.GPU(0), .C(output_dim)))
  n.randn(std: 1, mean: 0)
  let act_f = n .* Functional.exp(sigma) + act
  // This is tanh + action scaling
  act_f.tanh()
  let action = act_f.rawValue.toCPU()
  let map_action = (action_range * act_f).rawValue.toCPU()
  return (
    map_action,
    SAC.ContinuousState(action: action, observation: variable.rawValue.toCPU())
  )
}

var replays = [Replay]()
let total_discount = Array(repeating: gamma, count: n_step).reduce(1, *)
var one = graph.constant(.GPU(0), .NC(batch_size, output_dim), of: Float32.self)
one.full(1 + .ulpOfOne)
var logSqrt2Pi = graph.constant(.GPU(0), .NC(batch_size, output_dim), of: Float32.self)
logSqrt2Pi.full(.log((2 * .pi).squareRoot()))
let _ = training_collector.collect(nStep: start_timestamps)
for epoch in 0..<max_epoch {
  var step_in_epoch = 0
  while step_in_epoch < step_per_epoch {
    let stats = training_collector.collect(nStep: step_per_collect)
    let env_step_count = stats.stepCount
    let collectedData = training_collector.data
    training_collector.resetData(keepLastN: n_step)
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
          reward: rew, terminated: (i >= data.states.count - n_step),
          observation: state.observation,
          action: state.action,
          nextObservation: i + n_step < data.states.count - 1
            ? data.states[i + n_step].observation : data.lastObservation)
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
    var d = Tensor<Float32>(.CPU, .NC(batch_size, 1))
    for i in 0..<batch_size {
      let replay = batch[i % batch.count]
      obs[i, ...] = replay.observation
      obs_next[i, ...] = replay.nextObservation
      act[i, ...] = replay.action
      r[i, 0] = replay.reward
      d[i, 0] = replay.terminated ? 0 : total_discount
    }
    // Compute the q.
    let obs_next_v = graph.variable(obs_next.toGPU(0))
    var obs_act_next_v = graph.variable(
      .GPU(0), .NC(batch_size, input_dim + output_dim), of: Float32.self)
    let obs_next_result = actor(inputs: obs_next_v)
    let act_next = obs_next_result[0].as(of: Float32.self)
    let sigma_next = obs_next_result[1].as(of: Float32.self)
    sigma_next.clamp(SIGMA_MIN...SIGMA_MAX)
    let n = graph.constant(Tensor<Float32>(.GPU(0), .NC(batch_size, output_dim)))
    n.randn(std: 1, mean: 0)
    let exp_sigma_next = Functional.exp(sigma_next)
    let act_next_v = n .* exp_sigma_next + act_next
    let squashed_act_next_v = Functional.tanh(act_next_v)
    obs_act_next_v[0..<batch_size, 0..<input_dim] = obs_next_v
    obs_act_next_v[0..<batch_size, input_dim..<(input_dim + output_dim)] = squashed_act_next_v
    let target1_q = critic1Old(inputs: obs_act_next_v)[0].as(of: Float32.self)
    let target2_q = critic2Old(inputs: obs_act_next_v)[0].as(of: Float32.self)
    let var2_next = 1 / (2 * (exp_sigma_next .* exp_sigma_next))
    let log_prob_next =
      (act_next - act_next_v) .* (act_next - act_next_v) .* var2_next + sigma_next + logSqrt2Pi
      + Functional.log(one - squashed_act_next_v .* squashed_act_next_v).reduced(.sum, axis: [1])
    let target_q =
      Functional.min(target1_q, target2_q) + alpha * log_prob_next.reduced(.mean, axis: [1])
    let r_q = graph.constant(r.toGPU(0)) .+ graph.constant(d.toGPU(0)) .* target_q

    let obs_v = graph.variable(obs.toGPU(0))
    let act_v = graph.constant(act.toGPU(0))
    var obs_act_v = graph.variable(
      .GPU(0), .NC(batch_size, input_dim + output_dim), of: Float32.self)
    obs_act_v[0..<batch_size, 0..<input_dim] = obs_v
    obs_act_v[0..<batch_size, input_dim..<(input_dim + output_dim)] = act_v
    if env_step == 0 {
      // First time use critic, copy its parameters from criticOld.
      critic1.parameters.copy(from: critic1Old.parameters)
      critic2.parameters.copy(from: critic2Old.parameters)
    }

    // Need to weight this. (priority buffer).
    let pred1_q = critic1(inputs: obs_act_v)[0].as(of: Float32.self)
    let loss1 = MSELoss()(pred1_q, target: r_q)[0].as(of: Float32.self)
    let cpuLoss1 = loss1.toCPU()
    for i in 0..<batch_size {
      critic1Loss += cpuLoss1[i, 0]
    }
    loss1.backward(to: obs_act_v)
    critic1Optim.step()

    let pred2_q = critic2(inputs: obs_act_v)[0].as(of: Float32.self)
    let loss2 = MSELoss()(pred2_q, target: r_q)[0].as(of: Float32.self)
    let cpuLoss2 = loss2.toCPU()
    for i in 0..<batch_size {
      critic2Loss += cpuLoss2[i, 0]
    }
    loss2.backward(to: obs_act_v)
    critic2Optim.step()

    let new_obs_result = actor(inputs: obs_v)
    let new_act = new_obs_result[0].as(of: Float32.self)
    let new_sigma = new_obs_result[1].as(of: Float32.self)
    new_sigma.clamp(SIGMA_MIN...SIGMA_MAX)
    let new_n = graph.constant(Tensor<Float32>(.GPU(0), .NC(batch_size, output_dim)))
    new_n.randn(std: 1, mean: 0)
    let exp_new_sigma = Functional.exp(new_sigma)
    let new_act_v = new_n .* exp_new_sigma + new_act
    let squashed_new_act_v = Functional.tanh(new_act_v)
    var new_obs_act_v = graph.variable(
      .GPU(0), .NC(batch_size, input_dim + output_dim), of: Float32.self)
    new_obs_act_v[0..<batch_size, 0..<input_dim] = obs_v
    new_obs_act_v[0..<batch_size, input_dim..<(input_dim + output_dim)] = squashed_new_act_v
    let new_var2 = 1 / (2 * (exp_new_sigma .* exp_new_sigma))
    let new_log_prob =
      (new_act - new_act_v) .* (new_act - new_act_v) .* new_var2 + new_sigma
      + Functional.log(one - squashed_new_act_v .* squashed_new_act_v).reduced(.sum, axis: [1])
    let current_q1a = critic1(inputs: new_obs_act_v)[0].as(of: Float32.self)
    let current_q2a = critic2(inputs: new_obs_act_v)[0].as(of: Float32.self)
    let actor_loss =
      Functional.min(current_q1a, current_q2a) + alpha * new_log_prob.reduced(.mean, axis: [1])
    let cpuActorLoss = actor_loss.toCPU()
    for i in 0..<batch_size {
      actorLoss += cpuActorLoss[i, 0]
    }
    let grad = graph.variable(.GPU(0), .NC(batch_size, 1), of: Float32.self)
    grad.full(-1.0 / Float(batch_size))
    actor_loss.grad = grad
    actor_loss.backward(to: obs_v)
    actorOptim.step()

    critic1Old.parameters.lerp(tau, to: critic1.parameters)
    critic2Old.parameters.lerp(tau, to: critic2.parameters)

    env_step += env_step_count
    step_in_epoch += env_step_count

    critic1Loss = critic1Loss / Float(batch_size)
    critic2Loss = critic2Loss / Float(batch_size)
    actorLoss = -actorLoss / Float(batch_size)
    summary.addScalar("critic1_loss", critic1Loss, step: env_step)
    summary.addScalar("critic2_loss", critic2Loss, step: env_step)
    summary.addScalar("actor_loss", actorLoss, step: env_step)
    if stats.episodeReward.mean != 0 || stats.episodeReward.std != 0 {
      summary.addScalar("avg_reward", stats.episodeReward.mean, step: env_step)
    }
    if stats.episodeLength.mean != 0 || stats.episodeLength.std != 0 {
      summary.addScalar("avg_length", stats.episodeLength.mean, step: env_step)
    }
  }
  summary.addGraph("actor", actor)
  summary.addGraph("critic1", critic1)
  summary.addGraph("critic2", critic2)
  // Running test and print how many steps we can perform in an episode before it fails.
  let (obs, _) = testEnv.reset()
  var last_obs = Tensor<Float>(from: obs)
  var testing_rewards = [Float]()
  for _ in 0..<testing_num {
    var rewards: Float = 0
    while true {
      let lastObs = graph.variable(last_obs.toGPU(0))
      let variable = lastObs
      let act = actor(inputs: variable)[0].as(of: Float32.self)
      act.tanh()
      let act_v = (action_range * act).rawValue.toCPU()
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
  summary.addScalar("testing_reward", avg_testing_rewards.mean, step: epoch)
  if avg_testing_rewards.mean > TargetEnv.rewardThreshold {
    break
  }
}

let (obs, _) = testEnv.reset()
var last_obs = Tensor<Float>(from: obs)
while episodes < 10 {
  let lastObs = graph.variable(last_obs.toGPU(0))
  let variable = lastObs
  let act = actor(inputs: variable)[0].as(of: Float32.self)
  act.tanh()
  let act_v = (action_range * act).rawValue.toCPU()
  let (obs, _, done, _) = testEnv.step(action: Tensor(from: act_v))
  last_obs = Tensor(from: obs)
  if done {
    let (obs, _) = testEnv.reset()
    last_obs = Tensor(from: obs)
    episodes += 1
  }
  viewer.render()
}
