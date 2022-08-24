import Algorithms
import Foundation
import Gym
import GymVideo
import NNC
import NNCPythonConversion
import Numerics
import TensorBoard

typealias TargetEnv = Walker2D

let output_dim = TargetEnv.actionSpace.count
let action_range: Float = TargetEnv.actionSpace[0].upperBound

let graph = DynamicGraph()
var sfmt = SFMT(seed: 10)

DynamicGraph.setSeed(0)
var testEnv = TimeLimit(env: try TargetEnv(), maxEpisodeSteps: 1_000)
let _ = testEnv.reset(seed: 180)
let video = MuJoCoVideo(
  env: testEnv, filePath: "/home/liu/workspace/s4nnc/examples/random/random.mp4")
var episodes = 0
while episodes < 10 {
  let act = graph.variable(Tensor<Float32>(.GPU(0), .C(output_dim)))
  act.randn(std: 1, mean: 0)
  act.clamp(-1...1)
  let act_v = (action_range * act).rawValue.toCPU()
  let (_, _, done, _) = testEnv.step(action: Tensor(from: act_v))
  if done {
    let _ = testEnv.reset()
    episodes += 1
  }
  video.render()
}

video.close()
