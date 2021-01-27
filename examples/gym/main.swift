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

let gym = Python.import("gym")

let env = gym.make("CartPole-v0")

env.seed(0)

let action_space = env.action_space

let graph = DynamicGraph()

for _ in 0..<10 {
  let _ = env.reset()
  while true {
    let action = action_space.sample()
    let (ob, reward, done, _) = env.step(action).tuple4
    print("\(ob), \(reward), \(done)")
    let tensor: Tensor<Float64> = try! Tensor(numpy: ob)
    let tensor32: Tensor<Float32> = Tensor(from: tensor)
    print(tensor32)
    if Bool(done)! {
      break
    }
  }
}
env.close()
