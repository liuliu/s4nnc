import NNC
import PythonKit

let gym = Python.import("gym")

let env = gym.make("CartPole-v0")

env.seed(0)

let action_space = env.action_space

for _ in 0..<10 {
  let _ = env.reset()
  while true {
    let action = action_space.sample()
    let (ob, reward, done, _) = env.step(action).tuple4
    print("\(ob), \(reward), \(done)")
    if Bool(done)! {
      break
    }
  }
}
env.close()
