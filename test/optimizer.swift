import NNC
import XCTest

final class OptimizerTests: XCTestCase {

  func testSGDOnModel() throws {
    let dynamicGraph = DynamicGraph()
    let linear = Dense(count: 1)
    let z = dynamicGraph.variable(Tensor<Float32>([5], .C(1)))
    var sgd = SGDOptimizer(
      dynamicGraph, nesterov: false, rate: 0.01, scale: 1, decay: 0.01, momentum: 0, dampening: 0)
    sgd.parameters = [linear.parameters]
    for i in 0..<100 {
      let x: DynamicGraph.Tensor<Float32>
      if i % 2 == 1 {
        x = dynamicGraph.variable(.CPU, .NC(2, 1))
        x[0, 0] = 10
        x[1, 0] = 10
      } else {
        x = dynamicGraph.variable(.CPU, .C(1))
        x[0] = 10
      }
      var y = Functional.log(x)
      y = linear(y)
      y = y - z
      let f = y .* y
      f.backward(to: x)
      sgd.step()
    }
    let x: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .C(1))
    x[0] = 10
    var y = Functional.log(x)
    y = linear(y)
    let bar = abs(y[0] - 5)
    XCTAssert(bar < 1e-2)
  }

  func testSGDOnGraph() throws {
    let dynamicGraph = DynamicGraph()
    let weight: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .C(1))
    let bias = dynamicGraph.variable(Tensor<Float32>([0], .C(1)))
    weight.rand(-1, 1)
    let z = dynamicGraph.variable(Tensor<Float32>([5], .C(1)))
    var sgd = SGDOptimizer(
      dynamicGraph, nesterov: false, rate: 0.01, scale: 1, decay: 0.01, momentum: 0, dampening: 0)
    sgd.parameters = [weight, bias]
    for i in 0..<100 {
      let x: DynamicGraph.Tensor<Float32>
      if i % 2 == 1 {
        x = dynamicGraph.variable(.CPU, .NC(2, 1))
        x[0, 0] = 10
        x[1, 0] = 10
      } else {
        x = dynamicGraph.variable(.CPU, .C(1))
        x[0] = 10
      }
      var y = Functional.log(x)
      y = y * weight + bias
      y = y - z
      let f = y .* y
      f.backward(to: x)
      sgd.step()
    }
    let x: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .C(1))
    x[0] = 10
    var y = Functional.log(x)
    y = y * weight + bias
    let bar = abs(y[0] - 5)
    XCTAssert(bar < 1e-2)
  }

  static let allTests = [
    ("testSGDOnModel", testSGDOnModel),
    ("testSGDOnGraph", testSGDOnGraph),
  ]
}
