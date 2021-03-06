import NNC
import XCTest

final class ModelTests: XCTestCase {

  func testModel() throws {
    let dynamicGraph = DynamicGraph()

    func MulAdd() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let i3 = Input()
      let i4 = i2 - i3
      return Model([i0, i1, i3], [i4])
    }

    let muladd = MulAdd()
    let tv0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = dynamicGraph.variable(Tensor<Float32>([2.2], .CPU, .C(1)))
    let tv2 = dynamicGraph.variable(Tensor<Float32>([0.2], .CPU, .C(1)))
    let tv3 = DynamicGraph.Tensor<Float32>(muladd(inputs: tv0, tv1, tv2)[0])
    XCTAssertEqual(tv3.rawValue[0], 1.1 * 2.2 - 0.2, accuracy: 1e-5)
  }

  func testModelBuilder() throws {
    let dynamicGraph = DynamicGraph()

    let builder = ModelBuilder { inputs in
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      return Model([i0, i1], [i2])
    }

    let b0 = dynamicGraph.variable(Tensor<Float32>([1.2], .CPU, .C(1)))
    let b1 = dynamicGraph.constant(Tensor<Float32>([2.2], .CPU, .C(1)))
    let b2 = DynamicGraph.Tensor<Float32>(builder(inputs: b0, b1)[0])
    XCTAssertEqual(b2.rawValue[0], 1.2 * 2.2, accuracy: 1e-5)

    let b3 = dynamicGraph.variable(Tensor<Float32>([1.2, 2.2], .CPU, .C(2)))
    let b4 = dynamicGraph.constant(Tensor<Float32>([2.2, 3.3], .CPU, .C(2)))
    let b5 = DynamicGraph.Tensor<Float32>(builder(inputs: b3, b4)[0])
    XCTAssertEqual(b5.rawValue[0], 1.2 * 2.2, accuracy: 1e-5)
    XCTAssertEqual(b5.rawValue[1], 2.2 * 3.3, accuracy: 1e-5)
  }

  func testSequential() throws {
    let dynamicGraph = DynamicGraph()

    @Sequential
    func MulAdd() -> Model {
      Dense(count: 1)
      RELU()
    }

    let muladd = MulAdd()
    let tv0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = dynamicGraph.variable(Tensor<Float32>([-2.2], .CPU, .C(1)))
    let _ = DynamicGraph.Tensor<Float32>(muladd(inputs: tv0)[0])
    muladd.parameters.clamp(1...1)
    let tv2 = DynamicGraph.Tensor<Float32>(muladd(inputs: tv0)[0])
    let tv3 = DynamicGraph.Tensor<Float32>(muladd(inputs: tv1)[0])
    XCTAssertEqual(tv2.rawValue[0], 2.1, accuracy: 1e-5)
    XCTAssertEqual(tv3.rawValue[0], 0, accuracy: 1e-5)
  }

  static let allTests = [
    ("testModel", testModel),
    ("testModelBuilder", testModelBuilder),
    ("testSequential", testSequential),
  ]
}
