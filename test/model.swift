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
      ReLU()
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

  func testModelWithScalar() throws {
    let dynamicGraph = DynamicGraph()

    let tv0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = dynamicGraph.variable(Tensor<Float32>([2.2], .CPU, .C(1)))

    func MulAdd1() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let i3 = i2 + 1.2
      return Model([i0, i1], [i3])
    }
    let muladd1 = MulAdd1()
    let tv31 = DynamicGraph.Tensor<Float32>(muladd1(inputs: tv0, tv1)[0])
    XCTAssertEqual(tv31.rawValue[0], 1.1 * 2.2 + 1.2, accuracy: 1e-5)

    func MulAdd2() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let i3 = i2 - 1.2
      return Model([i0, i1], [i3])
    }
    let muladd2 = MulAdd2()
    let tv32 = DynamicGraph.Tensor<Float32>(muladd2(inputs: tv0, tv1)[0])
    XCTAssertEqual(tv32.rawValue[0], 1.1 * 2.2 - 1.2, accuracy: 1e-5)

    func MulAdd3() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let i3 = 2.2 + i2
      return Model([i0, i1], [i3])
    }
    let muladd3 = MulAdd3()
    let tv33 = DynamicGraph.Tensor<Float32>(muladd3(inputs: tv0, tv1)[0])
    XCTAssertEqual(tv33.rawValue[0], 1.1 * 2.2 + 2.2, accuracy: 1e-5)

    func MulAdd4() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let i3 = 1.2 - i2
      return Model([i0, i1], [i3])
    }
    let muladd4 = MulAdd4()
    let tv34 = DynamicGraph.Tensor<Float32>(muladd4(inputs: tv0, tv1)[0])
    XCTAssertEqual(tv34.rawValue[0], 1.2 - 1.1 * 2.2, accuracy: 1e-5)
  }

  func testModelWithParameter() throws {
    let dynamicGraph = DynamicGraph()

    let tv0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = dynamicGraph.variable(Tensor<Float32>([2.2], .CPU, .C(1)))

    func MulAdd() -> (Model, Model) {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let param = Parameter<Float32>(.CPU, .C(1))
      let i3 = i2 + param
      return (param, Model([i0, i1], [i3]))
    }
    let (param, muladd) = MulAdd()
    muladd.compile(inputs: tv0, tv1)
    param.weight.copy(from: Tensor<Float32>([3.1], .CPU, .C(1)))
    let tv3 = DynamicGraph.Tensor<Float32>(muladd(inputs: tv0, tv1)[0])
    XCTAssertEqual(tv3.rawValue[0], 1.1 * 2.2 + 3.1, accuracy: 1e-5)
  }

  func testModelScaledDotProductAttention() throws {
    let dynamicGraph = DynamicGraph()
    let q = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .NHWC(1, 8, 10, 20)))
    let k = dynamicGraph.variable(Tensor<Float32>([2.2], .CPU, .NHWC(1, 8, 20, 20)))
    let v = dynamicGraph.variable(Tensor<Float32>([2.2], .CPU, .NHWC(1, 8, 20, 30)))
    q.randn()
    k.randn()
    v.randn()
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1)
    let out = scaledDotProductAttention(queries: q, keys: k, values: v)
    var dot = Functional.matmul(left: q, right: k, rightTranspose: (2, 3))
    dot = dot.reshaped(.NC(8 * 10, 20))
    dot.softmax()
    dot = dot.reshaped(.NHWC(1, 8, 10, 20))
    let out2 = dot * v
    for i in 0..<8 {
      for j in 0..<10 {
        for k in 0..<30 {
          XCTAssertEqual(out[0, i, j, k], out2[0, i, j, k], accuracy: 1e-5)
        }
      }
    }
  }

  static let allTests = [
    ("testModel", testModel),
    ("testModelBuilder", testModelBuilder),
    ("testSequential", testSequential),
    ("testModelWithScalar", testModelWithScalar),
    ("testModelWithParameter", testModelWithParameter),
    ("testModelScaledDotProductAttention", testModelScaledDotProductAttention),
  ]
}
