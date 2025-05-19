import NNC
import XCTest

final class GraphTests: XCTestCase {

  func testGEMM() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2], .CPU, .NC(2, 1)))
    let a1 = dynamicGraph.variable(Tensor<Float32>([2.2, 3.3], .CPU, .NC(1, 2)))
    let a2 = a0 * a1
    XCTAssertEqual(a2.rawValue.shape, [2, 2])
    XCTAssertEqual(a2.rawValue[0, 0], 1.1 * 2.2)
    XCTAssertEqual(a2.rawValue[0, 1], 1.1 * 3.3)
    XCTAssertEqual(a2.rawValue[1, 0], 2.2 * 2.2)
    XCTAssertEqual(a2.rawValue[1, 1], 2.2 * 3.3)
  }

  func testGEMMGrad() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2], .CPU, .NC(2, 1)))
    a0.requiresGrad = true
    let a1 = dynamicGraph.variable(Tensor<Float32>([2.2, 3.3], .CPU, .NC(1, 2)))
    a1.requiresGrad = true
    let a2 = a0 * a1
    a2.backward(to: a0)
    let a0Grad = DynamicGraph.Tensor<Float32>(a0.grad!)
    let a1Grad = DynamicGraph.Tensor<Float32>(a1.grad!)
    XCTAssertEqual(a0Grad.rawValue[0, 0], 5.5, accuracy: 1e-5)
    XCTAssertEqual(a0Grad.rawValue[1, 0], 5.5, accuracy: 1e-5)
    XCTAssertEqual(a1Grad.rawValue[0, 0], 3.3, accuracy: 1e-5)
    XCTAssertEqual(a1Grad.rawValue[0, 1], 3.3, accuracy: 1e-5)
  }

  func testFull() throws {
    let dynamicGraph = DynamicGraph()
    let a0: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 1))
    a0.full(10)
    XCTAssertEqual(a0.rawValue[0, 0], 10, accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[1, 0], 10, accuracy: 1e-5)
    a0.full(-1)
    XCTAssertEqual(a0.rawValue[0, 0], -1, accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[1, 0], -1, accuracy: 1e-5)
  }

  func testLerp() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(.CPU, .NC(2, 1), of: Float32.self)
    let a1: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 1))
    a0.full(10)
    a1.full(-1)
    a0.lerp(0.3, to: a1)
    XCTAssertEqual(a0.rawValue[0, 0], 6.7, accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[1, 0], 6.7, accuracy: 1e-5)
  }

  func testClamp() throws {
    let dynamicGraph = DynamicGraph()
    let a0: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 1))
    a0[0, 0] = 10
    a0[1, 0] = 2
    let a1 = dynamicGraph.variable(.CPU, .NC(2, 1), of: Float32.self)
    a1[0, 0] = -1
    a1[1, 0] = -2
    let a3 = a0.clamped(0...8)
    a0.clamp(3...6)
    let a2 = a1.clamped((-1.1)...)
    XCTAssertEqual(a0.rawValue[0, 0], 6, accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[1, 0], 3, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[0, 0], -1, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 0], -1.1, accuracy: 1e-5)
    XCTAssertEqual(a3.rawValue[0, 0], 8, accuracy: 1e-5)
    XCTAssertEqual(a3.rawValue[1, 0], 2, accuracy: 1e-5)
  }

  func testPartialAssign() throws {
    let dynamicGraph = DynamicGraph()
    let a0: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 1))
    a0[0, 0] = 10
    a0[1, 0] = 2
    a0[2, 0] = 5
    let a1: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 1))
    a1[0, 0] = -1
    a1[1, 0] = -2
    a1[2, 0] = -3
    var a2: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 2))
    a2[0..<3, 0..<1] = a0[0..<3, 0..<1]
    a2[0..<3, 1..<2] = a1[0..<3, 0..<1]
    XCTAssertEqual(a2.rawValue[0, 0], 10, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 0], 2, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[2, 0], 5, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[0, 1], -1, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 1], -2, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[2, 1], -3, accuracy: 1e-5)
  }

  func testPartialAssignWithIndices() throws {
    let dynamicGraph = DynamicGraph()
    let a0: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a0[0, 0] = 10
    a0[0, 1] = 2
    a0[0, 2] = 5
    let a1: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a1[0, 0] = -1
    a1[0, 1] = -2
    a1[0, 2] = -3
    var a2: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 3))
    a2[0, 0..<2] = a0[0, 1..<3]
    a2[1, 1..<3] = a1[0, 0..<2]
    XCTAssertEqual(a2.rawValue[0, 0], 2, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[0, 1], 5, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 1], -1, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 2], -2, accuracy: 1e-5)
  }

  func testPartialAssignWithUnboundedRange() throws {
    let dynamicGraph = DynamicGraph()
    let a0: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a0[0, 0] = 10
    a0[0, 1] = 2
    a0[0, 2] = 5
    let a1: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a1[0, 0] = -1
    a1[0, 1] = -2
    a1[0, 2] = -3
    var a2: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 3))
    a2[0, ...] = a0[...]
    a2[1, ...] = a1[...]
    XCTAssertEqual(a2.rawValue[0, 0], 10, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[0, 1], 2, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[0, 2], 5, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 0], -1, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 1], -2, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 2], -3, accuracy: 1e-5)
  }

  func testPartialAssignWithGroup() throws {
    let dynamicGraph = DynamicGraph()
    let a00: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 1))
    a00[0, 0] = 10
    a00[1, 0] = 2
    a00[2, 0] = 5
    let a01: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 1))
    a01[0, 0] = 2
    a01[1, 0] = 3
    a01[2, 0] = 0
    let a0 = DynamicGraph.Group(a00, a01)
    let a10: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 1))
    a10[0, 0] = -1
    a10[1, 0] = -2
    a10[2, 0] = -3
    let a11: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 1))
    a11[0, 0] = 1
    a11[1, 0] = 2
    a11[2, 0] = 3
    let a1 = DynamicGraph.Group(a10, a11)
    let a20: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 2))
    let a21: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 2))
    var a2 = DynamicGraph.Group(a20, a21)
    a2[0..<3, 0..<1] = a0[0..<3, 0..<1]
    a2[0..<3, 1..<2] = a1[0..<3, 0..<1]
    XCTAssertEqual(a20.rawValue[0, 0], 10, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[1, 0], 2, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[2, 0], 5, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[0, 1], -1, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[1, 1], -2, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[2, 1], -3, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[0, 0], 2, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[1, 0], 3, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[2, 0], 0, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[0, 1], 1, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[1, 1], 2, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[2, 1], 3, accuracy: 1e-5)
  }

  func testPartialAssignWithGroupWithIndices() throws {
    let dynamicGraph = DynamicGraph()
    let a00: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a00[0, 0] = 10
    a00[0, 1] = 2
    a00[0, 2] = 5
    let a01: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a01[0, 0] = 2
    a01[0, 1] = 3
    a01[0, 2] = 0
    let a0 = DynamicGraph.Group(a00, a01)
    let a10: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a10[0, 0] = -1
    a10[0, 1] = -2
    a10[0, 2] = -3
    let a11: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a11[0, 0] = 1
    a11[0, 1] = 2
    a11[0, 2] = 3
    let a1 = DynamicGraph.Group(a10, a11)
    let a20: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 3))
    let a21: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 3))
    var a2 = DynamicGraph.Group(a20, a21)
    a2[0, 0..<2] = a0[0, 1..<3]
    a2[1, 1..<3] = a1[0, 0..<2]
    XCTAssertEqual(a20.rawValue[0, 0], 2, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[0, 1], 5, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[1, 1], -1, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[1, 2], -2, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[0, 0], 3, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[0, 1], 0, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[1, 1], 1, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[1, 2], 2, accuracy: 1e-5)
  }

  func testPartialAssignWithGroupWithUnboundedRange() throws {
    let dynamicGraph = DynamicGraph()
    let a00: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a00[0, 0] = 10
    a00[0, 1] = 2
    a00[0, 2] = 5
    let a01: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a01[0, 0] = 2
    a01[0, 1] = 3
    a01[0, 2] = 0
    let a0 = DynamicGraph.Group(a00, a01)
    let a10: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a10[0, 0] = -1
    a10[0, 1] = -2
    a10[0, 2] = -3
    let a11: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(1, 3))
    a11[0, 0] = 1
    a11[0, 1] = 2
    a11[0, 2] = 3
    let a1 = DynamicGraph.Group(a10, a11)
    let a20: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 3))
    let a21: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 3))
    var a2 = DynamicGraph.Group(a20, a21)
    a2[0, ...] = a0[...]
    a2[1, ...] = a1[...]
    XCTAssertEqual(a20.rawValue[0, 0], 10, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[0, 1], 2, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[0, 2], 5, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[1, 0], -1, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[1, 1], -2, accuracy: 1e-5)
    XCTAssertEqual(a20.rawValue[1, 2], -3, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[0, 0], 2, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[0, 1], 3, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[0, 2], 0, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[1, 0], 1, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[1, 1], 2, accuracy: 1e-5)
    XCTAssertEqual(a21.rawValue[1, 2], 3, accuracy: 1e-5)
  }

  func testMin() throws {
    let dynamicGraph = DynamicGraph()
    let a0: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 1))
    a0[0, 0] = 10
    a0[1, 0] = 1
    let a1: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 1))
    a1[0, 0] = 1
    a1[1, 0] = 5
    let a2 = Functional.min(a0, a1)
    XCTAssertEqual(a2.rawValue[0, 0], 1, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 0], 1, accuracy: 1e-5)
  }

  func testMax() throws {
    let dynamicGraph = DynamicGraph()
    let a0: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 1))
    a0[0, 0] = 10
    a0[1, 0] = 1
    let a1: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 1))
    a1[0, 0] = 1
    a1[1, 0] = 5
    let a2 = Functional.max(a0, a1)
    XCTAssertEqual(a2.rawValue[0, 0], 10, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 0], 5, accuracy: 1e-5)
  }

  func testScale() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, -1.1], .CPU, .C(2)))
    a0.scale(by: -2)
    XCTAssertEqual(a0.rawValue[0], -2.2, accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[1], 2.2, accuracy: 1e-5)
  }

  func testReLU() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, -1.1], .CPU, .C(2)))
    a0.ReLU()
    XCTAssertEqual(a0.rawValue[0], 1.1, accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[1], 0, accuracy: 1e-5)
  }

  func testSigmoid() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    a0.sigmoid()
    XCTAssertEqual(a0.rawValue[0], 1.0 / (1.0 + exp(-1.1)), accuracy: 1e-5)
  }

  func testTanh() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([4.4], .CPU, .C(1)))
    a0.tanh()
    XCTAssertEqual(a0.rawValue[0], tanh(4.4), accuracy: 1e-5)
  }

  func testSwish() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([4.4], .CPU, .C(1)))
    a0.swish()
    XCTAssertEqual(a0.rawValue[0], 4.4 / (1.0 + exp(-4.4)), accuracy: 1e-5)
  }

  func testSoftmax() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.0, 2.0, 3.0], .CPU, .C(3)))
    a0.softmax()
    let e0 = exp(1.0)
    let e1 = exp(2.0)
    let e2 = exp(3.0)
    let sum = e0 + e1 + e2
    XCTAssertEqual(a0.rawValue[0], Float32(e0 / sum), accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[1], Float32(e1 / sum), accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[2], Float32(e2 / sum), accuracy: 1e-5)
  }

  func testArgmax() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.2, 2.2, 3.2, 3.4], .CPU, .C(4)))
    let b0 = Functional.argmax(a0, axis: 0)
    XCTAssertEqual(b0.rawValue[0], 3)
    let a1 = dynamicGraph.variable(Tensor<Float32>([1, 3.1, 2, 2, 3, 4], .CPU, .NC(2, 3)))
    let b10 = Functional.argmax(a1, axis: 0)
    let b11 = Functional.argmax(a1, axis: 1)
    XCTAssertEqual(b10.rawValue[0, 0], 1)
    XCTAssertEqual(b10.rawValue[0, 1], 0)
    XCTAssertEqual(b10.rawValue[0, 2], 1)
    XCTAssertEqual(b11.rawValue[0, 0], 1)
    XCTAssertEqual(b11.rawValue[1, 0], 2)
  }

  func testMaskedFill() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1, 2, 3, 4], .CPU, .C(4)))
    let m0 = dynamicGraph.variable(Tensor<Int32>([0, 3, 0, 1], .CPU, .C(4)))
    let b0 = Functional.maskedFill(input: a0, mask: m0, equalTo: 3, fillWith: 5)
    XCTAssertEqual(b0.rawValue[0], 1, accuracy: 1e-5)
    XCTAssertEqual(b0.rawValue[1], 5, accuracy: 1e-5)
    XCTAssertEqual(b0.rawValue[2], 3, accuracy: 1e-5)
    XCTAssertEqual(b0.rawValue[3], 4, accuracy: 1e-5)
  }

  func testPermute() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(
      Tensor<Float32>(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], .CPU,
        .HWC(2, 3, 4)))
    let a1 = a0.permuted(2, 0, 1)
    XCTAssertEqual(a1[2, 0, 0], a0[0, 0, 2])
    XCTAssertEqual(a1[1, 0, 0], a0[0, 0, 1])
    XCTAssertEqual(a1[2, 1, 2], a0[1, 2, 2])
  }

  func testPermuteAndGetASubset() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(
      Tensor<Float32>(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], .CPU,
        .HWC(2, 3, 4)))
    let a1 = a0.permuted(2, 0, 1)
    let sa0 = a0[0..<2, 0..<2, 0..<2]
    let sa1 = a1[0..<2, 0..<2, 0..<2]
    XCTAssertEqual(sa1[1, 0, 0], sa0[0, 0, 1])
    XCTAssertEqual(sa1[1, 1, 0], sa0[1, 0, 1])
    XCTAssertEqual(sa1[1, 1, 1], sa0[1, 1, 1])
    XCTAssertEqual(sa1[1, 0, 1], sa0[0, 1, 1])
    XCTAssertEqual(sa1[0, 0, 1], sa0[0, 1, 0])
  }

  func testPermuteAndReshape() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(
      Tensor<Float32>(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], .CPU,
        .HWC(2, 3, 4)))
    let a1 = a0.permuted(2, 0, 1)
    let b0 = a1.copied().reshaped(.C(4))
    XCTAssertEqual(b0[0], a1[0, 0, 0])
    XCTAssertEqual(b0[1], a1[0, 0, 1])
    XCTAssertEqual(b0[2], a1[0, 0, 2])
    XCTAssertEqual(b0[3], a1[0, 1, 0])
  }

  func testReshapeWithNegativeOne() throws {
    let graph = DynamicGraph()
    let tensor = graph.variable(Tensor<Int32>([1, 2, 3, 4, 5, 6], .CPU, .NC(2, 3)))
    let reshaped = tensor.reshaped(format: .NCHW, shape: [-1])
    XCTAssertEqual([6], Array(reshaped.shape))
    XCTAssertEqual(1, reshaped.rawValue[0])
    XCTAssertEqual(6, reshaped.rawValue[5])
  }

  func testConcatZeroLengthTensor() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(.CPU, format: .NCHW, shape: [], of: Float.self)
    let a1 = dynamicGraph.variable(Tensor<Float>([1, 2, 3, 4], .CPU, .NC(2, 2)))
    let b0 = Concat(axis: 1)(inputs: a0, a1)[0].as(of: Float.self)
    XCTAssertEqual(b0[0, 0], 1)
    XCTAssertEqual(b0[0, 1], 2)
    XCTAssertEqual(b0[1, 0], 3)
    XCTAssertEqual(b0[1, 1], 4)
  }

  static let allTests = [
    ("testGEMM", testGEMM),
    ("testGEMMGrad", testGEMMGrad),
    ("testFull", testFull),
    ("testLerp", testLerp),
    ("testClamp", testClamp),
    ("testPartialAssign", testPartialAssign),
    ("testPartialAssignWithIndices", testPartialAssignWithIndices),
    ("testPartialAssignWithUnboundedRange", testPartialAssignWithUnboundedRange),
    ("testPartialAssignWithGroup", testPartialAssignWithGroup),
    ("testPartialAssignWithGroupWithIndices", testPartialAssignWithGroupWithIndices),
    ("testPartialAssignWithGroupWithUnboundedRange", testPartialAssignWithGroupWithUnboundedRange),
    ("testMin", testMin),
    ("testMax", testMax),
    ("testScale", testScale),
    ("testReLU", testReLU),
    ("testSigmoid", testSigmoid),
    ("testTanh", testTanh),
    ("testSwish", testSwish),
    ("testSoftmax", testSoftmax),
    ("testArgmax", testArgmax),
    ("testMaskedFill", testMaskedFill),
    ("testPermute", testPermute),
    ("testPermuteAndGetASubset", testPermuteAndGetASubset),
    ("testPermuteAndReshape", testPermuteAndReshape),
    ("testReshapeWithNegativeOne", testReshapeWithNegativeOne),
    ("testConcatZeroLengthTensor", testConcatZeroLengthTensor),
  ]
}
