import NNC
import XCTest

final class GraphTests: XCTestCase {

  func testGEMM() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2], .CPU, .NC(2, 1)))
    let a1 = dynamicGraph.variable(Tensor<Float32>([2.2, 3.3], .CPU, .NC(1, 2)))
    let a2 = a0 * a1
    XCTAssertEqual(a2.rawValue.dimensions, [2, 2])
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
    let a0: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 1))
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
    let a1: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 1))
    a1[0, 0] = -1
    a1[1, 0] = -2
    a0.clamp(3...6)
    let a2 = a1.clamped((-1.1)...)
    XCTAssertEqual(a0.rawValue[0, 0], 6, accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[1, 0], 3, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[0, 0], -1, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 0], -1.1, accuracy: 1e-5)
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
    let a2: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 2))
    a2[0..<3, 0..<1] = a0[0..<3, 0..<1]
    a2[0..<3, 1..<2] = a1[0..<3, 0..<1]
    XCTAssertEqual(a2.rawValue[0, 0], 10, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 0], 2, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[2, 0], 5, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[0, 1], -1, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 1], -2, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[2, 1], -3, accuracy: 1e-5)
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
    let a0 = Group(a00, a01)
    let a10: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 1))
    a10[0, 0] = -1
    a10[1, 0] = -2
    a10[2, 0] = -3
    let a11: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 1))
    a11[0, 0] = 1
    a11[1, 0] = 2
    a11[2, 0] = 3
    let a1 = Group(a10, a11)
    let a20: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 2))
    let a21: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(3, 2))
    var a2 = Group(a20, a21)
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

  static let allTests = [
    ("testGEMM", testGEMM),
    ("testGEMMGrad", testGEMMGrad),
    ("testFull", testFull),
    ("testLerp", testLerp),
    ("testClamp", testClamp),
    ("testPartialAssign", testPartialAssign),
    ("testPartialAssignWithGroup", testPartialAssignWithGroup),
    ("testMin", testMin),
    ("testMax", testMax),
  ]
}
