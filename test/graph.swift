import NNC
import XCTest

final class GraphTests: XCTestCase {

  func testGEMM() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2], .NC(2, 1)))
    let a1 = dynamicGraph.variable(Tensor<Float32>([2.2, 3.3], .NC(1, 2)))
    let a2 = a0 * a1
    XCTAssertEqual(a2.rawValue.dimensions, [2, 2])
    XCTAssertEqual(a2.rawValue[0, 0], 1.1 * 2.2)
    XCTAssertEqual(a2.rawValue[0, 1], 1.1 * 3.3)
    XCTAssertEqual(a2.rawValue[1, 0], 2.2 * 2.2)
    XCTAssertEqual(a2.rawValue[1, 1], 2.2 * 3.3)
  }

  func testGEMMGrad() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2], .NC(2, 1)))
    a0.requiresGrad = true
    let a1 = dynamicGraph.variable(Tensor<Float32>([2.2, 3.3], .NC(1, 2)))
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

  func testFill() throws {
    let dynamicGraph = DynamicGraph()
    let a0: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .NC(2, 1))
    a0.fill(10)
    XCTAssertEqual(a0.rawValue[0, 0], 10, accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[1, 0], 10, accuracy: 1e-5)
    a0.fill(-1)
    XCTAssertEqual(a0.rawValue[0, 0], -1, accuracy: 1e-5)
    XCTAssertEqual(a0.rawValue[1, 0], -1, accuracy: 1e-5)
  }

  static let allTests = [
    ("testGEMM", testGEMM),
    ("testGEMMGrad", testGEMMGrad),
    ("testFill", testFill),
  ]
}
