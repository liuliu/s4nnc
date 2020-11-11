import XCTest
import nnc

final class LossTests: XCTestCase {

  func testLoss() throws {
    let dynamicGraph = DynamicGraph()
    let tv0 = dynamicGraph.variable(Tensor<Float32>([0.5, 0.5, 0.2, 0.8], .NC(2, 2)))
    let tv1 = dynamicGraph.variable(Tensor<Float32>([0, 1, 1, 0], .NC(2, 2)))
    let loss = SoftmaxCrossEntropyLoss()
    let tv2 = loss(tv0, target: tv1)
    XCTAssertEqual([2, 1], tv2[0].dimensions)
    XCTAssertEqual([2, 2], tv2[1].dimensions)
  }

  func testTargetLoss() throws {
    let dynamicGraph = DynamicGraph()
    let tv0 = dynamicGraph.variable(Tensor<Float32>([0.5, 0.5, 0.2, 0.8], .NC(2, 2)))
    let tv1 = dynamicGraph.variable(Tensor<Int32>([0, 1], .NC(2, 1)))
    let loss = SoftmaxCrossEntropyLoss()
    let tv2 = loss(tv0, target: tv1)
    XCTAssertEqual([2, 1], tv2[0].dimensions)
    XCTAssertEqual([2, 2], tv2[1].dimensions)
  }

  static let allTests = [
    ("testLoss", testLoss),
    ("testTargetLoss", testTargetLoss)
  ]
}
