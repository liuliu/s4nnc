import NNC
import XCTest

final class OpsTests: XCTestCase {

  func testReduceSum() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = a0.reduced(.sum, axis: [1])
    XCTAssertEqual(a1.rawValue.dimensions, [2, 1])
    XCTAssertEqual(a1.rawValue[0, 0], 1.1 + 2.2)
    XCTAssertEqual(a1.rawValue[1, 0], 3.3 + 4.4)
  }

  func testReduceMax() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = a0.reduced(.max, axis: [0])
    XCTAssertEqual(a1.rawValue.dimensions, [1, 2])
    XCTAssertEqual(a1.rawValue[0, 0], 3.3)
    XCTAssertEqual(a1.rawValue[0, 1], 4.4)
  }

  func testReduceSumModel() throws {
    let dynamicGraph = DynamicGraph()
    let input = Input()
    let model = Model([input], [input.reduced(.sum, axis: [1])])
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = DynamicGraph.Tensor<Float32>(model(inputs: a0)[0])
    XCTAssertEqual(a1.rawValue.dimensions, [2, 1])
    XCTAssertEqual(a1.rawValue[0, 0], 1.1 + 2.2)
    XCTAssertEqual(a1.rawValue[1, 0], 3.3 + 4.4)
  }

  func testReduceMaxModel() throws {
    let dynamicGraph = DynamicGraph()
    let input = Input()
    let model = Model([input], [input.reduced(.max, axis: [0])])
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = DynamicGraph.Tensor<Float32>(model(inputs: a0)[0])
    XCTAssertEqual(a1.rawValue.dimensions, [1, 2])
    XCTAssertEqual(a1.rawValue[0, 0], 3.3)
    XCTAssertEqual(a1.rawValue[0, 1], 4.4)
  }

  func testMinModel() throws {
    let dynamicGraph = DynamicGraph()
    let i0 = Input()
    let i1 = Input()
    let model = Model([i0, i1], [Functional.min(i0, i1)])
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 4.4], .CPU, .NC(2, 1)))
    let a1 = dynamicGraph.variable(Tensor<Float32>([2.2, 3.3], .CPU, .NC(2, 1)))
    let a2 = DynamicGraph.Tensor<Float32>(model(inputs: a0, a1)[0])
    XCTAssertEqual(a2.rawValue.dimensions, [2, 1])
    XCTAssertEqual(a2.rawValue[0, 0], 1.1)
    XCTAssertEqual(a2.rawValue[1, 0], 3.3)
  }

  func testMaxModel() throws {
    let dynamicGraph = DynamicGraph()
    let i0 = Input()
    let i1 = Input()
    let model = Model([i0, i1], [Functional.max(i0, i1)])
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 4.4], .CPU, .NC(2, 1)))
    let a1 = dynamicGraph.variable(Tensor<Float32>([2.2, 3.3], .CPU, .NC(2, 1)))
    let a2 = DynamicGraph.Tensor<Float32>(model(inputs: a0, a1)[0])
    XCTAssertEqual(a2.rawValue.dimensions, [2, 1])
    XCTAssertEqual(a2.rawValue[0, 0], 2.2)
    XCTAssertEqual(a2.rawValue[1, 0], 4.4)
  }

  static let allTests = [
    ("testReduceSum", testReduceSum),
    ("testReduceMax", testReduceMax),
    ("testReduceSumModel", testReduceSumModel),
    ("testReduceMaxModel", testReduceMaxModel),
    ("testMinModel", testMinModel),
    ("testMaxModel", testMaxModel),
  ]
}
