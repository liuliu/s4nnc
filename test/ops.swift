import NNC
import XCTest

final class OpsTests: XCTestCase {

  func testDivision() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2], .CPU, .NC(2, 1)))
    let a1 = dynamicGraph.variable(Tensor<Float32>([3.3, 4.4], .CPU, .NC(2, 1)))
    let a2 = a0 ./ a1
    XCTAssertEqual(a2.rawValue.shape, [2, 1])
    XCTAssertEqual(a2.rawValue[0, 0], 1.1 / 3.3, accuracy: 1e-5)
    XCTAssertEqual(a2.rawValue[1, 0], 2.2 / 4.4, accuracy: 1e-5)
  }

  func testScalarDivision1() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2], .CPU, .NC(2, 1)))
    let a1 = 0.5 / a0
    XCTAssertEqual(a1.rawValue.shape, [2, 1])
    XCTAssertEqual(a1.rawValue[0, 0], 0.5 / 1.1, accuracy: 1e-5)
    XCTAssertEqual(a1.rawValue[1, 0], 0.5 / 2.2, accuracy: 1e-5)
  }

  func testScalarDivision2() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2], .CPU, .NC(2, 1)))
    let a1 = a0 / 2
    XCTAssertEqual(a1.rawValue.shape, [2, 1])
    XCTAssertEqual(a1.rawValue[0, 0], 1.1 / 2, accuracy: 1e-5)
    XCTAssertEqual(a1.rawValue[1, 0], 2.2 / 2, accuracy: 1e-5)
  }

  func testReduceSum() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = a0.reduced(.sum, axis: [1])
    XCTAssertEqual(a1.rawValue.shape, [2, 1])
    XCTAssertEqual(a1.rawValue[0, 0], 1.1 + 2.2)
    XCTAssertEqual(a1.rawValue[1, 0], 3.3 + 4.4)
  }

  func testReduceMean() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = a0.reduced(.mean, axis: [1])
    XCTAssertEqual(a1.rawValue.shape, [2, 1])
    XCTAssertEqual(a1.rawValue[0, 0], (1.1 + 2.2) / 2)
    XCTAssertEqual(a1.rawValue[1, 0], (3.3 + 4.4) / 2)
  }

  func testReduceMax() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = a0.reduced(.max, axis: [0])
    XCTAssertEqual(a1.rawValue.shape, [1, 2])
    XCTAssertEqual(a1.rawValue[0, 0], 3.3)
    XCTAssertEqual(a1.rawValue[0, 1], 4.4)
  }

  func testOpAdd() throws {
    let dynamicGraph = DynamicGraph()
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = 2.2 + a0
    XCTAssertEqual(a1.rawValue.shape, [2, 2])
    XCTAssertEqual(a1.rawValue[0, 0], 1.1 + 2.2)
    XCTAssertEqual(a1.rawValue[0, 1], 2.2 + 2.2)
    XCTAssertEqual(a1.rawValue[1, 0], 3.3 + 2.2)
    XCTAssertEqual(a1.rawValue[1, 1], 4.4 + 2.2)
    let a2 = a0 + 1.1
    XCTAssertEqual(a2.rawValue.shape, [2, 2])
    XCTAssertEqual(a2.rawValue[0, 0], 1.1 + 1.1)
    XCTAssertEqual(a2.rawValue[0, 1], 2.2 + 1.1)
    XCTAssertEqual(a2.rawValue[1, 0], 3.3 + 1.1)
    XCTAssertEqual(a2.rawValue[1, 1], 4.4 + 1.1)
    let a3 = 1.1 - a0
    XCTAssertEqual(a3.rawValue.shape, [2, 2])
    XCTAssertEqual(a3.rawValue[0, 0], 1.1 - 1.1)
    XCTAssertEqual(a3.rawValue[0, 1], 1.1 - 2.2)
    XCTAssertEqual(a3.rawValue[1, 0], 1.1 - 3.3)
    XCTAssertEqual(a3.rawValue[1, 1], 1.1 - 4.4)
    let a4 = a0 - 5
    XCTAssertEqual(a4.rawValue.shape, [2, 2])
    XCTAssertEqual(a4.rawValue[0, 0], 1.1 - 5)
    XCTAssertEqual(a4.rawValue[0, 1], 2.2 - 5)
    XCTAssertEqual(a4.rawValue[1, 0], 3.3 - 5)
    XCTAssertEqual(a4.rawValue[1, 1], 4.4 - 5)
  }

  func testReduceSumModel() throws {
    let dynamicGraph = DynamicGraph()
    let input = Input()
    let model = Model([input], [input.reduced(.sum, axis: [1])])
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = DynamicGraph.Tensor<Float32>(model(inputs: a0)[0])
    XCTAssertEqual(a1.rawValue.shape, [2, 1])
    XCTAssertEqual(a1.rawValue[0, 0], 1.1 + 2.2)
    XCTAssertEqual(a1.rawValue[1, 0], 3.3 + 4.4)
  }

  func testReduceMeanModel() throws {
    let dynamicGraph = DynamicGraph()
    let input = Input()
    let model = Model([input], [input.reduced(.mean, axis: [1])])
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = DynamicGraph.Tensor<Float32>(model(inputs: a0)[0])
    XCTAssertEqual(a1.rawValue.shape, [2, 1])
    XCTAssertEqual(a1.rawValue[0, 0], (1.1 + 2.2) / 2)
    XCTAssertEqual(a1.rawValue[1, 0], (3.3 + 4.4) / 2)
  }

  func testReduceMaxModel() throws {
    let dynamicGraph = DynamicGraph()
    let input = Input()
    let model = Model([input], [input.reduced(.max, axis: [0])])
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 2.2, 3.3, 4.4], .CPU, .NC(2, 2)))
    let a1 = DynamicGraph.Tensor<Float32>(model(inputs: a0)[0])
    XCTAssertEqual(a1.rawValue.shape, [1, 2])
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
    XCTAssertEqual(a2.rawValue.shape, [2, 1])
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
    XCTAssertEqual(a2.rawValue.shape, [2, 1])
    XCTAssertEqual(a2.rawValue[0, 0], 2.2)
    XCTAssertEqual(a2.rawValue[1, 0], 4.4)
  }

  func testConcatModel() throws {
    let dynamicGraph = DynamicGraph()
    let i0 = Input()
    let i1 = Input()
    let model = Model([i0, i1], [Functional.concat(axis: 1, i0, i1)])
    let a0 = dynamicGraph.variable(Tensor<Float32>([1.1, 4.4], .CPU, .NC(2, 1)))
    let a1 = dynamicGraph.variable(Tensor<Float32>([2.2, 3.3], .CPU, .NC(2, 1)))
    let a2 = DynamicGraph.Tensor<Float32>(model(inputs: a0, a1)[0])
    XCTAssertEqual(a2.rawValue.shape, [2, 2])
    XCTAssertEqual(a2.rawValue[0, 0], 1.1)
    XCTAssertEqual(a2.rawValue[1, 0], 4.4)
    XCTAssertEqual(a2.rawValue[0, 1], 2.2)
    XCTAssertEqual(a2.rawValue[1, 1], 3.3)
  }

  static let allTests = [
    ("testDivision", testDivision),
    ("testScalarDivision1", testScalarDivision1),
    ("testScalarDivision2", testScalarDivision2),
    ("testReduceSum", testReduceSum),
    ("testReduceMean", testReduceMean),
    ("testReduceMax", testReduceMax),
    ("testOpAdd", testOpAdd),
    ("testReduceSumModel", testReduceSumModel),
    ("testReduceMeanModel", testReduceMeanModel),
    ("testReduceMaxModel", testReduceMaxModel),
    ("testMinModel", testMinModel),
    ("testMaxModel", testMaxModel),
    ("testConcatModel", testConcatModel),
  ]
}
