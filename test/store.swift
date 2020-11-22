import XCTest
import NNC

final class StoreTests: XCTestCase {

  func testReadNonexistTensor() throws {
    let graph = DynamicGraph()
    let variable = graph.variable()
    var result = true
    graph.openStore("test/nonexist.db") { store in
      result = store.read("a", variable: variable)
    }
    XCTAssertFalse(result)
  }

  func testReadExistTensorWithShape() throws {
    let graph = DynamicGraph()
    let variable: DynamicGraph.Tensor<Float32> = graph.variable(.CPU, .C(2))
    variable[0] = 0
    variable[1] = 0
    graph.openStore("test/some_variables.db") { store in
      store.read("b", variable: variable)
    }
    XCTAssertEqual(variable[0], 1.1)
    XCTAssertEqual(variable[1], 2.2)
  }

  func testReadExistTensorWithoutShape() throws {
    let graph = DynamicGraph()
    let variable = graph.variable()
    graph.openStore("test/some_variables.db") { store in
      store.read("b", variable: variable)
    }
    let varf = DynamicGraph.Tensor<Float32>(variable)
    XCTAssertEqual(varf[0], 1.1)
    XCTAssertEqual(varf[1], 2.2)
  }

  func testReadExistRawTensor() throws {
    let graph = DynamicGraph()
    var tensor: AnyTensor? = nil
    graph.openStore("test/some_variables.db") { store in
      tensor = store.read("b")
    }
    let varf = Tensor<Float32>(tensor!)
    XCTAssertEqual(varf[0], 1.1)
    XCTAssertEqual(varf[1], 2.2)
  }

  func testReadExistTensorGroupWithoutShape() throws {
    let graph = DynamicGraph()
    let variable1 = graph.variable()
    let variable2 = graph.variable()
    let group = Group(variable1, variable2)
    graph.openStore("test/some_variables.db") { store in
      store.read("c", variable: group)
    }
    let varf1 = DynamicGraph.Tensor<Float32>(group[0])
    let varf2 = DynamicGraph.Tensor<Float32>(group[1])
    XCTAssertEqual(varf1[0], 1.1)
    XCTAssertEqual(varf1[1], 2.2)
    XCTAssertEqual(varf2[0], 3.3)
    XCTAssertEqual(varf2[1], 4.4)
  }

  func testWriteTensorAndReadBack() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(2))
    tensor[0] = 2.2
    tensor[1] = 1.1
    var readout: AnyTensor? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor)
      readout = store.read("a")
    }
    let varf = Tensor<Float32>(readout!)
    XCTAssertEqual(varf[0], 2.2)
    XCTAssertEqual(varf[1], 1.1)
  }

  func testWriteTensorConstantAndReadBack() throws {
    let graph = DynamicGraph()
    let constant: DynamicGraph.Tensor<Float32> = graph.constant(.CPU, .C(2))
    constant[0] = 3.3
    constant[1] = 4.4
    let variable = graph.variable()
    graph.openStore("test/tmp.db") { store in
      store.write("b", variable: constant)
    }
    graph.openStore("test/tmp.db") { store in
      store.read("b", variable: variable)
    }
    let varf = DynamicGraph.Tensor<Float32>(variable)
    XCTAssertEqual(3.3, varf[0])
    XCTAssertEqual(4.4, varf[1])
  }

  static let allTests = [
    ("testReadNonexistTensor", testReadNonexistTensor),
    ("testReadExistTensorWithShape", testReadExistTensorWithShape),
    ("testReadExistTensorWithoutShape", testReadExistTensorWithoutShape),
    ("testReadExistRawTensor", testReadExistRawTensor),
    ("testReadExistTensorGroupWithoutShape", testReadExistTensorGroupWithoutShape),
    ("testWriteTensorAndReadBack", testWriteTensorAndReadBack),
    ("testWriteTensorConstantAndReadBack", testWriteTensorConstantAndReadBack)
  ]
}
