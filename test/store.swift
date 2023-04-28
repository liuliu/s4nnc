import NNC
import XCTest

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
    let group = DynamicGraph.Group(variable1, variable2)
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

  func testWriteTensorsAndRetrieveKeys() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(2))
    tensor[0] = 2.2
    tensor[1] = 1.1
    var keys: [String]? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor)
      store.write("b", tensor: tensor)
      keys = store.keys
    }
    XCTAssertEqual(keys![0], "a")
    XCTAssertEqual(keys![1], "b")
  }

  func testWriteTensorReadBackAndDelete() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(2))
    tensor[0] = 2.2
    tensor[1] = 1.1
    var readout1: AnyTensor? = nil
    var readout2: AnyTensor? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor)
      readout1 = store.read("a")
      store.remove("a")
      readout2 = store.read("a")
    }
    let varf = Tensor<Float32>(readout1!)
    XCTAssertEqual(varf[0], 2.2)
    XCTAssertEqual(varf[1], 1.1)
    XCTAssertNil(readout2)
  }

  func testWriteModelAndReadWithDifferentName() throws {
    let graph = DynamicGraph()
    let linear0 = Dense(count: 1, noBias: true, name: "linear")
    let tv0 = graph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = linear0(inputs: tv0)[0].as(of: Float32.self)
    let linear1 = Dense(count: 1, noBias: true)
    linear1.compile(inputs: tv0)
    graph.openStore("test/model.db") { store in
      store.write("a", model: linear0)
      store.read("a", model: linear1) { name, _, _, _ in
        return .continue("__a__[t-linear-0-0]")
      }
    }
    let tv2 = linear1(inputs: tv0)[0].as(of: Float32.self)
    XCTAssertEqual(tv1[0], tv2[0])
  }

  func testWriteModelAndLoadFromNothing() throws {
    let graph = DynamicGraph()
    let linear0 = Dense(count: 1, noBias: true, name: "linear")
    let tv0 = graph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let _ = linear0(inputs: tv0)[0].as(of: Float32.self)
    let linear1 = Dense(count: 1, noBias: true)
    linear1.compile(inputs: tv0)
    graph.openStore("test/model.db") { store in
      store.write("a", model: linear0)
      store.read("a", model: linear1) { name, _, format, shape in
        var a = Tensor<Float32>(.CPU, format: format, shape: shape)
        a[0, 0] = 2
        return .final(a)
      }
    }
    let tv2 = linear1(inputs: tv0)[0].as(of: Float32.self)
    XCTAssertEqual(tv2[0], 2.2)
  }

  func testWriteTensorAndReadBackWithFPZIP() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(2))
    tensor[0] = 2.2
    tensor[1] = 1.1
    var readout: AnyTensor? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .fpzip)
      readout = store.read("a", codec: .fpzip)
    }
    let varf = Tensor<Float32>(readout!)
    XCTAssertEqual(varf[0], 2.2)
    XCTAssertEqual(varf[1], 1.1)
  }

  func testWriteTensorAndReadBackWithFPZIPFloat16() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(2))
    tensor[0] = 2.2
    tensor[1] = 1.1
    var readout: AnyTensor? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .fpzip)
      readout = store.read("a", codec: .fpzip)
    }
    let varf = Tensor<Float16>(readout!)
    XCTAssertEqual(varf[0], tensor[0])
    XCTAssertEqual(varf[1], tensor[1])
  }

  func testWriteTensorAndReadBackWithLargerFPZIPFloat16() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(128))
    for i in 0..<128 {
      tensor[i] = 1.1 * Float16(i)
    }
    var readout: AnyTensor? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .fpzip)
      readout = store.read("a", codec: .fpzip)
    }
    let varf = Tensor<Float16>(readout!)
    for i in 0..<128 {
      XCTAssertEqual(varf[i], tensor[i])
    }
  }

  func testWriteTensorAndReadBackWithFPZIPDouble() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Double> = Tensor(.CPU, .C(2))
    tensor[0] = 2.2
    tensor[1] = 1.1
    var readout: AnyTensor? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .fpzip)
      readout = store.read("a", codec: .fpzip)
    }
    let varf = Tensor<Double>(readout!)
    XCTAssertEqual(varf[0], 2.2)
    XCTAssertEqual(varf[1], 1.1)
  }

  func testWriteTensorAndReadBackWithZIP() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(2))
    tensor[0] = 2.2
    tensor[1] = 1.1
    var readout: AnyTensor? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .zip)
      readout = store.read("a", codec: .zip)
    }
    let varf = Tensor<Float32>(readout!)
    XCTAssertEqual(varf[0], 2.2)
    XCTAssertEqual(varf[1], 1.1)
  }

  func testWriteTensorAndReadBackPartialWithZIP() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Int32> = Tensor(.CPU, .C(2048))
    for i in 0..<2048 {
      tensor[i] = Int32(i)
    }
    let varf = graph.variable(.CPU, .C(64), of: Int32.self)
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .zip)
      store.read("a", variable: varf, codec: .zip)
    }
    for i in 0..<64 {
      XCTAssertEqual(varf[i], Int32(i))
    }
  }

  static let allTests = [
    ("testReadNonexistTensor", testReadNonexistTensor),
    ("testReadExistTensorWithShape", testReadExistTensorWithShape),
    ("testReadExistTensorWithoutShape", testReadExistTensorWithoutShape),
    ("testReadExistRawTensor", testReadExistRawTensor),
    ("testReadExistTensorGroupWithoutShape", testReadExistTensorGroupWithoutShape),
    ("testWriteTensorAndReadBack", testWriteTensorAndReadBack),
    ("testWriteTensorConstantAndReadBack", testWriteTensorConstantAndReadBack),
    ("testWriteTensorsAndRetrieveKeys", testWriteTensorsAndRetrieveKeys),
    ("testWriteTensorReadBackAndDelete", testWriteTensorReadBackAndDelete),
    ("testWriteModelAndReadWithDifferentName", testWriteModelAndReadWithDifferentName),
    ("testWriteModelAndLoadFromNothing", testWriteModelAndLoadFromNothing),
    ("testWriteTensorAndReadBackWithFPZIP", testWriteTensorAndReadBackWithFPZIP),
    ("testWriteTensorAndReadBackWithFPZIPFloat16", testWriteTensorAndReadBackWithFPZIPFloat16),
    (
      "testWriteTensorAndReadBackWithLargerFPZIPFloat16",
      testWriteTensorAndReadBackWithLargerFPZIPFloat16
    ),
    ("testWriteTensorAndReadBackWithFPZIPDouble", testWriteTensorAndReadBackWithFPZIPDouble),
    ("testWriteTensorAndReadBackWithZIP", testWriteTensorAndReadBackWithZIP),
    ("testWriteTensorAndReadBackPartialWithZIP", testWriteTensorAndReadBackPartialWithZIP),
  ]
}
