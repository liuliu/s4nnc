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
      let b = store.read(like: "b")!
      XCTAssertEqual(b.shape[0], 2)
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

  func testWriteModelWriteDifferentNameAndRead() throws {
    let graph = DynamicGraph()
    let linear0 = Dense(count: 1, noBias: true, name: "linear")
    let tv0 = graph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = linear0(inputs: tv0)[0].as(of: Float32.self)
    let linear1 = Dense(count: 1, noBias: true)
    linear1.compile(inputs: tv0)
    graph.openStore("test/modelw.db") { store in
      store.write("a", model: linear0) { name, _ in
        return .continue("__a__[t-0-0]")
      }
      store.read("a", model: linear1)
    }
    let tv2 = linear1(inputs: tv0)[0].as(of: Float32.self)
    XCTAssertEqual(tv1[0], tv2[0])
  }

  func testWriteTensorAndReadBackWithFPZIP() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(2))
    tensor[0] = 2.2
    tensor[1] = 1.1
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .fpzip)
      readout = store.read("a", codec: .fpzip)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .fpzip)
    let varf = Tensor<Float32>(readout!)
    XCTAssertEqual(varf[0], 2.2)
    XCTAssertEqual(varf[1], 1.1)
  }

  #if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
    func testWriteTensorAndReadBackWithFPZIPFloat16() throws {
      let graph = DynamicGraph()
      var tensor: Tensor<Float16> = Tensor(.CPU, .C(2))
      tensor[0] = 2.2
      tensor[1] = 1.1
      var readout: AnyTensor? = nil
      var readoutCodec: DynamicGraph.Store.Codec? = nil
      graph.openStore("test/tmp.db") { store in
        store.write("a", tensor: tensor, codec: .fpzip)
        readout = store.read("a", codec: .fpzip)
        readoutCodec = store.codec(for: "a")
      }
      XCTAssertEqual(readoutCodec!, .fpzip)
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
      var readoutCodec: DynamicGraph.Store.Codec? = nil
      graph.openStore("test/tmp.db") { store in
        store.write("a", tensor: tensor, codec: .fpzip)
        readout = store.read("a", codec: .fpzip)
        readoutCodec = store.codec(for: "a")
      }
      XCTAssertEqual(readoutCodec!, .fpzip)
      let varf = Tensor<Float16>(readout!)
      for i in 0..<128 {
        XCTAssertEqual(varf[i], tensor[i])
      }
    }
  #else
    func testWriteTensorAndReadBackWithFPZIPFloat16() throws {
    }

    func testWriteTensorAndReadBackWithLargerFPZIPFloat16() throws {
    }
  #endif

  func testWriteTensorAndReadBackWithFPZIPDouble() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Double> = Tensor(.CPU, .C(2))
    tensor[0] = 2.2
    tensor[1] = 1.1
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .fpzip)
      readout = store.read("a", codec: .fpzip)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .fpzip)
    let varf = Tensor<Double>(readout!)
    XCTAssertEqual(varf[0], 2.2)
    XCTAssertEqual(varf[1], 1.1)
  }

  func testWriteTensorAndReadBackWithEZM7() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(128))
    for i in 0..<128 {
      tensor[i] = 1.1 * Float16(i)
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .ezm7)
      readout = store.read("a", codec: .ezm7)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .ezm7)
    let varf = Tensor<Float16>(readout!)
    for i in 0..<128 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 1.0)
    }
  }

  func testWriteTensorAndReadBackPartialWithEZM7() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(2048))
    for i in 0..<2048 {
      tensor[i] = 1.1 * Float16(i)
    }
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    let varf = graph.variable(.CPU, .C(64), of: Float16.self)
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .ezm7)
      store.read("a", variable: varf, codec: .ezm7)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .ezm7)
    for i in 0..<64 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 1.0)
    }
  }

  func testWriteTensorAndReadBackWithZIP() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(128))
    for i in 0..<128 {
      tensor[i] = 1.1 * Float(i)
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .zip)
      readout = store.read("a", codec: .zip)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .zip)
    let varf = Tensor<Float32>(readout!)
    for i in 0..<128 {
      XCTAssertEqual(varf[i], tensor[i])
    }
  }

  func testWriteTensorAndReadBackPartialWithZIP() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Int32> = Tensor(.CPU, .C(2048))
    for i in 0..<2048 {
      tensor[i] = Int32(i)
    }
    let varf = graph.variable(.CPU, .C(64), of: Int32.self)
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .zip)
      store.read("a", variable: varf, codec: .zip)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .zip)
    for i in 0..<64 {
      XCTAssertEqual(varf[i], Int32(i))
    }
  }

  func testWriteTensorAndReadBackWithQ4P() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(128))
    for i in 0..<128 {
      tensor[i] = 1.1 * Float16(i)
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .q4p)
      readout = store.read("a", codec: .q4p)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .q4p)
    let varf = Tensor<Float16>(readout!)
    for i in 0..<128 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 3.9)
    }
  }

  func testWriteTensorAndReadBackPartialWithQ4P() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(2048))
    for i in 0..<2048 {
      tensor[i] = 1.1 * Float16(i % 16)
    }
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    let varf = graph.variable(.CPU, .C(64), of: Float16.self)
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .q4p)
      store.read("a", variable: varf, codec: .q4p)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .q4p)
    for i in 0..<64 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 1e-6)
    }
  }

  func testWriteTensorAndReadBackWithQ5P() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(128))
    for i in 0..<128 {
      tensor[i] = 1.1 * Float16(i)
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .q5p)
      readout = store.read("a", codec: .q5p)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .q5p)
    let varf = Tensor<Float16>(readout!)
    for i in 0..<128 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 1.9)
    }
  }

  func testWriteTensorAndReadBackPartialWithQ5P() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(2048))
    for i in 0..<2048 {
      tensor[i] = 1.1 * Float16(i % 32)
    }
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    let varf = graph.variable(.CPU, .C(64), of: Float16.self)
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .q5p)
      store.read("a", variable: varf, codec: .q5p)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .q5p)
    for i in 0..<64 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 1e-6)
    }
  }

  func testWriteTensorAndReadBackWithQ6P() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(128))
    for i in 0..<128 {
      tensor[i] = 1.1 * Float16(i)
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .q6p)
      readout = store.read("a", codec: .q6p)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .q6p)
    let varf = Tensor<Float16>(readout!)
    for i in 0..<128 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 0.8)
    }
  }

  func testWriteTensorAndReadBackPartialWithQ6P() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(2048))
    for i in 0..<2048 {
      tensor[i] = 1.1 * Float16(i % 64)
    }
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    let varf = graph.variable(.CPU, .C(64), of: Float16.self)
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .q6p)
      store.read("a", variable: varf, codec: .q6p)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .q6p)
    for i in 0..<64 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 1e-6)
    }
  }

  func testWriteTensorAndReadBackWithQ7P() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(768))
    for i in 0..<768 {
      tensor[i] = 1.1 * Float16(i)
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .q7p)
      readout = store.read("a", codec: .q7p)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .q7p)
    let varf = Tensor<Float16>(readout!)
    for i in 0..<768 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 3.6)
    }
  }

  func testWriteTensorAndReadBackPartialWithQ7P() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(24_048))
    for i in 0..<24_048 {
      tensor[i] = 1.1 * Float16(i % 128)
    }
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    let varf = graph.variable(.CPU, .C(64), of: Float16.self)
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .q7p)
      store.read("a", variable: varf, codec: .q7p)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .q7p)
    for i in 0..<64 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 1e-6)
    }
  }

  func testWriteTensorAndReadBackWithQ8P() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(768))
    for i in 0..<768 {
      tensor[i] = 1.1 * Float16(i)
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .q8p)
      readout = store.read("a", codec: .q8p)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .q8p)
    let varf = Tensor<Float16>(readout!)
    for i in 0..<768 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 1.8)
    }
  }

  func testWriteTensorAndReadBackPartialWithQ8P() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(24_048))
    for i in 0..<24_048 {
      tensor[i] = 1.1 * Float16(i % 256)
    }
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    let varf = graph.variable(.CPU, .C(64), of: Float16.self)
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .q8p)
      store.read("a", variable: varf, codec: .q8p)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .q8p)
    for i in 0..<64 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 1e-6)
    }
  }

  func testWriteTensorAndReadBackCodec() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(2))
    tensor[0] = 2.2
    tensor[1] = 1.1
    var tensor16: Tensor<Float16> = Tensor(.CPU, .C(2))
    tensor16[0] = 2.2
    tensor16[1] = 1.1
    var intTensor: Tensor<Int32> = Tensor(.CPU, .C(2048))
    for i in 0..<2048 {
      intTensor[i] = Int32(i)
    }
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    var readoutCodecNil: DynamicGraph.Store.Codec? = nil
    var readoutCodecB: DynamicGraph.Store.Codec? = nil
    var readoutCodecC: DynamicGraph.Store.Codec? = nil
    var readoutCodecD: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmpcodec.db") { store in
      store.write("a", tensor: tensor, codec: .fpzip)
      readoutCodec = store.codec(for: "a")
      readoutCodecNil = store.codec(for: "z")
      store.write("b", tensor: intTensor, codec: .zip)
      readoutCodecB = store.codec(for: "b")
      store.write("c", tensor: intTensor)
      readoutCodecC = store.codec(for: "c")
      store.write("d", tensor: tensor16, codec: .ezm7)
      readoutCodecD = store.codec(for: "d")
    }
    XCTAssertEqual(readoutCodec!, .fpzip)
    XCTAssertNil(readoutCodecNil)
    XCTAssertEqual(readoutCodecB!, .zip)
    XCTAssertEqual(readoutCodecC!, [])
    XCTAssertEqual(readoutCodecD!, .ezm7)
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
    ("testWriteModelWriteDifferentNameAndRead", testWriteModelWriteDifferentNameAndRead),
    ("testWriteTensorAndReadBackWithFPZIP", testWriteTensorAndReadBackWithFPZIP),
    ("testWriteTensorAndReadBackWithFPZIPFloat16", testWriteTensorAndReadBackWithFPZIPFloat16),
    (
      "testWriteTensorAndReadBackWithLargerFPZIPFloat16",
      testWriteTensorAndReadBackWithLargerFPZIPFloat16
    ),
    ("testWriteTensorAndReadBackWithFPZIPDouble", testWriteTensorAndReadBackWithFPZIPDouble),
    ("testWriteTensorAndReadBackWithZIP", testWriteTensorAndReadBackWithZIP),
    ("testWriteTensorAndReadBackPartialWithZIP", testWriteTensorAndReadBackPartialWithZIP),
    ("testWriteTensorAndReadBackWithEZM7", testWriteTensorAndReadBackWithEZM7),
    ("testWriteTensorAndReadBackPartialWithEZM7", testWriteTensorAndReadBackPartialWithEZM7),
    ("testWriteTensorAndReadBackWithQ4P", testWriteTensorAndReadBackWithQ4P),
    ("testWriteTensorAndReadBackPartialWithQ4P", testWriteTensorAndReadBackPartialWithQ4P),
    ("testWriteTensorAndReadBackWithQ5P", testWriteTensorAndReadBackWithQ5P),
    ("testWriteTensorAndReadBackPartialWithQ5P", testWriteTensorAndReadBackPartialWithQ5P),
    ("testWriteTensorAndReadBackWithQ6P", testWriteTensorAndReadBackWithQ6P),
    ("testWriteTensorAndReadBackPartialWithQ6P", testWriteTensorAndReadBackPartialWithQ6P),
    ("testWriteTensorAndReadBackWithQ7P", testWriteTensorAndReadBackWithQ7P),
    ("testWriteTensorAndReadBackPartialWithQ7P", testWriteTensorAndReadBackPartialWithQ7P),
    ("testWriteTensorAndReadBackWithQ8P", testWriteTensorAndReadBackWithQ8P),
    ("testWriteTensorAndReadBackPartialWithQ8P", testWriteTensorAndReadBackPartialWithQ8P),
    ("testWriteTensorAndReadBackCodec", testWriteTensorAndReadBackCodec),
  ]
}
