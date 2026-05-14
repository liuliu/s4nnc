import NNC
import XCTest
import C_nnc
import Foundation

final class StoreTests: XCTestCase {
  private func removeFiles(_ paths: String...) {
    for path in paths {
      try? FileManager.default.removeItem(atPath: path)
    }
  }

  private func trailerOffset(_ path: String) -> UInt64 {
    guard let fp = fopen(path, "rb") else { return 0 }
    defer { fclose(fp) }
    guard fseek(fp, 60, SEEK_SET) == 0 else { return 0 }
    var bytes = [UInt8](repeating: 0, count: 4)
    let readBytes = bytes.withUnsafeMutableBytes { fread($0.baseAddress, 1, 4, fp) }
    guard readBytes == 4 else { return 0 }
    return bytes.reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
  }

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

  func testWriteModelAndReadFromDifferentStore() throws {
    let graph = DynamicGraph()
    let linear0 = Dense(count: 1, noBias: true, name: "linear")
    let tv0 = graph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = linear0(inputs: tv0)[0].as(of: Float32.self)
    let linear1 = Dense(count: 1, noBias: true)
    linear1.compile(inputs: tv0)
    graph.openStore("test/model.db") { store in
      store.write("a", model: linear0)
    }
    graph.openStore("test/modelw.db") { store in
      graph.openStore("test/model.db") { storeB in
        store.read("a", model: linear1) { name, _, _, _ in
          return .continue("__a__[t-linear-0-0]", store: storeB)
        }
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
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(128))
    tensor[0] = 2.2
    tensor[1] = 1.1
    for i in 2..<128 {
      tensor[i] = 3.3
    }
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
      var tensor: Tensor<Float16> = Tensor(.CPU, .C(64))
      tensor[0] = 2.2
      tensor[1] = 1.1
      for i in 2..<64 {
        tensor[i] = 3.3
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
    var tensor: Tensor<Double> = Tensor(.CPU, .C(32))
    tensor[0] = 2.2
    tensor[1] = 1.1
    for i in 2..<32 {
      tensor[i] = 3.3
    }
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
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(128))
    tensor[0] = 2.2
    tensor[1] = 1.1
    for i in 2..<128 {
      tensor[i] = 3.3
    }
    var tensor16: Tensor<Float16> = Tensor(.CPU, .C(128))
    tensor16[0] = 2.2
    tensor16[1] = 1.1
    for i in 2..<128 {
      tensor16[i] = 3.3
    }
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

  func testWriteTensorAndReadBackWithQ8PAndExternalStore() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .C(768))
    for i in 0..<768 {
      tensor[i] = 1.1 * Float16(i)
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db", externalStore: "test/tmp.db-tensordata") { store in
      store.write("a", tensor: tensor, codec: [.externalData, .q8p])
      readout = store.read("a", codec: [.externalData, .q8p])
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, [.q8p, .externalData])
    let varf = Tensor<Float16>(readout!)
    for i in 0..<768 {
      XCTAssertEqual(varf[i], tensor[i], accuracy: 1.8)
    }
  }

  func testWriteTensorAndReadBackWithI8X() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .NC(16, 128))
    for i in 0..<16 {
      for j in 0..<128 {
        tensor[i, j] = Float16(i) * 0.25 + Float16(j - 64) * 0.125
      }
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .i8x)
      readout = store.read("a", codec: .i8x)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .i8x)
    let varf = Tensor<Float16>(readout!)
    for i in 0..<16 {
      for j in 0..<128 {
        XCTAssertEqual(varf[i, j], tensor[i, j], accuracy: 0.25)
      }
    }
  }

  func testWriteTensorAndReadBackPartialWithI8X() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .NC(16, 128))
    for i in 0..<16 {
      for j in 0..<128 {
        tensor[i, j] = Float16(i) * 0.5 + Float16(j % 17) * 0.25
      }
    }
    let varf = graph.variable(.CPU, .NC(2, 128), of: Float16.self)
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .i8x)
      store.read("a", variable: varf, codec: .i8x)
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .i8x)
    for i in 0..<2 {
      for j in 0..<128 {
        XCTAssertEqual(varf[i, j], tensor[i, j], accuracy: 0.25)
      }
    }
  }

  func testWriteTensorAndReadBackWithI8XJit() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .NC(16, 128))
    for i in 0..<16 {
      for j in 0..<128 {
        tensor[i, j] = Float16(i - 8) * 0.25 + Float16(j) * 0.0625
      }
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db") { store in
      store.write("a", tensor: tensor, codec: .i8x)
      readout = store.read("a", codec: [.i8x, .jit])
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, .i8x)
    XCTAssertNotNil(readout)
    XCTAssertEqual(Int(readout!.cTensor.pointee.info.datatype & 0xff000), CCV_QX)
    XCTAssertEqual(Int(readout!.cTensor.pointee.info.datatype & 0xf00), CCV_NNC_QX_8I_ROWWISE)
  }

  func testWriteTensorAndReadBackWithI8XFormats() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .NC(4, 128))
    for i in 0..<4 {
      for j in 0..<128 {
        tensor[i, j] = Float16(i - 2) * 0.125 + Float16(j % 29 - 14) * 0.0625
      }
    }
    var imatrix: Tensor<Float> = Tensor(.CPU, .C(128))
    for i in 0..<128 {
      imatrix[i] = 1 + Float(i % 11) * 0.125
    }
    let formats: [(String, DynamicGraph.Store.Codec, Int32)] = [
      ("q4k", .i8x(.q4k), Int32(CCV_NNC_QX_8I_ROWWISE_Q4_K)),
      ("q5k", .i8x(.q5k), Int32(CCV_NNC_QX_8I_ROWWISE_Q5_K)),
      ("q3k", .i8x(.q3k), Int32(CCV_NNC_QX_8I_ROWWISE_Q3_K)),
      ("q2k", .i8x(.q2k), Int32(CCV_NNC_QX_8I_ROWWISE_Q2_K)),
      ("iq2s", .i8x(.iq2s), Int32(CCV_NNC_QX_8I_ROWWISE_IQ2_S)),
      ("iq2xs", .i8x(.iq2xs), Int32(CCV_NNC_QX_8I_ROWWISE_IQ2_XS)),
      ("iq2xxs", .i8x(.iq2xxs), Int32(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS)),
      ("iq3s", .i8x(.iq3s), Int32(CCV_NNC_QX_8I_ROWWISE_IQ3_S)),
      ("iq3xxs", .i8x(.iq3xxs), Int32(CCV_NNC_QX_8I_ROWWISE_IQ3_XXS)),
    ]
    var readouts = Array<AnyTensor?>(repeating: nil, count: formats.count)
    var jitReadouts = Array<AnyTensor?>(repeating: nil, count: formats.count)
    var readoutCodecs = Array<DynamicGraph.Store.Codec?>(repeating: nil, count: formats.count)
    var writeError: Error? = nil
    graph.openStore("test/tmp.db") { store in
      for (index, format) in formats.enumerated() {
        do {
          try store.write(
            "i8x-format-\(format.0)", tensor: tensor, strict: true, codec: format.1,
            imatrix: imatrix)
        } catch {
          writeError = error
          return
        }
        readouts[index] = store.read("i8x-format-\(format.0)", codec: format.1)
        var jitCodec = format.1
        jitCodec.insert(.jit)
        jitReadouts[index] = store.read("i8x-format-\(format.0)", codec: jitCodec)
        readoutCodecs[index] = store.codec(for: "i8x-format-\(format.0)")
      }
    }
    XCTAssertNil(writeError)
    for (index, format) in formats.enumerated() {
      XCTAssertEqual(readoutCodecs[index]!, format.1)
      XCTAssertNotNil(readouts[index])
      let decoded = Tensor<Float16>(readouts[index]!)
      for i in 0..<4 {
        for j in 0..<128 {
          XCTAssertEqual(Float(decoded[i, j]), Float(tensor[i, j]), accuracy: 2)
        }
      }
      XCTAssertNotNil(jitReadouts[index])
      XCTAssertEqual(Int(jitReadouts[index]!.cTensor.pointee.info.datatype & 0xff000), CCV_QX)
      XCTAssertEqual(
        Int(jitReadouts[index]!.cTensor.pointee.info.datatype & 0xf00),
        CCV_NNC_QX_8I_ROWWISE_X)
      XCTAssertEqual(jitReadouts[index]!.cTensor.pointee.info.reserved, format.2)
    }
  }

  func testWriteTensorAndReadBackWithI8XFormatsAndExternalStore() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .NC(4, 128))
    for i in 0..<4 {
      for j in 0..<128 {
        tensor[i, j] = Float16(i) * 0.25 - Float16(j % 23) * 0.03125
      }
    }
    var imatrix: Tensor<Float> = Tensor(.CPU, .C(128))
    for i in 0..<128 {
      imatrix[i] = 0.5 + Float(i % 7) * 0.25
    }
    let formats: [(String, DynamicGraph.Store.Codec, Int32)] = [
      ("q4k", .i8x(.q4k), Int32(CCV_NNC_QX_8I_ROWWISE_Q4_K)),
      ("q5k", .i8x(.q5k), Int32(CCV_NNC_QX_8I_ROWWISE_Q5_K)),
      ("q3k", .i8x(.q3k), Int32(CCV_NNC_QX_8I_ROWWISE_Q3_K)),
      ("q2k", .i8x(.q2k), Int32(CCV_NNC_QX_8I_ROWWISE_Q2_K)),
      ("iq2s", .i8x(.iq2s), Int32(CCV_NNC_QX_8I_ROWWISE_IQ2_S)),
      ("iq2xs", .i8x(.iq2xs), Int32(CCV_NNC_QX_8I_ROWWISE_IQ2_XS)),
      ("iq2xxs", .i8x(.iq2xxs), Int32(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS)),
      ("iq3s", .i8x(.iq3s), Int32(CCV_NNC_QX_8I_ROWWISE_IQ3_S)),
      ("iq3xxs", .i8x(.iq3xxs), Int32(CCV_NNC_QX_8I_ROWWISE_IQ3_XXS)),
    ]
    var readouts = Array<AnyTensor?>(repeating: nil, count: formats.count)
    var jitReadouts = Array<AnyTensor?>(repeating: nil, count: formats.count)
    var readoutCodecs = Array<DynamicGraph.Store.Codec?>(repeating: nil, count: formats.count)
    var writeError: Error? = nil
    graph.openStore("test/tmp.db", externalStore: "test/tmp.db-tensordata") { store in
      for (index, format) in formats.enumerated() {
        var codec = format.1
        codec.insert(.externalData)
        do {
          try store.write(
            "i8x-external-format-\(format.0)", tensor: tensor, strict: true, codec: codec,
            imatrix: imatrix)
        } catch {
          writeError = error
          return
        }
        readouts[index] = store.read("i8x-external-format-\(format.0)", codec: codec)
        var jitCodec = codec
        jitCodec.insert(.jit)
        jitReadouts[index] = store.read("i8x-external-format-\(format.0)", codec: jitCodec)
        readoutCodecs[index] = store.codec(for: "i8x-external-format-\(format.0)")
      }
    }
    XCTAssertNil(writeError)
    for (index, format) in formats.enumerated() {
      var codec = format.1
      codec.insert(.externalData)
      XCTAssertEqual(readoutCodecs[index]!, codec)
      XCTAssertNotNil(readouts[index])
      let decoded = Tensor<Float16>(readouts[index]!)
      for i in 0..<4 {
        for j in 0..<128 {
          XCTAssertEqual(Float(decoded[i, j]), Float(tensor[i, j]), accuracy: 2)
        }
      }
      XCTAssertNotNil(jitReadouts[index])
      XCTAssertEqual(Int(jitReadouts[index]!.cTensor.pointee.info.datatype & 0xff000), CCV_QX)
      XCTAssertEqual(
        Int(jitReadouts[index]!.cTensor.pointee.info.datatype & 0xf00),
        CCV_NNC_QX_8I_ROWWISE_X)
      XCTAssertEqual(jitReadouts[index]!.cTensor.pointee.info.reserved, format.2)
    }
  }

  func testWriteTensorWithI8XFormatRejectsInvalidImatrix() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .NC(4, 128))
    for i in 0..<4 {
      for j in 0..<128 {
        tensor[i, j] = Float16(i + j) * 0.03125
      }
    }
    var imatrix: Tensor<Float> = Tensor(.CPU, .C(127))
    for i in 0..<127 {
      imatrix[i] = 1
    }
    var rejected = false
    var unexpectedError: Error? = nil
    graph.openStore("test/tmp.db") { store in
      do {
        try store.write(
          "i8x-invalid-imatrix", tensor: tensor, strict: true, codec: .i8x(.q4k),
          imatrix: imatrix)
      } catch DynamicGraph.Store.ModelWriteError.invalidI8XImatrix {
        rejected = true
      } catch {
        unexpectedError = error
      }
    }
    XCTAssertTrue(rejected)
    XCTAssertNil(unexpectedError)
  }

  func testWriteTensorAndReadBackWithI8XAndExternalStore() throws {
    let graph = DynamicGraph()
    var tensor: Tensor<Float16> = Tensor(.CPU, .NC(16, 128))
    for i in 0..<16 {
      for j in 0..<128 {
        tensor[i, j] = Float16(i) * 0.125 + Float16(j - 32) * 0.1875
      }
    }
    var readout: AnyTensor? = nil
    var readoutCodec: DynamicGraph.Store.Codec? = nil
    graph.openStore("test/tmp.db", externalStore: "test/tmp.db-tensordata") { store in
      store.write("a", tensor: tensor, codec: [.externalData, .i8x])
      readout = store.read("a", codec: [.externalData, .i8x])
      readoutCodec = store.codec(for: "a")
    }
    XCTAssertEqual(readoutCodec!, [.i8x, .externalData])
    let varf = Tensor<Float16>(readout!)
    for i in 0..<16 {
      for j in 0..<128 {
        XCTAssertEqual(varf[i, j], tensor[i, j], accuracy: 0.25)
      }
    }
  }

  func testMakeTrailerStoreAndReadBackWithExternalData() throws {
    let sqliteStore = "test/tmp_trailer.db"
    let externalStore = "test/tmp_trailer.db-tensordata"
    let trailerStore = "test/tmp_trailer_combined.db"
    let bogusExternalStore = "test/tmp_trailer_bogus"
    removeFiles(sqliteStore, externalStore, trailerStore, bogusExternalStore)

    let graph = DynamicGraph()
    var tensor: Tensor<Float32> = Tensor(.CPU, .C(4))
    tensor[0] = 1.25
    tensor[1] = -2.5
    tensor[2] = 3.75
    tensor[3] = -4.125
    graph.openStore(sqliteStore, externalStore: externalStore) { store in
      store.write("raw", tensor: tensor, codec: .externalData)
    }

    try DynamicGraph.Store.makeTrailerStore(
      sqliteStore, externalStore: externalStore, to: trailerStore)
    XCTAssertFalse(DynamicGraph.Store.isTrailerStore(sqliteStore))
    XCTAssertTrue(DynamicGraph.Store.isTrailerStore(trailerStore))
    XCTAssertEqual(trailerOffset(trailerStore) % 16_384, 0)

    var readout: AnyTensor? = nil
    var mmapReadout: AnyTensor? = nil
    graph.openStore(trailerStore, externalStore: bogusExternalStore) { store in
      readout = store.read("raw", codec: .externalData)
      mmapReadout = store.read("raw", codec: [.externalData(.mmap), .jit])
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: bogusExternalStore))
    let varf = Tensor<Float32>(readout!)
    let mmapVarf = Tensor<Float32>(mmapReadout!)
    for i in 0..<4 {
      XCTAssertEqual(varf[i], tensor[i])
      XCTAssertEqual(mmapVarf[i], tensor[i])
    }
  }

  func testTrailerStoreReadsTrailerAndWritesExternalStore() throws {
    let sqliteStore = "test/tmp_trailer_write.db"
    let externalStore = "test/tmp_trailer_write.db-tensordata"
    let trailerStore = "test/tmp_trailer_write_combined.db"
    let sidecarExternalStore = "test/tmp_trailer_write_sidecar"
    removeFiles(sqliteStore, externalStore, trailerStore, sidecarExternalStore)

    let graph = DynamicGraph()
    var original: Tensor<Float32> = Tensor(.CPU, .C(3))
    original[0] = 1
    original[1] = 2
    original[2] = 3
    graph.openStore(sqliteStore, externalStore: externalStore) { store in
      store.write("original", tensor: original, codec: .externalData)
    }
    try DynamicGraph.Store.makeTrailerStore(
      sqliteStore, externalStore: externalStore, to: trailerStore)

    var added: Tensor<Float32> = Tensor(.CPU, .C(3))
    added[0] = -1
    added[1] = -2
    added[2] = -3
    var originalReadout: AnyTensor? = nil
    var writeError: Error? = nil
    graph.openStore(trailerStore, externalStore: sidecarExternalStore) { store in
      do {
        try store.write("added", tensor: added, strict: true, codec: .externalData)
      } catch {
        writeError = error
      }
      originalReadout = store.read("original", codec: .externalData)
    }
    XCTAssertNil(writeError)
    XCTAssertTrue(FileManager.default.fileExists(atPath: sidecarExternalStore))
    let originalVarf = Tensor<Float32>(originalReadout!)
    for i in 0..<3 {
      XCTAssertEqual(originalVarf[i], original[i])
    }

    var originalReopenReadout: AnyTensor? = nil
    graph.openStore(trailerStore, externalStore: sidecarExternalStore) { store in
      originalReopenReadout = store.read("original", codec: .externalData)
    }
    XCTAssertNotNil(originalReopenReadout)
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
    ("testWriteModelAndReadFromDifferentStore", testWriteModelAndReadFromDifferentStore),
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
    (
      "testWriteTensorAndReadBackWithQ8PAndExternalStore",
      testWriteTensorAndReadBackWithQ8PAndExternalStore
    ),
    ("testWriteTensorAndReadBackWithI8X", testWriteTensorAndReadBackWithI8X),
    ("testWriteTensorAndReadBackPartialWithI8X", testWriteTensorAndReadBackPartialWithI8X),
    ("testWriteTensorAndReadBackWithI8XJit", testWriteTensorAndReadBackWithI8XJit),
    ("testWriteTensorAndReadBackWithI8XFormats", testWriteTensorAndReadBackWithI8XFormats),
    (
      "testWriteTensorAndReadBackWithI8XFormatsAndExternalStore",
      testWriteTensorAndReadBackWithI8XFormatsAndExternalStore
    ),
    (
      "testWriteTensorWithI8XFormatRejectsInvalidImatrix",
      testWriteTensorWithI8XFormatRejectsInvalidImatrix
    ),
    (
      "testWriteTensorAndReadBackWithI8XAndExternalStore",
      testWriteTensorAndReadBackWithI8XAndExternalStore
    ),
    (
      "testMakeTrailerStoreAndReadBackWithExternalData",
      testMakeTrailerStoreAndReadBackWithExternalData
    ),
    (
      "testTrailerStoreReadsTrailerAndWritesExternalStore",
      testTrailerStoreReadsTrailerAndWritesExternalStore
    ),
    ("testWriteTensorAndReadBackCodec", testWriteTensorAndReadBackCodec),
  ]
}
