import NNC
import XCTest

final class DataFrameTests: XCTestCase {

  func testBasicIteration() throws {
    let df = DataFrame(from: [1, 2, 3, 4])
    let iter = df["0", Int.self]
    iter.prefetch(2)
    var newArray = [Int]()
    for i in iter {
      newArray.append(i)
    }
    XCTAssertEqual(newArray, [1, 2, 3, 4])
    for i in df["0"]! {
      newArray.append(i as! Int)
    }
    XCTAssertEqual(newArray, [1, 2, 3, 4, 1, 2, 3, 4])
  }

  func testAddScalar() throws {
    var df = DataFrame(from: [1, 2, 3, 4])
    df["1"] = .from(10)
    var newArray = [[Int]]()
    for i in df["0", "1"] {
      newArray.append(i as! [Int])
    }
    XCTAssertEqual(newArray, [[1, 10], [2, 10], [3, 10], [4, 10]])
  }

  func testAddScalarWithSequenceIndices() throws {
    var df = DataFrame(from: [1, 2, 3, 4])
    df["1"] = .from(10)
    var newArray = [[Int]]()
    for i in df[["0", "1"]] {
      newArray.append(i as! [Int])
    }
    XCTAssertEqual(newArray, [[1, 10], [2, 10], [3, 10], [4, 10]])
  }

  func testRename() throws {
    var df = DataFrame(from: [1, 2, 3, 4])
    df["2"] = .from([4, 3, 2, 1])
    df["rename"] = df["2"]
    var newArray = [[Int]]()
    for i in df["0", "rename"] {
      newArray.append(i as! [Int])
    }
    XCTAssertEqual(newArray, [[1, 4], [2, 3], [3, 2], [4, 1]])
  }

  func testStruct() throws {
    struct MyStruct: Equatable {
      var value: Float32
      var string: String
    }

    let df = DataFrame(from: [
      MyStruct(value: 1.0, string: "1.0"), nil, MyStruct(value: 1.2, string: "1.2"),
    ])
    var newArray = [MyStruct?]()

    for i in df["0"]! {
      let s = i as? MyStruct
      newArray.append(s)
    }
    XCTAssertEqual(
      newArray, [MyStruct(value: 1.0, string: "1.0"), nil, MyStruct(value: 1.2, string: "1.2")])
  }

  func testEnum() throws {
    enum MyEnum: Equatable {
      case value(Float32)
      case string(String)
    }

    let df = DataFrame(from: [MyEnum.value(1.0), MyEnum.string("1.2")])
    var newArray = [MyEnum]()

    for i in df["0"]! {
      let s = i as! MyEnum
      newArray.append(s)
    }
    XCTAssertEqual(newArray, [MyEnum.value(1.0), MyEnum.string("1.2")])
  }

  func testMap() throws {
    var df = DataFrame(from: [1, 2, 3, 4, 5])
    df["+1"] = df["0"]!.map { (i: Int) -> Int in
      i + 1
    }
    var newArray = [Int]()
    for i in df["+1", Int.self] {
      newArray.append(i)
    }
    XCTAssertEqual(newArray, [2, 3, 4, 5, 6])
  }

  func testMultiMap() throws {
    var df = DataFrame(from: [1, 2, 3, 4, 5])

    df["+1"] = df["0"]!.map { (i: Int) -> Int in
      i + 1
    }

    df["2"] = .from([1, 1, 1, 1, 1])

    df["++"] = df["0", "+1"].map { (i: Int, j: Int) -> Int in
      i + j
    }
    df["3"] = .from([1, 1, 1, 1, 1])
    df["4"] = .from([1, 1, 1, 1, 1])
    df["5"] = .from([1, 1, 1, 1, 1])
    df["6"] = .from([1, 1, 1, 1, 1])
    df["9"] = .from([1, 1, 1, 1, 1])
    df["10"] = .from([1, 1, 1, 1, 1])
    df["z"] = df["0", "+1", "++", "2", "3", "4", "5", "6", "9", "10"].map {
      (c0: Int, c1: Int, c2: Int, c3: Int, c4: Int, c5: Int, c6: Int, c7: Int, c8: Int, c9: Int)
        -> Int in
      return c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
    }

    var newArray = [Int]()
    for i in df["z", Int.self] {
      newArray.append(i)
    }
    XCTAssertEqual(newArray, [13, 17, 21, 25, 29])
  }

  func testFromTensor() throws {
    var df = DataFrame(from: [1, 2])
    var tensor = Tensor<Float32>(.CPU, .C(1))
    tensor[0] = 1.2
    df["image"] = .from(tensor)
    var newArray = [Float32]()
    for i in df["image", Tensor<Float32>.self] {
      newArray.append(i[0])
    }
    XCTAssertEqual(newArray, [1.2, 1.2])
  }

  func testFromTensorArray() throws {
    var df = DataFrame(from: [1, 2])
    var tensor0 = Tensor<Float32>(.CPU, .C(1))
    tensor0[0] = 1.2
    var tensor1 = Tensor<Float32>(.CPU, .C(1))
    tensor1[0] = 2.2
    df["image"] = .from([tensor0, tensor1])
    var newArray = [Float32]()
    for i in df["image", Tensor<Float32>.self] {
      newArray.append(i[0])
    }
    XCTAssertEqual(newArray, [1.2, 2.2])
  }

  func testReadCSV() throws {
    let df = DataFrame(fromCSV: "test/scaled_data.csv")!
    var newArray = [String]()
    for i in df["V0001", String.self] {
      newArray.append(i)
    }
    let truthArray: [String] = [
      "0.0239389181223648",
      "0.128521802419456",
      "0.157117194502824",
      "0.255931328978371",
      "0.160808946156887",
      "0.129455089453963",
      "0.0944587345381377",
      "0.0619675688122819",
      "0.125194372114953",
      "0.114054752304399",
      "0.0331194635169463",
      "0.0796469310314676",
      "0.0971290536167178",
      "0.0576220663889997",
      "0.134768058679109",
      "0.0655047474218985",
      "0.189655655632105",
      "-0.019471558855501",
      "-0.0161545694780594",
      "0.131137326326646",
      "0.160254357051892",
      "0.0506895539858365",
      "-0.01033458071889",
      "-0.12347980825132",
    ]
    XCTAssertEqual(newArray, truthArray)
  }

  func testBatching() throws {
    var tensor0 = Tensor<Float32>(.CPU, .C(1))
    tensor0[0] = 1.1
    var tensor1 = Tensor<Float32>(.CPU, .C(1))
    tensor1[0] = 2.2
    let df = DataFrame(from: [tensor0, tensor1], name: "main")
    let batched = df["main"]!.combine(size: 2)
    for tensor in batched["main", Tensor<Float32>.self] {
      XCTAssertEqual(1.1, tensor[0, 0])
      XCTAssertEqual(2.2, tensor[1, 0])
    }
  }

  func testBatchingRepeating() throws {
    var tensor00 = Tensor<Float32>(.CPU, .C(1))
    tensor00[0] = 1.1
    var tensor01 = Tensor<Float32>(.CPU, .C(1))
    tensor01[0] = 2.2
    var tensor02 = Tensor<Float32>(.CPU, .C(1))
    tensor02[0] = 3.2
    var tensor03 = Tensor<Float32>(.CPU, .C(1))
    tensor03[0] = 4.2
    var df = DataFrame(from: [tensor00, tensor01, tensor02, tensor03], name: "main")
    var tensor10 = Tensor<Float32>(.CPU, .C(1))
    tensor10[0] = 1.2
    var tensor11 = Tensor<Float32>(.CPU, .C(1))
    tensor11[0] = 2.3
    var tensor12 = Tensor<Float32>(.CPU, .C(1))
    tensor12[0] = 3.3
    var tensor13 = Tensor<Float32>(.CPU, .C(1))
    tensor13[0] = 4.3
    df["1"] = .from([tensor10, tensor11, tensor12, tensor13])
    let batched = df["main", "1"].combine(size: 2, repeating: 2)
    for tensor in batched["main_0", Tensor<Float32>.self] {
      XCTAssertEqual(1.1, tensor[0, 0])
      XCTAssertEqual(2.2, tensor[1, 0])
    }
    for tensor in batched["main_1", Tensor<Float32>.self] {
      XCTAssertEqual(3.2, tensor[0, 0])
      XCTAssertEqual(4.2, tensor[1, 0])
    }
    for tensor in batched["1_0", Tensor<Float32>.self] {
      XCTAssertEqual(1.2, tensor[0, 0])
      XCTAssertEqual(2.3, tensor[1, 0])
    }
    for tensor in batched["1_1", Tensor<Float32>.self] {
      XCTAssertEqual(3.3, tensor[0, 0])
      XCTAssertEqual(4.3, tensor[1, 0])
    }
  }

  func testTypedBatching() throws {
    var tensor0 = Tensor<Float32>(.CPU, .C(1))
    tensor0[0] = 1.1
    var tensor1 = Tensor<Float32>(.CPU, .C(1))
    tensor1[0] = 2.2
    let df = DataFrame(from: [tensor0, tensor1], name: "main")
    let batched = df["main", Tensor<Float32>.self].combine(size: 2)
    for tensor in batched["main", Tensor<Float32>.self] {
      XCTAssertEqual(1.1, tensor[0, 0])
      XCTAssertEqual(2.2, tensor[1, 0])
    }
  }

  func testMultiBatching() throws {
    var tensor0 = Tensor<Float32>(.CPU, .C(1))
    tensor0[0] = 1.1
    var tensor1 = Tensor<Float32>(.CPU, .C(1))
    tensor1[0] = 2.2
    var df = DataFrame(from: [tensor0, tensor1], name: "main")
    df["1"] = df["main", Tensor<Float32>.self].map { input -> Tensor<Float32> in
      var output = Tensor<Float32>(.CPU, .C(1))
      output[0] = input[0] + 1
      return output
    }
    let batched = df["main", "1"].combine(size: 2)
    for tensor in batched["main", Tensor<Float32>.self] {
      XCTAssertEqual(1.1, tensor[0, 0])
      XCTAssertEqual(2.2, tensor[1, 0])
    }
    for tensor in batched["1", Tensor<Float32>.self] {
      XCTAssertEqual(1.1 + 1, tensor[0, 0])
      XCTAssertEqual(2.2 + 1, tensor[1, 0])
    }
  }

  func testOneHot() throws {
    var df = DataFrame(from: [0, 1, 2], name: "main")
    df["oneHot"] = df["main"]!.toOneHot(Float32.self, count: 3)
    var i: Int = 0
    for tensor in df["oneHot", Tensor<Float32>.self] {
      XCTAssertEqual(1, tensor[i])
      for j in 0..<3 where j != i {
        XCTAssertEqual(0, tensor[j])
      }
      i += 1
    }
  }

  func testStructAndIntWrap() throws {
    struct Holder {
      var str: String
      var val: Int
    }
    var df = DataFrame(
      from: [Holder(str: "abc", val: 1), Holder(str: "happy", val: 2)], name: "main")
    df["c"] = df["main", Holder.self].map(\.val)
    df["oneHot"] = df["c", Int.self].toOneHot(Float32.self, count: 3)
    var i: Int = 1
    for tensor in df["oneHot", Tensor<Float32>.self] {
      XCTAssertEqual(1, tensor[i])
      for j in 0..<3 where j != i {
        XCTAssertEqual(0, tensor[j])
      }
      i += 1
    }
  }

  func testToGPU() throws {
    guard DeviceKind.GPUs.count > 0 else { return }
    var tensor0 = Tensor<Float32>(.CPU, .C(1))
    tensor0[0] = 1.1
    var tensor1 = Tensor<Float32>(.CPU, .C(1))
    tensor1[0] = 2.2
    var df = DataFrame(from: [tensor0, tensor1], name: "main")
    df["main_gpu"] = df["main"]!.toGPU()
    var i: Int = 0
    for tensor in df["main_gpu", Tensor<Float32>.self] {
      let cpu = tensor.toCPU()
      switch i {
      case 0:
        XCTAssertEqual(1.1, cpu[0])
      case 1:
        XCTAssertEqual(2.2, cpu[0])
      default:
        break
      }
      i += 1
    }
  }

  func testToManyGPU() throws {
    guard DeviceKind.GPUs.count > 0 else { return }
    var tensor0 = Tensor<Float32>(.CPU, .C(1))
    tensor0[0] = 1.1
    var tensor1 = Tensor<Float32>(.CPU, .C(1))
    tensor1[0] = 2.2
    var df = DataFrame(from: [tensor0, tensor1], name: "main")
    df["1"] = df["main", Tensor<Float32>.self].map { input -> Tensor<Float32> in
      var output = Tensor<Float32>(.CPU, .C(1))
      output[0] = input[0] + 1
      return output
    }
    let holder = df["main", "1"].toGPU()
    df["main_gpu"] = holder["main"]
    df["1_gpu"] = holder["1"]
    var i: Int = 0
    for tensor in df["main_gpu", Tensor<Float32>.self] {
      let cpu = tensor.toCPU()
      switch i {
      case 0:
        XCTAssertEqual(1.1, cpu[0])
      case 1:
        XCTAssertEqual(2.2, cpu[0])
      default:
        break
      }
      i += 1
    }
    i = 0
    for tensor in df["1_gpu", Tensor<Float32>.self] {
      let cpu = tensor.toCPU()
      switch i {
      case 0:
        XCTAssertEqual(1.1 + 1, cpu[0])
      case 1:
        XCTAssertEqual(2.2 + 1, cpu[0])
      default:
        break
      }
      i += 1
    }
  }

  func testOneSquared() throws {
    var tensor0 = Tensor<Int32>(.CPU, .NC(2, 1))
    tensor0[0, 0] = 1
    tensor0[1, 0] = 2
    var df = DataFrame(from: [tensor0], name: "main")
    df["squared"] = df["main"]!.toOneSquared(maxLength: 3)
    df["squared_max"] = df["main"]!.toOneSquared(maxLength: 3, variableLength: false)
    for tensor in df["squared", Tensor<Int32>.self] {
      XCTAssertEqual(1, tensor[0, 0, 0])
      XCTAssertEqual(0, tensor[0, 0, 1])
      XCTAssertEqual(0, tensor[0, 1, 0])
      XCTAssertEqual(0, tensor[0, 1, 1])
      XCTAssertEqual(1, tensor[1, 0, 0])
      XCTAssertEqual(1, tensor[1, 0, 1])
      XCTAssertEqual(1, tensor[1, 1, 0])
      XCTAssertEqual(1, tensor[1, 1, 1])
    }
    for tensor in df["squared_max", Tensor<Int32>.self] {
      XCTAssertEqual(1, tensor[0, 0, 0])
      XCTAssertEqual(0, tensor[0, 0, 1])
      XCTAssertEqual(0, tensor[0, 0, 2])
      XCTAssertEqual(0, tensor[0, 1, 0])
      XCTAssertEqual(0, tensor[0, 1, 1])
      XCTAssertEqual(0, tensor[0, 1, 2])
      XCTAssertEqual(0, tensor[0, 2, 0])
      XCTAssertEqual(0, tensor[0, 2, 1])
      XCTAssertEqual(0, tensor[0, 2, 2])
      XCTAssertEqual(1, tensor[1, 0, 0])
      XCTAssertEqual(1, tensor[1, 0, 1])
      XCTAssertEqual(0, tensor[1, 0, 2])
      XCTAssertEqual(1, tensor[1, 1, 0])
      XCTAssertEqual(1, tensor[1, 1, 1])
      XCTAssertEqual(0, tensor[1, 1, 2])
      XCTAssertEqual(0, tensor[1, 2, 0])
      XCTAssertEqual(0, tensor[1, 2, 1])
      XCTAssertEqual(0, tensor[1, 2, 2])
    }
  }

  func testTruncate() throws {
    var tensor0 = Tensor<Float32>(.CPU, .NC(1, 3))
    tensor0[0, 0] = 1.1
    tensor0[0, 1] = 2.1
    tensor0[0, 2] = 1.2
    var tensor1 = Tensor<Int32>(.CPU, .C(1))
    tensor1[0] = 2
    var df = DataFrame(from: [tensor0], name: "main")
    df["len"] = .from([tensor1])
    df["truncated"] = df["main"]!.toTruncate(df["len"]!)
    for tensor in df["truncated", Tensor<Float32>.self] {
      XCTAssertEqual(1.1, tensor[0, 0])
      XCTAssertEqual(2.1, tensor[0, 1])
      XCTAssertEqual(1, tensor.shape[0])
      XCTAssertEqual(2, tensor.shape[1])
    }
  }

  func testSimpleSampler() throws {
    let df = DataFrame(from: [1, 2, 3, 4, 5, 6, 7, 8], name: "main")
    let sampled = df["main"]!.sample(size: 3) { (input: [Int]) -> String in
      return "\(input[0])"
    }
    var result = [String]()
    for val in sampled["main", String.self] {
      result.append(val)
    }
    XCTAssertEqual(result, ["1", "4", "7"])
  }

  func testTensorSampler() throws {
    var tensor0 = Tensor<Float32>(.CPU, .C(1))
    tensor0[0] = 1.1
    var tensor1 = Tensor<Float32>(.CPU, .C(1))
    tensor1[0] = 2.2
    var tensor2 = Tensor<Float32>(.CPU, .C(1))
    tensor2[0] = 2.3
    var tensor3 = Tensor<Float32>(.CPU, .C(1))
    tensor3[0] = 3.2
    let df = DataFrame(from: [tensor0, tensor1, tensor2, tensor3], name: "main")
    let sampled = df["main", Tensor<Float32>.self].sample(size: 2) { input -> Tensor<Float32> in
      var tensor = Tensor<Float32>(.CPU, .C(2))
      tensor[0] = input[1][0]
      return tensor
    }
    var result = [Float]()
    for val in sampled["main", Tensor<Float32>.self] {
      result.append(val[0])
    }
    XCTAssertEqual(result, [2.2, 3.2])
  }

  func testSimpleRepeatingSampler() throws {
    let df = DataFrame(from: [1, 2, 3, 4, 5, 6, 7, 8], name: "main")
    let sampled = df["main"]!.sample(size: 3, repeating: 2) { (input: [Int]) -> String in
      return "\(input[0])"
    }
    var result = [String]()
    for val in sampled["main_0", String.self] {
      result.append(val)
    }
    XCTAssertEqual(result, ["1", "7"])
    result.removeAll()
    for val in sampled["main_1", String.self] {
      result.append(val)
    }
    XCTAssertEqual(result, ["4", "8"])
  }

  func testTensorRepeatingSampler() throws {
    var tensor0 = Tensor<Float32>(.CPU, .C(1))
    tensor0[0] = 1.1
    var tensor1 = Tensor<Float32>(.CPU, .C(1))
    tensor1[0] = 2.2
    var tensor2 = Tensor<Float32>(.CPU, .C(1))
    tensor2[0] = 2.3
    var tensor3 = Tensor<Float32>(.CPU, .C(1))
    tensor3[0] = 3.2
    var tensor4 = Tensor<Float32>(.CPU, .C(1))
    tensor4[0] = 3.3
    let df = DataFrame(from: [tensor0, tensor1, tensor2, tensor3, tensor4], name: "main")
    let sampled = df["main", Tensor<Float32>.self].sample(size: 2, repeating: 2) {
      input -> Tensor<Float32> in
      var tensor = Tensor<Float32>(.CPU, .C(1))
      tensor[0] = input.last![0]
      return tensor
    }
    var result = [Float]()
    for val in sampled["main_0", Tensor<Float32>.self] {
      result.append(val[0])
    }
    XCTAssertEqual(result, [2.2, 3.3])
    result.removeAll()
    for val in sampled["main_1", Tensor<Float32>.self] {
      result.append(val[0])
    }
    XCTAssertEqual(result, [3.2, 3.3])
  }

  func testStructSampler() throws {
    struct MyStruct: Equatable {
      var value: Float32
      var string: String
    }

    let df = DataFrame(from: [
      MyStruct(value: 1.0, string: "1.0"), nil, MyStruct(value: 1.2, string: "1.2"),
      MyStruct(value: 1.3, string: "1.3"),
    ])
    let sampled = df["0", MyStruct?.self].sample(size: 2) { input -> Float32 in
      return input[0]!.value
    }
    var newArray = [Float]()
    for i in sampled["0"]! {
      let s = i as! Float
      newArray.append(s)
    }
    XCTAssertEqual(newArray, [1.0, 1.2])
  }

  func testManySampler() throws {
    var df = DataFrame(from: [Float(1.1), Float(2.2), Float(1.2), Float(2.1), Float(2.3)])
    df["1"] = df["0", Float.self].map { input -> Float in
      return input + 1.2
    }
    let sampled = df["0", "1"].sample(size: 2) {
      (f: [(Float, Float)]) -> (Tensor<Float32>, Float) in
      let sum = f[0].0 + f[0].1
      var tensor = Tensor<Float32>(.CPU, .C(1))
      tensor[0] = sum
      return (tensor, sum)
    }
    var newArray = [Float]()
    for i in sampled["0", Tensor<Float32>.self] {
      newArray.append(i[0])
    }
    XCTAssertEqual(
      newArray,
      [Float(1.1) + Float(1.1) + 1.2, Float(1.2) + Float(1.2) + 1.2, Float(2.3) + Float(2.3) + 1.2])
    newArray.removeAll()
    for i in sampled["1", Float.self] {
      newArray.append(i)
    }
    XCTAssertEqual(
      newArray,
      [Float(1.1) + Float(1.1) + 1.2, Float(1.2) + Float(1.2) + 1.2, Float(2.3) + Float(2.3) + 1.2])
  }

  func testManyRepeatingSampler() throws {
    var df = DataFrame(from: [Float(1.1), Float(2.2), Float(1.2), Float(2.1), Float(2.3)])
    df["1"] = df["0", Float.self].map { input -> Float in
      return input + 1.2
    }
    df["2"] = df["1", Float.self].map { input -> Float in
      return input + 0.2
    }
    let sampled = df["0", "1", "2"].sample(size: 2, repeating: 2) {
      (f: [(Float, Float, Float)]) -> (Tensor<Float32>, Float, Float) in
      let sum = f[0].0 + f[0].1
      var tensor = Tensor<Float32>(.CPU, .C(1))
      tensor[0] = sum
      return (tensor, sum, f[0].1 + f[0].2)
    }
    var newArray = [Float]()
    for i in sampled["0_0", Tensor<Float32>.self] {
      newArray.append(i[0])
    }
    XCTAssertEqual(newArray, [Float(1.1) + Float(1.1) + 1.2, Float(2.3) + Float(2.3) + 1.2])
    newArray.removeAll()
    for i in sampled["2_1", Float.self] {
      newArray.append(i)
    }
    XCTAssertEqual(
      newArray,
      [Float(1.2) + Float(1.2) + 1.2 + 1.2 + 0.2, Float(2.3) + Float(2.3) + 1.2 + 1.2 + 0.2])
  }

  func testMany4Sampler() throws {
    var df = DataFrame(from: [Float(1.1), Float(2.2), Float(1.2), Float(2.1), Float(2.3)])
    df["1"] = df["0", Float.self].map { input -> Float in
      return input + 1.2
    }
    df["2"] = df["1", Float.self].map { input -> Float in
      return input + 0.2
    }
    df["3"] = df["2", Float.self].map { input -> Float in
      return input + 1.0
    }
    let sampled = df["0", "1", "2", "3"].sample(size: 2) {
      (f: [(Float, Float, Float, Float)]) -> (Tensor<Float32>, Float, Float, Float) in
      let sum = f[0].0 + f[0].1
      var tensor = Tensor<Float32>(.CPU, .C(1))
      tensor[0] = sum
      return (tensor, sum, f[0].1 + f[0].2, f[0].2 + f[0].3)
    }
    var newArray = [Float]()
    for i in sampled["3", Float.self] {
      newArray.append(i)
    }
    let truth: [Float] = [
      1.1 + 1.2 + 0.2 + 1.1 + 1.2 + 0.2 + 1.0, 1.2 + 1.2 + 1.2 + 1.2 + 0.2 + 0.2 + 1.0,
      2.3 + 2.3 + 1.2 + 1.2 + 0.2 + 0.2 + 1.0,
    ]
    XCTAssertEqual(newArray[0], truth[0], accuracy: 1e-4)
    XCTAssertEqual(newArray[1], truth[1], accuracy: 1e-4)
    XCTAssertEqual(newArray[2], truth[2], accuracy: 1e-4)
  }

  static let allTests = [
    ("testBasicIteration", testBasicIteration),
    ("testAddScalar", testAddScalar),
    ("testAddScalarWithSequenceIndices", testAddScalarWithSequenceIndices),
    ("testRename", testRename),
    ("testStruct", testStruct),
    ("testEnum", testEnum),
    ("testMap", testMap),
    ("testMultiMap", testMultiMap),
    ("testFromTensor", testFromTensor),
    ("testFromTensorArray", testFromTensorArray),
    ("testReadCSV", testReadCSV),
    ("testBatching", testBatching),
    ("testTypedBatching", testTypedBatching),
    ("testMultiBatching", testMultiBatching),
    ("testBatchingRepeating", testBatchingRepeating),
    ("testOneHot", testOneHot),
    ("testStructAndIntWrap", testStructAndIntWrap),
    ("testToGPU", testToGPU),
    ("testToManyGPU", testToManyGPU),
    ("testOneSquared", testOneSquared),
    ("testTruncate", testTruncate),
    ("testSimpleSampler", testSimpleSampler),
    ("testTensorSampler", testTensorSampler),
    ("testSimpleRepeatingSampler", testSimpleRepeatingSampler),
    ("testTensorRepeatingSampler", testTensorRepeatingSampler),
    ("testStructSampler", testStructSampler),
    ("testManySampler", testManySampler),
    ("testManyRepeatingSampler", testManyRepeatingSampler),
    ("testMany4Sampler", testMany4Sampler),
  ]
}
