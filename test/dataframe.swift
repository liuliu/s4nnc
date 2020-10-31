import XCTest
import nnc

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
    let df = DataFrame(from: [1, 2, 3, 4])
    df["1"] = .from(10)
    var newArray = [[Int]]()
    for i in df["0", "1"] {
      newArray.append(i as! [Int])
    }
    XCTAssertEqual(newArray, [[1, 10], [2, 10], [3, 10], [4, 10]])
  }

  func testAddScalarWithSequenceIndices() throws {
    let df = DataFrame(from: [1, 2, 3, 4])
    df["1"] = .from(10)
    var newArray = [[Int]]()
    for i in df[["0", "1"]] {
      newArray.append(i as! [Int])
    }
    XCTAssertEqual(newArray, [[1, 10], [2, 10], [3, 10], [4, 10]])
  }

  func testRename() throws {
    let df = DataFrame(from: [1, 2, 3, 4])
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

    let df = DataFrame(from: [MyStruct(value: 1.0, string: "1.0"), nil, MyStruct(value: 1.2, string: "1.2")])
    var newArray = [MyStruct?]()

    for i in df["0"]! {
      let s = i as? MyStruct
      newArray.append(s)
    }
    XCTAssertEqual(newArray, [MyStruct(value: 1.0, string: "1.0"), nil, MyStruct(value: 1.2, string: "1.2")])
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
    let df = DataFrame(from: [1, 2, 3, 4, 5])
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
    let df = DataFrame(from: [1, 2, 3, 4, 5])

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
    df["z"] = df["0", "+1", "++", "2", "3", "4", "5", "6", "9", "10"].map { (c0: Int, c1: Int, c2: Int, c3: Int, c4: Int, c5: Int, c6: Int, c7: Int, c8: Int, c9: Int) -> Int in
      return c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
    }

    var newArray = [Int]()
    for i in df["z", Int.self] {
      newArray.append(i)
    }
    XCTAssertEqual(newArray, [13, 17, 21, 25, 29])
  }

  func testFromTensor() throws {
    let df = DataFrame(from: [1, 2])
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
    let df = DataFrame(from: [1, 2])
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
      "-0.12347980825132"
    ]
    XCTAssertEqual(newArray, truthArray)
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
    ("testReadCSV", testReadCSV)
  ]
}
