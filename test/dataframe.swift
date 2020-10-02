import XCTest
import nnc

final class DataFrameTests: XCTestCase {

  func testBasicIteration() throws {
    let df = DataFrame(from: [1, 2, 3, 4])
    let iter = df["main", Int.self]
    iter.prefetch(2)
    var newArray = [Int]()
    for i in iter {
      newArray.append(i)
    }
    XCTAssertEqual(newArray, [1, 2, 3, 4])
    for i in df["main"] {
      newArray.append(i as! Int)
    }
    XCTAssertEqual(newArray, [1, 2, 3, 4, 1, 2, 3, 4])
  }

  func testAddScalar() throws {
    let df = DataFrame(from: [1, 2, 3, 4])
    df["1"] = .from(10)
    var newArray = [[Int]]()
    for i in df["main", "1"] {
      newArray.append(i as! [Int])
    }
    XCTAssertEqual(newArray, [[1, 10], [2, 10], [3, 10], [4, 10]])
  }

  func testRename() throws {
    let df = DataFrame(from: [1, 2, 3, 4])
    df["2"] = .from([4, 3, 2, 1])
    df["rename"] = df["2"]
    var newArray = [[Int]]()
    for i in df["main", "rename"] {
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

    for i in df["main"] {
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

    for i in df["main"] {
      let s = i as! MyEnum
      newArray.append(s)
    }
    XCTAssertEqual(newArray, [MyEnum.value(1.0), MyEnum.string("1.2")])
  }

  func testMap() throws {
    let df = DataFrame(from: [1, 2, 3, 4, 5])
    df["+1"] = df["main"].map { (i: Int) -> Int in
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

    df["+1"] = df["main"].map { (i: Int) -> Int in
      i + 1
    }

    df["2"] = .from([1, 1, 1, 1, 1])

    df["++"] = df["main", "+1"].map { (i: Int, j: Int) -> Int in
      i + j
    }
    df["3"] = .from([1, 1, 1, 1, 1])
    df["4"] = .from([1, 1, 1, 1, 1])
    df["5"] = .from([1, 1, 1, 1, 1])
    df["6"] = .from([1, 1, 1, 1, 1])
    df["9"] = .from([1, 1, 1, 1, 1])
    df["10"] = .from([1, 1, 1, 1, 1])
    df["z"] = df["main", "+1", "++", "2", "3", "4", "5", "6", "9", "10"].map { (c0: Int, c1: Int, c2: Int, c3: Int, c4: Int, c5: Int, c6: Int, c7: Int, c8: Int, c9: Int) -> Int in
      return c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
    }

    var newArray = [Int]()
    for i in df["z", Int.self] {
      newArray.append(i)
    }
    XCTAssertEqual(newArray, [13, 17, 21, 25, 29])
  }

  static var allTests = [
    ("testBasicIteration", testBasicIteration),
    ("testAddScalar", testAddScalar),
    ("testRename", testRename),
    ("testStruct", testStruct),
    ("testEnum", testEnum),
    ("testMap", testMap),
    ("testMultiMap", testMultiMap)
  ]
}
