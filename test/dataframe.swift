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

  static var allTests = [
    ("testBasicIteration", testBasicIteration),
    ("testAddScalar", testAddScalar),
    ("testRename", testRename),
    ("testStruct", testStruct),
    ("testEnum", testEnum)
  ]
}
