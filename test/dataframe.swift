import XCTest
import nnc

final class DataFrameTests: XCTestCase {

  func testBasicIteration() throws {
    let dataframe = DataFrame(from: [1, 2, 3, 4])
    let iter = dataframe["main", Int.self]
    iter.prefetch(2)
    var newArray = [Int]()
    for i in iter {
      newArray.append(i)
    }
    XCTAssertEqual(newArray, [1, 2, 3, 4])
    for i in dataframe["main"] {
      newArray.append(i as! Int)
    }
    XCTAssertEqual(newArray, [1, 2, 3, 4, 1, 2, 3, 4])
  }

  func testAddScalar() throws {
    let dataframe = DataFrame(from: [1, 2, 3, 4])
    dataframe["1"] = .from(10)
    var newArray = [[Int]]()
    for i in dataframe["main", "1"] {
      newArray.append(i as! [Int])
    }
    XCTAssertEqual(newArray, [[1, 10], [2, 10], [3, 10], [4, 10]])
  }

  func testRename() throws {
    let dataframe = DataFrame(from: [1, 2, 3, 4])
    dataframe["2"] = .from([4, 3, 2, 1])
    dataframe["rename"] = dataframe["2"]
    var newArray = [[Int]]()
    for i in dataframe["main", "rename"] {
      newArray.append(i as! [Int])
    }
    XCTAssertEqual(newArray, [[1, 4], [2, 3], [3, 2], [4, 1]])
  }

  static var allTests = [
    ("testBasicIteration", testBasicIteration),
    ("testAddScalar", testAddScalar),
    ("testRename", testRename)
  ]
}
