import XCTest
import Foundation
import NNC

final class TensorTests: XCTestCase {

  func testGetSetPartTensor() throws {
    var tensor = Tensor<Int32>(.CPU, .NC(5, 2))
    tensor[0, 0] = 0
    tensor[0, 1] = 1
    tensor[1, 0] = 2
    tensor[1, 1] = 3
    tensor[2, 0] = 4
    tensor[2, 1] = 5
    tensor[3, 0] = 6
    tensor[3, 1] = 7
    tensor[4, 0] = 8
    tensor[4, 1] = 9
    let subT = tensor[2..<4, 1..<2]
    XCTAssertEqual(5, subT[0, 0])
    XCTAssertEqual(7, subT[1, 0])
    var outT = Tensor<Int32>([0, 0, 0, 0, 0, 0], .NC(3, 2))
    outT[1..<3, 0..<1] = subT
    XCTAssertEqual(0, outT[0, 0])
    XCTAssertEqual(5, outT[1, 0])
    XCTAssertEqual(7, outT[2, 0])
    XCTAssertEqual(0, outT[0, 1])
    XCTAssertEqual(0, outT[1, 1])
    XCTAssertEqual(0, outT[2, 1])
  }

  static let allTests = [
    ("testGetSetPartTensor", testGetSetPartTensor),
  ]
}
