import XCTest
import NNC
import NNCPythonConversion
import PythonKit

final class NumpyTests: XCTestCase {

  func testMakeNumpyArray() throws {
    var tensor = Tensor<Float32>(.CPU, .NC(2, 3))
    tensor[0, 0] = 1
    tensor[0, 1] = 2
    tensor[0, 2] = 3
    tensor[1, 0] = 4
    tensor[1, 1] = 5
    tensor[1, 2] = 6
    let array = tensor.makeNumpyArray()
    XCTAssertEqual(2.0, array[0, 1])
    XCTAssertEqual(6.0, array[1, 2])
  }

  func testReadNumpyArray() throws {
    let np = Python.import("numpy")
    let array = np.ones(PythonObject(tupleOf: 2, 3))
    let tensor = Tensor<Float64>(numpy: array)!
    XCTAssertEqual(1.0, tensor[0, 0])
    XCTAssertEqual(1.0, tensor[1, 2])
  }

  static let allTests = [
    ("testMakeNumpyArray", testMakeNumpyArray),
    ("testReadNumpyArray", testReadNumpyArray)
  ]
}
