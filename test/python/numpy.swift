import NNC
import NNCPythonConversion
import PythonKit
import XCTest

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
    let tensor = try Tensor<Float64>(numpy: array)
    XCTAssertEqual(1.0, tensor[0, 0])
    XCTAssertEqual(1.0, tensor[1, 2])
  }

  func testReadNumpyArrayTransposed() throws {
    let torch = Python.import("torch")
    let array = torch.randn(768, 768).type(torch.float).numpy()
    let np = Python.import("numpy")
    let nparray = np.transpose(array)
    let tensor = try Tensor<Float>(numpy: nparray)
    for i in 0..<768 {
      for j in 0..<768 {
        XCTAssertEqual(Float(nparray[i, j])!, tensor[i, j])
      }
    }
  }

  static let allTests = [
    ("testMakeNumpyArray", testMakeNumpyArray),
    ("testReadNumpyArray", testReadNumpyArray),
    ("testReadNumpyArrayTransposed", testReadNumpyArrayTransposed),
  ]
}
