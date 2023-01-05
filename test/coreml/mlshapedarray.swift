import NNC
import NNCCoreMLConversion
#if canImport(CoreML)
import CoreML
import XCTest

final class MLShapedArrayTests: XCTestCase {

  func testMakeArray() throws {
    var tensor = Tensor<Float32>(.CPU, .NC(2, 3))
    tensor[0, 0] = 1
    tensor[0, 1] = 2
    tensor[0, 2] = 3
    tensor[1, 0] = 4
    tensor[1, 1] = 5
    tensor[1, 2] = 6
    let array = MLShapedArray(tensor)
    XCTAssertEqual(2.0, array[scalarAt: 0, 1])
    XCTAssertEqual(6.0, array[scalarAt: 1, 2])
  }

  func testReadArray() throws {
    let array = MLShapedArray<Float>(scalars: [1, 2, 3, 4, 5, 6], shape: [2, 3])
    let tensor = Tensor(array)
    XCTAssertEqual(1.0, tensor[0, 0])
    XCTAssertEqual(6.0, tensor[1, 2])
  }

  func testMakeArrayFromGPU() throws {
    var tensor = Tensor<Float32>(.CPU, .NC(2, 3))
    tensor[0, 0] = 1
    tensor[0, 1] = 2
    tensor[0, 2] = 3
    tensor[1, 0] = 4
    tensor[1, 1] = 5
    tensor[1, 2] = 6
    let array = MLShapedArray(tensor.toGPU(0))
    XCTAssertEqual(2.0, array[scalarAt: 0, 1])
    XCTAssertEqual(6.0, array[scalarAt: 1, 2])
  }

  static let allTests = [
    ("testMakeArray", testMakeArray),
    ("testReadArray", testReadArray),
    ("testMakeArrayFromGPU", testMakeArrayFromGPU),
  ]
}
#endif
