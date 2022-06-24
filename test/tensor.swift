import Foundation
import NNC
import XCTest

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
    var outT = Tensor<Int32>([0, 0, 0, 0, 0, 0], .CPU, .NC(3, 2))
    outT[1..<3, 0..<1] = subT
    XCTAssertEqual(0, outT[0, 0])
    XCTAssertEqual(5, outT[1, 0])
    XCTAssertEqual(7, outT[2, 0])
    XCTAssertEqual(0, outT[0, 1])
    XCTAssertEqual(0, outT[1, 1])
    XCTAssertEqual(0, outT[2, 1])
  }

  func testGetSetPartTensorFromArray() throws {
    var tensor = Tensor<Int32>(.CPU, .NC(2, 3))
    tensor[1, 0..<3] = [1, 2, 3]
    tensor[0, 0..<3] = [-1, -2, -3]
    XCTAssertEqual(-1, tensor[0, 0])
    XCTAssertEqual(-2, tensor[0, 1])
    XCTAssertEqual(-3, tensor[0, 2])
    XCTAssertEqual(1, tensor[1, 0])
    XCTAssertEqual(2, tensor[1, 1])
    XCTAssertEqual(3, tensor[1, 2])
    let p0 = tensor[0, 1..<3]
    XCTAssertEqual(-2, p0[0, 0])
    XCTAssertEqual(-3, p0[0, 1])
    let p1 = tensor[1, 0..<2]
    XCTAssertEqual(1, p1[0, 0])
    XCTAssertEqual(2, p1[0, 1])
  }

  func testGetSetUnboundedPartTensorFromArray() throws {
    var tensor = Tensor<Int32>(.CPU, .NC(2, 3))
    tensor[1, ...] = [1, 2, 3]
    tensor[0, ...] = [-1, -2, -3]
    XCTAssertEqual(-1, tensor[0, 0])
    XCTAssertEqual(-2, tensor[0, 1])
    XCTAssertEqual(-3, tensor[0, 2])
    XCTAssertEqual(1, tensor[1, 0])
    XCTAssertEqual(2, tensor[1, 1])
    XCTAssertEqual(3, tensor[1, 2])
    let p0 = tensor[0, 1..<3]
    XCTAssertEqual(-2, p0[0, 0])
    XCTAssertEqual(-3, p0[0, 1])
    let p1 = tensor[1, 0..<2]
    XCTAssertEqual(1, p1[0, 0])
    XCTAssertEqual(2, p1[0, 1])
    let p2 = tensor[0, ...]
    XCTAssertEqual(-1, p2[0, 0])
    XCTAssertEqual(-2, p2[0, 1])
    XCTAssertEqual(-3, p2[0, 2])
    let p3 = tensor[1, ...]
    XCTAssertEqual(1, p3[0, 0])
    XCTAssertEqual(2, p3[0, 1])
    XCTAssertEqual(3, p3[0, 2])
  }

  func testTensorTypeConversion() throws {
    var tensor = Tensor<Float32>(.CPU, .NC(2, 3))
    tensor[1, ...] = [1, 2, 3]
    tensor[0, ...] = [-1, -2, -3]
    let tensor64 = Tensor<Float64>(from: tensor)
    for tuple in zip([-1.0, -2.0, -3.0], Array(tensor64[0, ...])) {
      XCTAssertEqual(tuple.0, tuple.1, accuracy: 1e-5)
    }
    for tuple in zip([1.0, 2.0, 3.0], Array(tensor64[1, ...])) {
      XCTAssertEqual(tuple.0, tuple.1, accuracy: 1e-5)
    }
  }

  func testNoneMatchTensorAssignments() throws {
    var tensor = Tensor<Int32>(.CPU, .NC(2, 3))
    var source = Tensor<Int32>(.CPU, .C(3))
    source[0] = 0
    source[1] = 1
    source[2] = 2
    tensor[0, ...] = source
    tensor[1, ...] = source
    XCTAssertEqual([0, 1, 2], Array(tensor[0, ...]))
    XCTAssertEqual([0, 1, 2], Array(tensor[1, ...]))
  }

  func testNoneMatchTensorAssignmentsWithGPU() throws {
    guard DeviceKind.GPUs.count > 0 else { return }
    var tensor = Tensor<Int32>(.GPU(0), .NC(2, 3))
    var source = Tensor<Int32>(.CPU, .C(3))
    source[0] = 2
    source[1] = 1
    source[2] = 0
    let gpuSource = source.toGPU(0)
    tensor[0, ...] = gpuSource
    tensor[1, ...] = gpuSource
    let cpuTensor = tensor.toCPU()
    XCTAssertEqual([2, 1, 0], Array(cpuTensor[0, ...]))
    XCTAssertEqual([2, 1, 0], Array(cpuTensor[1, ...]))
  }

  static let allTests = [
    ("testGetSetPartTensor", testGetSetPartTensor),
    ("testGetSetPartTensorFromArray", testGetSetPartTensorFromArray),
    ("testGetSetUnboundedPartTensorFromArray", testGetSetUnboundedPartTensorFromArray),
    ("testTensorTypeConversion", testTensorTypeConversion),
    ("testNoneMatchTensorAssignments", testNoneMatchTensorAssignments),
    ("testNoneMatchTensorAssignmentsWithGPU", testNoneMatchTensorAssignmentsWithGPU),
  ]
}
