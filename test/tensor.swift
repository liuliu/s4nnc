import Foundation
import NNC
import XCTest

final class TensorTests: XCTestCase {
  func testCreateZeroLengthTensor() throws {
    let tensor = Tensor<Float>(.CPU, format: .NHWC, shape: [])
    XCTAssertEqual([], tensor.shape)
  }

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

  func testTensorMakeContiguous() throws {
    var tensor = Tensor<Float32>(.CPU, .NC(2, 3))
    tensor[1, ...] = [1, 2, 3]
    tensor[0, ...] = [-1, -2, -3]
    let partTensor = Tensor<Float32>(from: tensor[0..<2, 1..<3])
    for tuple in zip([-2.0, -3.0], Array(partTensor[0, ...])) {
      XCTAssertEqual(tuple.0, Double(tuple.1), accuracy: 1e-5)
    }
    for tuple in zip([2.0, 3.0], Array(partTensor[1, ...])) {
      XCTAssertEqual(tuple.0, Double(tuple.1), accuracy: 1e-5)
    }
    XCTAssertFalse(tensor[0..<2, 1..<3].isContiguous)
    XCTAssertTrue(partTensor.isContiguous)
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

  func testTensorShapeAccessors() throws {
    let tensorShape: TensorShape = [1, 2, 3, 4]
    XCTAssertEqual(tensorShape[0], 1)
    XCTAssertEqual(tensorShape[1], 2)
    XCTAssertEqual(tensorShape[2], 3)
    XCTAssertEqual(tensorShape[3], 4)
    XCTAssertEqual(tensorShape[4], 0)
    var newTensorShape = tensorShape[2...]
    XCTAssertEqual(newTensorShape[0], 3)
    XCTAssertEqual(newTensorShape[1], 4)
    XCTAssertEqual(newTensorShape[2], 0)
    XCTAssertEqual(newTensorShape[3], 0)
    let array = Array(newTensorShape)
    XCTAssertEqual(array, [3, 4])
    var oldShape = [Int]()
    for i in tensorShape {
      oldShape.append(i)
    }
    XCTAssertEqual(oldShape, [1, 2, 3, 4])
    newTensorShape[1..<8] = tensorShape
    XCTAssertEqual(newTensorShape[0], 3)
    XCTAssertEqual(newTensorShape[1], 1)
    XCTAssertEqual(newTensorShape[2], 2)
    XCTAssertEqual(newTensorShape[3], 3)
    XCTAssertEqual(newTensorShape[4], 4)
    XCTAssertEqual(newTensorShape[5], 0)
  }

  func testPermute() throws {
    let a0 = Tensor<Float32>(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], .CPU,
      .HWC(2, 3, 4))
    let a1 = a0.permuted(2, 0, 1)
    XCTAssertEqual(a1[2, 0, 0], a0[0, 0, 2])
    XCTAssertEqual(a1[1, 0, 0], a0[0, 0, 1])
    XCTAssertEqual(a1[2, 1, 2], a0[1, 2, 2])
  }

  func testPermuteAndGetASubset() throws {
    let a0 = Tensor<Float32>(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], .CPU,
      .HWC(2, 3, 4))
    let a1 = a0.permuted(2, 0, 1)
    let sa0 = a0[0..<2, 0..<2, 0..<2]
    let sa1 = a1[0..<2, 0..<2, 0..<2]
    XCTAssertEqual(sa1[1, 0, 0], sa0[0, 0, 1])
    XCTAssertEqual(sa1[1, 1, 0], sa0[1, 0, 1])
    XCTAssertEqual(sa1[1, 1, 1], sa0[1, 1, 1])
    XCTAssertEqual(sa1[1, 0, 1], sa0[0, 1, 1])
    XCTAssertEqual(sa1[0, 0, 1], sa0[0, 1, 0])
  }

  func testPermuteAndReshape() throws {
    let a0 = Tensor<Float32>(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], .CPU,
      .HWC(2, 3, 4))
    let a1 = a0.permuted(2, 0, 1)
    let b0 = a1.copied().reshaped(.C(4))
    XCTAssertEqual(b0[0], a1[0, 0, 0])
    XCTAssertEqual(b0[1], a1[0, 0, 1])
    XCTAssertEqual(b0[2], a1[0, 0, 2])
    XCTAssertEqual(b0[3], a1[0, 1, 0])
  }

  func testReshapeWithNegativeOne() throws {
    let tensor = Tensor<Int32>([1, 2, 3, 4, 5, 6], .CPU, .NC(2, 3))
    let reshaped = tensor.reshaped(format: .NCHW, shape: [-1])
    XCTAssertEqual([6], Array(reshaped.shape))
    XCTAssertEqual(1, reshaped[0])
    XCTAssertEqual(6, reshaped[5])
  }

  func testSerializeTensorToData() throws {
    var tensor = Tensor<Int32>(.CPU, .NC(2, 3))
    tensor[1, 0..<3] = [1, 2, 3]
    tensor[0, 0..<3] = [-1, -2, -3]
    let data = tensor.data(using: [])
    let tensor1 = Tensor<Int32>(data: data, using: [])!
    XCTAssertEqual(-1, tensor1[0, 0])
    XCTAssertEqual(-2, tensor1[0, 1])
    XCTAssertEqual(-3, tensor[0, 2])
    XCTAssertEqual(1, tensor1[1, 0])
    XCTAssertEqual(2, tensor1[1, 1])
    XCTAssertEqual(3, tensor1[1, 2])
    var f32tensor = Tensor<Float>(.CPU, .C(1024))
    for i in 0..<1024 {
      f32tensor[i] = Float(i)
    }
    let f32Data = f32tensor.data(using: [.fpzip])
    let f32tensor1 = Tensor<Float>(data: f32Data, using: [.fpzip])!
    for i in 0..<1024 {
      XCTAssertEqual(f32tensor1[i], Float(i))
    }
  }

  static let allTests = [
    ("testCreateZeroLengthTensor", testCreateZeroLengthTensor),
    ("testGetSetPartTensor", testGetSetPartTensor),
    ("testGetSetPartTensorFromArray", testGetSetPartTensorFromArray),
    ("testGetSetUnboundedPartTensorFromArray", testGetSetUnboundedPartTensorFromArray),
    ("testTensorTypeConversion", testTensorTypeConversion),
    ("testTensorMakeContiguous", testTensorMakeContiguous),
    ("testNoneMatchTensorAssignments", testNoneMatchTensorAssignments),
    ("testNoneMatchTensorAssignmentsWithGPU", testNoneMatchTensorAssignmentsWithGPU),
    ("testTensorShapeAccessors", testTensorShapeAccessors),
    ("testPermute", testPermute),
    ("testPermuteAndGetASubset", testPermuteAndGetASubset),
    ("testPermuteAndReshape", testPermuteAndReshape),
    ("testReshapeWithNegativeOne", testReshapeWithNegativeOne),
    ("testSerializeTensorToData", testSerializeTensorToData),
  ]
}
