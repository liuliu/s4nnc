import C_ccv
import C_fpzip
import C_nnc
import C_zlib
import Dispatch
import SQLite3

// Quantize to 4-bit.
private let q4pEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard let data = data, let dimensions = dimensions, var encoded = encoded,
      let encodedSize = encodedSize, dimensionCount > 0
    else { return 0 }
    var elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      elementSize = MemoryLayout<Double>.size
    case Int32(CCV_32F):
      elementSize = MemoryLayout<Float>.size
    case Int32(CCV_16F):
      elementSize = MemoryLayout<UInt16>.size
    default:
      return 0
    }
    var numberOfElements = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      numberOfElements *= Int(dimensions[i])
    }
    let numberOfBlocks = (numberOfElements + 1023) / 1024
    guard
      (numberOfElements + 1) / 2 + numberOfBlocks * 16 * elementSize + MemoryLayout<UInt32>.size
        < encodedSize[0]
    else { return 0 }
    encoded.storeBytes(of: UInt32(1024), as: UInt32.self)
    encoded += MemoryLayout<UInt32>.size
    DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
      let indices = UnsafeMutablePointer<Int32>.allocate(capacity: min(1024, numberOfElements))
      let centroids = UnsafeMutablePointer<Double>.allocate(capacity: 16)
      let nI = min(1024, numberOfElements - blockIdx * 1024)
      var input = ccv_dense_matrix(
        1, Int32(nI), dataType | Int32(CCV_C1),
        UnsafeMutableRawPointer(mutating: data + 1024 * blockIdx * elementSize), 0)
      ccv_kmeans1d(&input, 16, indices, centroids)
      let encodedBlock = encoded + (16 * elementSize + 512) * blockIdx
      switch dataType {
      case Int32(CCV_64F):
        // Write centroids directly to the output.
        memcpy(encodedBlock, centroids, elementSize * 16)
      case Int32(CCV_32F):
        let f32 = encodedBlock.assumingMemoryBound(to: Float32.self)
        for i in 0..<16 {
          f32[i] = Float(centroids[i])
        }
      case Int32(CCV_16F):
        let f32 = UnsafeMutableRawPointer(centroids).assumingMemoryBound(to: Float32.self)
        for i in 0..<16 {
          f32[i] = Float(centroids[i])
        }
        ccv_float_to_half_precision(f32, encodedBlock.assumingMemoryBound(to: UInt16.self), 16)
      default:
        return
      }
      let u8 = encodedBlock.assumingMemoryBound(to: UInt8.self) + 16 * elementSize
      var j = 0
      for i in stride(from: 0, to: nI, by: 2) {
        let i0 = UInt8(indices[i])
        let i1 = i + 1 < nI ? UInt8(indices[i + 1]) : 0
        let u0 = (i0 << 4) | i1
        u8[j] = u0
        j += 1
      }
      centroids.deallocate()
      indices.deallocate()
    }
    identifier?[0] = 0x8a1e4b
    encodedSize[0] =
      (numberOfElements + 1) / 2 + numberOfBlocks * 16 * elementSize + MemoryLayout<UInt32>.size
    return 1
  }

private let q4pDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    guard identifier == 0x8a1e4b else { return 0 }
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard var data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    let elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      elementSize = MemoryLayout<Double>.size
    case Int32(CCV_32F):
      elementSize = MemoryLayout<Float>.size
    case Int32(CCV_16F):
      elementSize = MemoryLayout<UInt16>.size
    default:
      return 0
    }
    var numberOfElements = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      numberOfElements *= Int(dimensions[i])
    }
    numberOfElements = min(decodedSize[0] / elementSize, numberOfElements)
    guard dataSize > MemoryLayout<UInt32>.size else { return 0 }
    let blockSize = Int(data.load(as: UInt32.self))
    data += MemoryLayout<UInt32>.size
    let numberOfBlocks = (numberOfElements + blockSize - 1) / blockSize
    guard
      dataSize >= (numberOfElements + 1) / 2 + numberOfBlocks * 16 * elementSize
        + MemoryLayout<UInt32>.size
    else { return 0 }
    precondition(blockSize % 2 == 0)
    switch dataType {
    case Int32(CCV_64F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (16 * elementSize + blockSize / 2) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: Double.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 16
        let f64 = decodedBlock.assumingMemoryBound(to: Double.self)
        var j = 0
        if nI % 2 == 0 {
          for i in stride(from: 0, to: nI, by: 2) {
            let u0 = u8[j]
            let i0 = Int(u0 >> 4)
            let i1 = Int(u0 & 15)
            f64[i] = palette[i0]
            f64[i + 1] = palette[i1]
            j += 1
          }
        } else {
          for i in stride(from: 0, to: nI, by: 2) {
            let u0 = u8[j]
            let i0 = Int(u0 >> 4)
            let i1 = Int(u0 & 15)
            f64[i] = palette[i0]
            if i + 1 < nI {
              f64[i + 1] = palette[i1]
            }
            j += 1
          }
        }
      }
    case Int32(CCV_32F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (16 * elementSize + blockSize / 2) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: Float.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 16
        let f32 = decodedBlock.assumingMemoryBound(to: Float.self)
        var j = 0
        if nI % 2 == 0 {
          for i in stride(from: 0, to: nI, by: 2) {
            let u0 = u8[j]
            let i0 = Int(u0 >> 4)
            let i1 = Int(u0 & 15)
            f32[i] = palette[i0]
            f32[i + 1] = palette[i1]
            j += 1
          }
        } else {
          for i in stride(from: 0, to: nI, by: 2) {
            let u0 = u8[j]
            let i0 = Int(u0 >> 4)
            let i1 = Int(u0 & 15)
            f32[i] = palette[i0]
            if i + 1 < nI {
              f32[i + 1] = palette[i1]
            }
            j += 1
          }
        }
      }
    case Int32(CCV_16F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (16 * elementSize + blockSize / 2) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: UInt16.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 16
        let f16 = decodedBlock.assumingMemoryBound(to: UInt16.self)
        var j = 0
        if nI % 2 == 0 {
          for i in stride(from: 0, to: nI, by: 2) {
            let u0 = u8[j]
            let i0 = Int(u0 >> 4)
            let i1 = Int(u0 & 15)
            f16[i] = palette[i0]
            f16[i + 1] = palette[i1]
            j += 1
          }
        } else {
          for i in stride(from: 0, to: nI, by: 2) {
            let u0 = u8[j]
            let i0 = Int(u0 >> 4)
            let i1 = Int(u0 & 15)
            f16[i] = palette[i0]
            if i + 1 < nI {
              f16[i + 1] = palette[i1]
            }
            j += 1
          }
        }
      }
    default:
      return 0
    }
    decodedSize[0] = elementSize * numberOfElements
    return 1
  }

// Quantize to 5-bit.
private let q5pEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard let data = data, let dimensions = dimensions, var encoded = encoded,
      let encodedSize = encodedSize, dimensionCount > 0
    else { return 0 }
    var elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      elementSize = MemoryLayout<Double>.size
    case Int32(CCV_32F):
      elementSize = MemoryLayout<Float>.size
    case Int32(CCV_16F):
      elementSize = MemoryLayout<UInt16>.size
    default:
      return 0
    }
    var numberOfElements = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      numberOfElements *= Int(dimensions[i])
    }
    let numberOfBlocks = (numberOfElements + 2047) / 2048
    guard
      (numberOfElements + 7) / 8 * 5 + numberOfBlocks * 32 * elementSize + MemoryLayout<UInt32>.size
        < encodedSize[0]
    else { return 0 }
    encoded.storeBytes(of: UInt32(2048), as: UInt32.self)
    encoded += MemoryLayout<UInt32>.size
    DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
      let indices = UnsafeMutablePointer<Int32>.allocate(capacity: min(2048, numberOfElements))
      let centroids = UnsafeMutablePointer<Double>.allocate(capacity: 32)
      let nI = min(2048, numberOfElements - blockIdx * 2048)
      var input = ccv_dense_matrix(
        1, Int32(nI), dataType | Int32(CCV_C1),
        UnsafeMutableRawPointer(mutating: data + blockIdx * 2048 * elementSize), 0)
      ccv_kmeans1d(&input, 32, indices, centroids)
      let encodedBlock = encoded + (32 * elementSize + 1280) * blockIdx
      switch dataType {
      case Int32(CCV_64F):
        // Write centroids directly to the output.
        memcpy(encodedBlock, centroids, elementSize * 32)
      case Int32(CCV_32F):
        let f32 = encodedBlock.assumingMemoryBound(to: Float32.self)
        for i in 0..<32 {
          f32[i] = Float(centroids[i])
        }
      case Int32(CCV_16F):
        let f32 = UnsafeMutableRawPointer(centroids).assumingMemoryBound(to: Float32.self)
        for i in 0..<32 {
          f32[i] = Float(centroids[i])
        }
        ccv_float_to_half_precision(f32, encodedBlock.assumingMemoryBound(to: UInt16.self), 32)
      default:
        return
      }
      let u8 = encodedBlock.assumingMemoryBound(to: UInt8.self) + 32 * elementSize
      var j = 0
      for i in stride(from: 0, to: nI, by: 8) {
        let i0 = UInt8(indices[i])
        let i1 = i + 1 < nI ? UInt8(indices[i + 1]) : 0
        let i2 = i + 2 < nI ? UInt8(indices[i + 2]) : 0
        let i3 = i + 3 < nI ? UInt8(indices[i + 3]) : 0
        let i4 = i + 4 < nI ? UInt8(indices[i + 4]) : 0
        let i5 = i + 5 < nI ? UInt8(indices[i + 5]) : 0
        let i6 = i + 6 < nI ? UInt8(indices[i + 6]) : 0
        let i7 = i + 7 < nI ? UInt8(indices[i + 7]) : 0
        let u0 = (i0 << 3) | (i1 >> 2)
        let u1 = (i1 << 6) | (i2 << 1) | (i3 >> 4)
        let u2 = (i3 << 4) | (i4 >> 1)
        let u3 = (i4 << 7) | (i5 << 2) | (i6 >> 3)
        let u4 = (i6 << 5) | i7
        u8[j] = u0
        u8[j + 1] = u1
        u8[j + 2] = u2
        u8[j + 3] = u3
        u8[j + 4] = u4
        j += 5
      }
      centroids.deallocate()
      indices.deallocate()
    }
    identifier?[0] = 0x8a1e5b
    encodedSize[0] =
      (numberOfElements + 7) / 8 * 5 + numberOfBlocks * 32 * elementSize + MemoryLayout<UInt32>.size
    return 1
  }

private let q5pDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    guard identifier == 0x8a1e5b else { return 0 }
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard var data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    let elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      elementSize = MemoryLayout<Double>.size
    case Int32(CCV_32F):
      elementSize = MemoryLayout<Float>.size
    case Int32(CCV_16F):
      elementSize = MemoryLayout<UInt16>.size
    default:
      return 0
    }
    var numberOfElements = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      numberOfElements *= Int(dimensions[i])
    }
    numberOfElements = min(decodedSize[0] / elementSize, numberOfElements)
    guard dataSize > MemoryLayout<UInt32>.size else { return 0 }
    let blockSize = Int(data.load(as: UInt32.self))
    data += MemoryLayout<UInt32>.size
    let numberOfBlocks = (numberOfElements + blockSize - 1) / blockSize
    guard
      dataSize >= (numberOfElements + 7) / 8 * 5 + numberOfBlocks * 32 * elementSize
        + MemoryLayout<UInt32>.size
    else { return 0 }
    precondition(blockSize % 8 == 0)
    switch dataType {
    case Int32(CCV_64F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (32 * elementSize + blockSize / 8 * 5) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: Double.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 32
        let f64 = decodedBlock.assumingMemoryBound(to: Double.self)
        var j = 0
        if nI % 8 == 0 {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let i0 = Int(u0 >> 3)
            let i1 = Int(((u0 & 7) << 2) | (u1 >> 6))
            let i2 = Int((u1 >> 1) & 31)
            let i3 = Int(((u1 & 1) << 4) | (u2 >> 4))
            let i4 = Int(((u2 & 15) << 1) | (u3 >> 7))
            let i5 = Int((u3 >> 2) & 31)
            let i6 = Int(((u3 & 3) << 3) | (u4 >> 5))
            let i7 = Int(u4 & 31)
            f64[i] = palette[i0]
            f64[i + 1] = palette[i1]
            f64[i + 2] = palette[i2]
            f64[i + 3] = palette[i3]
            f64[i + 4] = palette[i4]
            f64[i + 5] = palette[i5]
            f64[i + 6] = palette[i6]
            f64[i + 7] = palette[i7]
            j += 5
          }
        } else {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let i0 = Int(u0 >> 3)
            let i1 = Int(((u0 & 7) << 2) | (u1 >> 6))
            let i2 = Int((u1 >> 1) & 31)
            let i3 = Int(((u1 & 1) << 4) | (u2 >> 4))
            let i4 = Int(((u2 & 15) << 1) | (u3 >> 7))
            let i5 = Int((u3 >> 2) & 31)
            let i6 = Int(((u3 & 3) << 3) | (u4 >> 5))
            let i7 = Int(u4 & 31)
            f64[i] = palette[i0]
            if i + 1 < nI {
              f64[i + 1] = palette[i1]
            }
            if i + 2 < nI {
              f64[i + 2] = palette[i2]
            }
            if i + 3 < nI {
              f64[i + 3] = palette[i3]
            }
            if i + 4 < nI {
              f64[i + 4] = palette[i4]
            }
            if i + 5 < nI {
              f64[i + 5] = palette[i5]
            }
            if i + 6 < nI {
              f64[i + 6] = palette[i6]
            }
            if i + 7 < nI {
              f64[i + 7] = palette[i7]
            }
            j += 5
          }
        }
      }
    case Int32(CCV_32F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (32 * elementSize + blockSize / 8 * 5) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: Float.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 32
        let f32 = decodedBlock.assumingMemoryBound(to: Float.self)
        var j = 0
        if nI % 8 == 0 {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let i0 = Int(u0 >> 3)
            let i1 = Int(((u0 & 7) << 2) | (u1 >> 6))
            let i2 = Int((u1 >> 1) & 31)
            let i3 = Int(((u1 & 1) << 4) | (u2 >> 4))
            let i4 = Int(((u2 & 15) << 1) | (u3 >> 7))
            let i5 = Int((u3 >> 2) & 31)
            let i6 = Int(((u3 & 3) << 3) | (u4 >> 5))
            let i7 = Int(u4 & 31)
            f32[i] = palette[i0]
            f32[i + 1] = palette[i1]
            f32[i + 2] = palette[i2]
            f32[i + 3] = palette[i3]
            f32[i + 4] = palette[i4]
            f32[i + 5] = palette[i5]
            f32[i + 6] = palette[i6]
            f32[i + 7] = palette[i7]
            j += 5
          }
        } else {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let i0 = Int(u0 >> 3)
            let i1 = Int(((u0 & 7) << 2) | (u1 >> 6))
            let i2 = Int((u1 >> 1) & 31)
            let i3 = Int(((u1 & 1) << 4) | (u2 >> 4))
            let i4 = Int(((u2 & 15) << 1) | (u3 >> 7))
            let i5 = Int((u3 >> 2) & 31)
            let i6 = Int(((u3 & 3) << 3) | (u4 >> 5))
            let i7 = Int(u4 & 31)
            f32[i] = palette[i0]
            if i + 1 < nI {
              f32[i + 1] = palette[i1]
            }
            if i + 2 < nI {
              f32[i + 2] = palette[i2]
            }
            if i + 3 < nI {
              f32[i + 3] = palette[i3]
            }
            if i + 4 < nI {
              f32[i + 4] = palette[i4]
            }
            if i + 5 < nI {
              f32[i + 5] = palette[i5]
            }
            if i + 6 < nI {
              f32[i + 6] = palette[i6]
            }
            if i + 7 < nI {
              f32[i + 7] = palette[i7]
            }
            j += 5
          }
        }
      }
    case Int32(CCV_16F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (32 * elementSize + blockSize / 8 * 5) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: UInt16.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 32
        let f16 = decodedBlock.assumingMemoryBound(to: UInt16.self)
        var j = 0
        if nI % 8 == 0 {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let i0 = Int(u0 >> 3)
            let i1 = Int(((u0 & 7) << 2) | (u1 >> 6))
            let i2 = Int((u1 >> 1) & 31)
            let i3 = Int(((u1 & 1) << 4) | (u2 >> 4))
            let i4 = Int(((u2 & 15) << 1) | (u3 >> 7))
            let i5 = Int((u3 >> 2) & 31)
            let i6 = Int(((u3 & 3) << 3) | (u4 >> 5))
            let i7 = Int(u4 & 31)
            f16[i] = palette[i0]
            f16[i + 1] = palette[i1]
            f16[i + 2] = palette[i2]
            f16[i + 3] = palette[i3]
            f16[i + 4] = palette[i4]
            f16[i + 5] = palette[i5]
            f16[i + 6] = palette[i6]
            f16[i + 7] = palette[i7]
            j += 5
          }
        } else {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let i0 = Int(u0 >> 3)
            let i1 = Int(((u0 & 7) << 2) | (u1 >> 6))
            let i2 = Int((u1 >> 1) & 31)
            let i3 = Int(((u1 & 1) << 4) | (u2 >> 4))
            let i4 = Int(((u2 & 15) << 1) | (u3 >> 7))
            let i5 = Int((u3 >> 2) & 31)
            let i6 = Int(((u3 & 3) << 3) | (u4 >> 5))
            let i7 = Int(u4 & 31)
            f16[i] = palette[i0]
            if i + 1 < nI {
              f16[i + 1] = palette[i1]
            }
            if i + 2 < nI {
              f16[i + 2] = palette[i2]
            }
            if i + 3 < nI {
              f16[i + 3] = palette[i3]
            }
            if i + 4 < nI {
              f16[i + 4] = palette[i4]
            }
            if i + 5 < nI {
              f16[i + 5] = palette[i5]
            }
            if i + 6 < nI {
              f16[i + 6] = palette[i6]
            }
            if i + 7 < nI {
              f16[i + 7] = palette[i7]
            }
            j += 5
          }
        }
      }
    default:
      return 0
    }
    decodedSize[0] = elementSize * numberOfElements
    return 1
  }

// Quantize to 6-bit.
private let q6pEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard let data = data, let dimensions = dimensions, var encoded = encoded,
      let encodedSize = encodedSize, dimensionCount > 0
    else { return 0 }
    var elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      elementSize = MemoryLayout<Double>.size
    case Int32(CCV_32F):
      elementSize = MemoryLayout<Float>.size
    case Int32(CCV_16F):
      elementSize = MemoryLayout<UInt16>.size
    default:
      return 0
    }
    var numberOfElements = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      numberOfElements *= Int(dimensions[i])
    }
    let numberOfBlocks = (numberOfElements + 4095) / 4096
    guard
      (numberOfElements + 3) / 4 * 3 + numberOfBlocks * 64 * elementSize + MemoryLayout<UInt32>.size
        < encodedSize[0]
    else { return 0 }
    encoded.storeBytes(of: UInt32(4096), as: UInt32.self)
    encoded += MemoryLayout<UInt32>.size
    DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
      let indices = UnsafeMutablePointer<Int32>.allocate(capacity: min(4096, numberOfElements))
      let centroids = UnsafeMutablePointer<Double>.allocate(capacity: 64)
      let nI = min(4096, numberOfElements - blockIdx * 4096)
      var input = ccv_dense_matrix(
        1, Int32(nI), dataType | Int32(CCV_C1),
        UnsafeMutableRawPointer(mutating: data + 4096 * elementSize * blockIdx), 0)
      ccv_kmeans1d(&input, 64, indices, centroids)
      let encodedBlock = encoded + (64 * elementSize + 3072) * blockIdx
      switch dataType {
      case Int32(CCV_64F):
        // Write centroids directly to the output.
        memcpy(encodedBlock, centroids, elementSize * 64)
      case Int32(CCV_32F):
        let f32 = encodedBlock.assumingMemoryBound(to: Float32.self)
        for i in 0..<64 {
          f32[i] = Float(centroids[i])
        }
      case Int32(CCV_16F):
        let f32 = UnsafeMutableRawPointer(centroids).assumingMemoryBound(to: Float32.self)
        for i in 0..<64 {
          f32[i] = Float(centroids[i])
        }
        ccv_float_to_half_precision(f32, encodedBlock.assumingMemoryBound(to: UInt16.self), 64)
      default:
        return
      }
      let u8 = encodedBlock.assumingMemoryBound(to: UInt8.self) + 64 * elementSize
      var j = 0
      for i in stride(from: 0, to: nI, by: 4) {
        let i0 = UInt8(indices[i])
        let i1 = i + 1 < nI ? UInt8(indices[i + 1]) : 0
        let i2 = i + 2 < nI ? UInt8(indices[i + 2]) : 0
        let i3 = i + 3 < nI ? UInt8(indices[i + 3]) : 0
        let u0 = (i0 << 2) | (i1 >> 4)
        let u1 = (i1 << 4) | (i2 >> 2)
        let u2 = (i2 << 6) | i3
        u8[j] = u0
        u8[j + 1] = u1
        u8[j + 2] = u2
        j += 3
      }
      centroids.deallocate()
      indices.deallocate()
    }
    identifier?[0] = 0x8a1e6b
    encodedSize[0] =
      (numberOfElements + 3) / 4 * 3 + numberOfBlocks * 64 * elementSize + MemoryLayout<UInt32>.size
    return 1
  }

private let q6pDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    guard identifier == 0x8a1e6b else { return 0 }
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard var data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    let elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      elementSize = MemoryLayout<Double>.size
    case Int32(CCV_32F):
      elementSize = MemoryLayout<Float>.size
    case Int32(CCV_16F):
      elementSize = MemoryLayout<UInt16>.size
    default:
      return 0
    }
    var numberOfElements = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      numberOfElements *= Int(dimensions[i])
    }
    numberOfElements = min(decodedSize[0] / elementSize, numberOfElements)
    guard dataSize > MemoryLayout<UInt32>.size else { return 0 }
    let blockSize = Int(data.load(as: UInt32.self))
    data += MemoryLayout<UInt32>.size
    let numberOfBlocks = (numberOfElements + blockSize - 1) / blockSize
    guard
      dataSize >= (numberOfElements + 3) / 4 * 3 + numberOfBlocks * 64 * elementSize
        + MemoryLayout<UInt32>.size
    else { return 0 }
    precondition(blockSize % 4 == 0)
    switch dataType {
    case Int32(CCV_64F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (64 * elementSize + blockSize / 4 * 3) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: Double.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 64
        let f64 = decodedBlock.assumingMemoryBound(to: Double.self)
        var j = 0
        if nI % 4 == 0 {
          for i in stride(from: 0, to: nI, by: 4) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let i0 = Int(u0 >> 2)
            let i1 = Int(((u0 & 3) << 4) | (u1 >> 4))
            let i2 = Int(((u1 & 15) << 2) | (u2 >> 6))
            let i3 = Int(u2 & 63)
            f64[i] = palette[i0]
            f64[i + 1] = palette[i1]
            f64[i + 2] = palette[i2]
            f64[i + 3] = palette[i3]
            j += 3
          }
        } else {
          for i in stride(from: 0, to: nI, by: 4) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let i0 = Int(u0 >> 2)
            let i1 = Int(((u0 & 3) << 4) | (u1 >> 4))
            let i2 = Int(((u1 & 15) << 2) | (u2 >> 6))
            let i3 = Int(u2 & 63)
            f64[i] = palette[i0]
            if i + 1 < nI {
              f64[i + 1] = palette[i1]
            }
            if i + 2 < nI {
              f64[i + 2] = palette[i2]
            }
            if i + 3 < nI {
              f64[i + 3] = palette[i3]
            }
            j += 3
          }
        }
      }
    case Int32(CCV_32F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (64 * elementSize + blockSize / 4 * 3) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: Float.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 64
        let f32 = decodedBlock.assumingMemoryBound(to: Float.self)
        var j = 0
        if nI % 4 == 0 {
          for i in stride(from: 0, to: nI, by: 4) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let i0 = Int(u0 >> 2)
            let i1 = Int(((u0 & 3) << 4) | (u1 >> 4))
            let i2 = Int(((u1 & 15) << 2) | (u2 >> 6))
            let i3 = Int(u2 & 63)
            f32[i] = palette[i0]
            f32[i + 1] = palette[i1]
            f32[i + 2] = palette[i2]
            f32[i + 3] = palette[i3]
            j += 3
          }
        } else {
          for i in stride(from: 0, to: nI, by: 4) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let i0 = Int(u0 >> 2)
            let i1 = Int(((u0 & 3) << 4) | (u1 >> 4))
            let i2 = Int(((u1 & 15) << 2) | (u2 >> 6))
            let i3 = Int(u2 & 63)
            f32[i] = palette[i0]
            if i + 1 < nI {
              f32[i + 1] = palette[i1]
            }
            if i + 2 < nI {
              f32[i + 2] = palette[i2]
            }
            if i + 3 < nI {
              f32[i + 3] = palette[i3]
            }
            j += 3
          }
        }
      }
    case Int32(CCV_16F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (64 * elementSize + blockSize / 4 * 3) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: UInt16.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 64
        let f16 = decodedBlock.assumingMemoryBound(to: UInt16.self)
        var j = 0
        if nI % 4 == 0 {
          for i in stride(from: 0, to: nI, by: 4) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let i0 = Int(u0 >> 2)
            let i1 = Int(((u0 & 3) << 4) | (u1 >> 4))
            let i2 = Int(((u1 & 15) << 2) | (u2 >> 6))
            let i3 = Int(u2 & 63)
            f16[i] = palette[i0]
            f16[i + 1] = palette[i1]
            f16[i + 2] = palette[i2]
            f16[i + 3] = palette[i3]
            j += 3
          }
        } else {
          for i in stride(from: 0, to: nI, by: 4) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let i0 = Int(u0 >> 2)
            let i1 = Int(((u0 & 3) << 4) | (u1 >> 4))
            let i2 = Int(((u1 & 15) << 2) | (u2 >> 6))
            let i3 = Int(u2 & 63)
            f16[i] = palette[i0]
            if i + 1 < nI {
              f16[i + 1] = palette[i1]
            }
            if i + 2 < nI {
              f16[i + 2] = palette[i2]
            }
            if i + 3 < nI {
              f16[i + 3] = palette[i3]
            }
            j += 3
          }
        }
      }
    default:
      return 0
    }
    decodedSize[0] = elementSize * numberOfElements
    return 1
  }

// Quantize to 7-bit.
private let q7pEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard let data = data, let dimensions = dimensions, var encoded = encoded,
      let encodedSize = encodedSize, dimensionCount > 0
    else { return 0 }
    var elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      elementSize = MemoryLayout<Double>.size
    case Int32(CCV_32F):
      elementSize = MemoryLayout<Float>.size
    case Int32(CCV_16F):
      elementSize = MemoryLayout<UInt16>.size
    default:
      return 0
    }
    var numberOfElements = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      numberOfElements *= Int(dimensions[i])
    }
    let numberOfBlocks = (numberOfElements + 8191) / 8192
    guard
      (numberOfElements + 7) / 8 * 7 + numberOfBlocks * 128 * elementSize
        + MemoryLayout<UInt32>.size
        < encodedSize[0]
    else { return 0 }
    encoded.storeBytes(of: UInt32(8192), as: UInt32.self)
    encoded += MemoryLayout<UInt32>.size
    DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
      let indices = UnsafeMutablePointer<Int32>.allocate(capacity: min(8192, numberOfElements))
      let centroids = UnsafeMutablePointer<Double>.allocate(capacity: 128)
      let nI = min(8192, numberOfElements - blockIdx * 8192)
      var input = ccv_dense_matrix(
        1, Int32(nI), dataType | Int32(CCV_C1),
        UnsafeMutableRawPointer(mutating: data + 8192 * elementSize * blockIdx), 0)
      ccv_kmeans1d(&input, 128, indices, centroids)
      let encodedBlock = encoded + (128 * elementSize + 7168) * blockIdx
      switch dataType {
      case Int32(CCV_64F):
        // Write centroids directly to the output.
        memcpy(encodedBlock, centroids, elementSize * 128)
      case Int32(CCV_32F):
        let f32 = encodedBlock.assumingMemoryBound(to: Float32.self)
        for i in 0..<128 {
          f32[i] = Float(centroids[i])
        }
      case Int32(CCV_16F):
        let f32 = UnsafeMutableRawPointer(centroids).assumingMemoryBound(to: Float32.self)
        for i in 0..<128 {
          f32[i] = Float(centroids[i])
        }
        ccv_float_to_half_precision(f32, encodedBlock.assumingMemoryBound(to: UInt16.self), 128)
      default:
        return
      }
      let u8 = encodedBlock.assumingMemoryBound(to: UInt8.self) + 128 * elementSize
      var j = 0
      for i in stride(from: 0, to: nI, by: 8) {
        let i0 = UInt8(indices[i])
        let i1 = i + 1 < nI ? UInt8(indices[i + 1]) : 0
        let i2 = i + 2 < nI ? UInt8(indices[i + 2]) : 0
        let i3 = i + 3 < nI ? UInt8(indices[i + 3]) : 0
        let i4 = i + 4 < nI ? UInt8(indices[i + 4]) : 0
        let i5 = i + 5 < nI ? UInt8(indices[i + 5]) : 0
        let i6 = i + 6 < nI ? UInt8(indices[i + 6]) : 0
        let i7 = i + 7 < nI ? UInt8(indices[i + 7]) : 0
        let u0 = (i0 << 1) | (i1 >> 6)
        let u1 = (i1 << 2) | (i2 >> 5)
        let u2 = (i2 << 3) | (i3 >> 4)
        let u3 = (i3 << 4) | (i4 >> 3)
        let u4 = (i4 << 5) | (i5 >> 2)
        let u5 = (i5 << 6) | (i6 >> 1)
        let u6 = (i6 << 7) | i7
        u8[j] = u0
        u8[j + 1] = u1
        u8[j + 2] = u2
        u8[j + 3] = u3
        u8[j + 4] = u4
        u8[j + 5] = u5
        u8[j + 6] = u6
        j += 7
      }
      centroids.deallocate()
      indices.deallocate()
    }
    identifier?[0] = 0x8a1e7b
    encodedSize[0] =
      (numberOfElements + 7) / 8 * 7 + numberOfBlocks * 128 * elementSize
      + MemoryLayout<UInt32>.size
    return 1
  }

private let q7pDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    guard identifier == 0x8a1e7b else { return 0 }
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard var data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    let elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      elementSize = MemoryLayout<Double>.size
    case Int32(CCV_32F):
      elementSize = MemoryLayout<Float>.size
    case Int32(CCV_16F):
      elementSize = MemoryLayout<UInt16>.size
    default:
      return 0
    }
    var numberOfElements = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      numberOfElements *= Int(dimensions[i])
    }
    numberOfElements = min(decodedSize[0] / elementSize, numberOfElements)
    guard dataSize > MemoryLayout<UInt32>.size else { return 0 }
    let blockSize = Int(data.load(as: UInt32.self))
    data += MemoryLayout<UInt32>.size
    let numberOfBlocks = (numberOfElements + blockSize - 1) / blockSize
    guard
      dataSize >= (numberOfElements + 7) / 8 * 7 + numberOfBlocks * 128 * elementSize
        + MemoryLayout<UInt32>.size
    else { return 0 }
    precondition(blockSize % 8 == 0)
    switch dataType {
    case Int32(CCV_64F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (128 * elementSize + blockSize / 8 * 7) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: Double.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 128
        let f64 = decodedBlock.assumingMemoryBound(to: Double.self)
        var j = 0
        if nI % 8 == 0 {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let u5 = u8[j + 5]
            let u6 = u8[j + 6]
            let i0 = Int(u0 >> 1)
            let i1 = Int(((u0 & 1) << 6) | (u1 >> 2))
            let i2 = Int(((u1 & 3) << 5) | (u2 >> 3))
            let i3 = Int(((u2 & 7) << 4) | (u3 >> 4))
            let i4 = Int(((u3 & 15) << 3) | (u4 >> 5))
            let i5 = Int(((u4 & 31) << 2) | (u5 >> 6))
            let i6 = Int(((u5 & 63) << 1) | (u6 >> 7))
            let i7 = Int(u6 & 127)
            f64[i] = palette[i0]
            f64[i + 1] = palette[i1]
            f64[i + 2] = palette[i2]
            f64[i + 3] = palette[i3]
            f64[i + 4] = palette[i4]
            f64[i + 5] = palette[i5]
            f64[i + 6] = palette[i6]
            f64[i + 7] = palette[i7]
            j += 7
          }
        } else {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let u5 = u8[j + 5]
            let u6 = u8[j + 6]
            let i0 = Int(u0 >> 1)
            let i1 = Int(((u0 & 1) << 6) | (u1 >> 2))
            let i2 = Int(((u1 & 3) << 5) | (u2 >> 3))
            let i3 = Int(((u2 & 7) << 4) | (u3 >> 4))
            let i4 = Int(((u3 & 15) << 3) | (u4 >> 5))
            let i5 = Int(((u4 & 31) << 2) | (u5 >> 6))
            let i6 = Int(((u5 & 63) << 1) | (u6 >> 7))
            let i7 = Int(u6 & 127)
            f64[i] = palette[i0]
            if i + 1 < nI {
              f64[i + 1] = palette[i1]
            }
            if i + 2 < nI {
              f64[i + 2] = palette[i2]
            }
            if i + 3 < nI {
              f64[i + 3] = palette[i3]
            }
            if i + 4 < nI {
              f64[i + 4] = palette[i4]
            }
            if i + 5 < nI {
              f64[i + 5] = palette[i5]
            }
            if i + 6 < nI {
              f64[i + 6] = palette[i6]
            }
            if i + 7 < nI {
              f64[i + 7] = palette[i7]
            }
            j += 7
          }
        }
      }
    case Int32(CCV_32F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (128 * elementSize + blockSize / 8 * 7) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: Float.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 128
        let f32 = decodedBlock.assumingMemoryBound(to: Float.self)
        var j = 0
        if nI % 8 == 0 {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let u5 = u8[j + 5]
            let u6 = u8[j + 6]
            let i0 = Int(u0 >> 1)
            let i1 = Int(((u0 & 1) << 6) | (u1 >> 2))
            let i2 = Int(((u1 & 3) << 5) | (u2 >> 3))
            let i3 = Int(((u2 & 7) << 4) | (u3 >> 4))
            let i4 = Int(((u3 & 15) << 3) | (u4 >> 5))
            let i5 = Int(((u4 & 31) << 2) | (u5 >> 6))
            let i6 = Int(((u5 & 63) << 1) | (u6 >> 7))
            let i7 = Int(u6 & 127)
            f32[i] = palette[i0]
            f32[i + 1] = palette[i1]
            f32[i + 2] = palette[i2]
            f32[i + 3] = palette[i3]
            f32[i + 4] = palette[i4]
            f32[i + 5] = palette[i5]
            f32[i + 6] = palette[i6]
            f32[i + 7] = palette[i7]
            j += 7
          }
        } else {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let u5 = u8[j + 5]
            let u6 = u8[j + 6]
            let i0 = Int(u0 >> 1)
            let i1 = Int(((u0 & 1) << 6) | (u1 >> 2))
            let i2 = Int(((u1 & 3) << 5) | (u2 >> 3))
            let i3 = Int(((u2 & 7) << 4) | (u3 >> 4))
            let i4 = Int(((u3 & 15) << 3) | (u4 >> 5))
            let i5 = Int(((u4 & 31) << 2) | (u5 >> 6))
            let i6 = Int(((u5 & 63) << 1) | (u6 >> 7))
            let i7 = Int(u6 & 127)
            f32[i] = palette[i0]
            if i + 1 < nI {
              f32[i + 1] = palette[i1]
            }
            if i + 2 < nI {
              f32[i + 2] = palette[i2]
            }
            if i + 3 < nI {
              f32[i + 3] = palette[i3]
            }
            if i + 4 < nI {
              f32[i + 4] = palette[i4]
            }
            if i + 5 < nI {
              f32[i + 5] = palette[i5]
            }
            if i + 6 < nI {
              f32[i + 6] = palette[i6]
            }
            if i + 7 < nI {
              f32[i + 7] = palette[i7]
            }
            j += 7
          }
        }
      }
    case Int32(CCV_16F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (128 * elementSize + blockSize / 8 * 7) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: UInt16.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 128
        let f16 = decodedBlock.assumingMemoryBound(to: UInt16.self)
        var j = 0
        if nI % 8 == 0 {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let u5 = u8[j + 5]
            let u6 = u8[j + 6]
            let i0 = Int(u0 >> 1)
            let i1 = Int(((u0 & 1) << 6) | (u1 >> 2))
            let i2 = Int(((u1 & 3) << 5) | (u2 >> 3))
            let i3 = Int(((u2 & 7) << 4) | (u3 >> 4))
            let i4 = Int(((u3 & 15) << 3) | (u4 >> 5))
            let i5 = Int(((u4 & 31) << 2) | (u5 >> 6))
            let i6 = Int(((u5 & 63) << 1) | (u6 >> 7))
            let i7 = Int(u6 & 127)
            f16[i] = palette[i0]
            f16[i + 1] = palette[i1]
            f16[i + 2] = palette[i2]
            f16[i + 3] = palette[i3]
            f16[i + 4] = palette[i4]
            f16[i + 5] = palette[i5]
            f16[i + 6] = palette[i6]
            f16[i + 7] = palette[i7]
            j += 7
          }
        } else {
          for i in stride(from: 0, to: nI, by: 8) {
            let u0 = u8[j]
            let u1 = u8[j + 1]
            let u2 = u8[j + 2]
            let u3 = u8[j + 3]
            let u4 = u8[j + 4]
            let u5 = u8[j + 5]
            let u6 = u8[j + 6]
            let i0 = Int(u0 >> 1)
            let i1 = Int(((u0 & 1) << 6) | (u1 >> 2))
            let i2 = Int(((u1 & 3) << 5) | (u2 >> 3))
            let i3 = Int(((u2 & 7) << 4) | (u3 >> 4))
            let i4 = Int(((u3 & 15) << 3) | (u4 >> 5))
            let i5 = Int(((u4 & 31) << 2) | (u5 >> 6))
            let i6 = Int(((u5 & 63) << 1) | (u6 >> 7))
            let i7 = Int(u6 & 127)
            f16[i] = palette[i0]
            if i + 1 < nI {
              f16[i + 1] = palette[i1]
            }
            if i + 2 < nI {
              f16[i + 2] = palette[i2]
            }
            if i + 3 < nI {
              f16[i + 3] = palette[i3]
            }
            if i + 4 < nI {
              f16[i + 4] = palette[i4]
            }
            if i + 5 < nI {
              f16[i + 5] = palette[i5]
            }
            if i + 6 < nI {
              f16[i + 6] = palette[i6]
            }
            if i + 7 < nI {
              f16[i + 7] = palette[i7]
            }
            j += 7
          }
        }
      }
    default:
      return 0
    }
    decodedSize[0] = elementSize * numberOfElements
    return 1
  }

// Quantize to 8-bit.
private let q8pEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard let data = data, let dimensions = dimensions, var encoded = encoded,
      let encodedSize = encodedSize, dimensionCount > 0
    else { return 0 }
    var elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      elementSize = MemoryLayout<Double>.size
    case Int32(CCV_32F):
      elementSize = MemoryLayout<Float>.size
    case Int32(CCV_16F):
      elementSize = MemoryLayout<UInt16>.size
    default:
      return 0
    }
    var numberOfElements = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      numberOfElements *= Int(dimensions[i])
    }
    let numberOfBlocks = (numberOfElements + 16_383) / 16_384
    guard numberOfElements + 256 * elementSize + MemoryLayout<UInt32>.size < encodedSize[0] else {
      return 0
    }
    encoded.storeBytes(of: UInt32(16_384), as: UInt32.self)
    encoded += MemoryLayout<UInt32>.size
    DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
      let indices = UnsafeMutablePointer<Int32>.allocate(capacity: min(16_384, numberOfElements))
      let centroids = UnsafeMutablePointer<Double>.allocate(capacity: 256)
      let nI = min(16_384, numberOfElements - blockIdx * 16_384)
      var input = ccv_dense_matrix(
        1, Int32(nI), dataType | Int32(CCV_C1),
        UnsafeMutableRawPointer(mutating: data + 16_384 * elementSize * blockIdx), 0)
      ccv_kmeans1d(&input, 256, indices, centroids)
      let encodedBlock = encoded + (256 * elementSize + 16_384) * blockIdx
      switch dataType {
      case Int32(CCV_64F):
        // Write centroids directly to the output.
        memcpy(encodedBlock, centroids, elementSize * 256)
      case Int32(CCV_32F):
        let f32 = encodedBlock.assumingMemoryBound(to: Float32.self)
        for i in 0..<256 {
          f32[i] = Float(centroids[i])
        }
      case Int32(CCV_16F):
        let f32 = UnsafeMutableRawPointer(centroids).assumingMemoryBound(to: Float32.self)
        for i in 0..<256 {
          f32[i] = Float(centroids[i])
        }
        ccv_float_to_half_precision(
          f32, encodedBlock.assumingMemoryBound(to: UInt16.self), 256)
      default:
        return
      }
      let u8 = encodedBlock.assumingMemoryBound(to: UInt8.self) + 256 * elementSize
      for i in 0..<nI {
        u8[i] = UInt8(indices[i])
      }
      centroids.deallocate()
      indices.deallocate()
    }
    identifier?[0] = 0x8a1e8b
    encodedSize[0] =
      numberOfElements + numberOfBlocks * 256 * elementSize + MemoryLayout<UInt32>.size
    return 1
  }

private let q8pDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    guard identifier == 0x8a1e8b else { return 0 }
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard var data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    let elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      elementSize = MemoryLayout<Double>.size
    case Int32(CCV_32F):
      elementSize = MemoryLayout<Float>.size
    case Int32(CCV_16F):
      elementSize = MemoryLayout<UInt16>.size
    default:
      return 0
    }
    var numberOfElements = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      numberOfElements *= Int(dimensions[i])
    }
    numberOfElements = min(decodedSize[0] / elementSize, numberOfElements)
    guard dataSize > MemoryLayout<UInt32>.size else { return 0 }
    let blockSize = Int(data.load(as: UInt32.self))
    data += MemoryLayout<UInt32>.size
    let numberOfBlocks = (numberOfElements + blockSize - 1) / blockSize
    guard
      dataSize >= numberOfElements + numberOfBlocks * 256 * elementSize + MemoryLayout<UInt32>.size
    else { return 0 }
    switch dataType {
    case Int32(CCV_64F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (256 * elementSize + blockSize) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: Double.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 256
        let f64 = decodedBlock.assumingMemoryBound(to: Double.self)
        for i in 0..<nI {
          f64[i] = palette[Int(u8[i])]
        }
      }
    case Int32(CCV_32F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (256 * elementSize + blockSize) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: Float.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 256
        let f32 = decodedBlock.assumingMemoryBound(to: Float.self)
        for i in 0..<nI {
          f32[i] = palette[Int(u8[i])]
        }
      }
    case Int32(CCV_16F):
      DispatchQueue.concurrentPerform(iterations: numberOfBlocks) { blockIdx in
        let nI = min(blockSize, numberOfElements - blockIdx * blockSize)
        let dataBlock = data + (256 * elementSize + blockSize) * blockIdx
        let decodedBlock = decoded + blockSize * elementSize * blockIdx
        let palette = dataBlock.assumingMemoryBound(to: UInt16.self)
        let u8 = dataBlock.assumingMemoryBound(to: UInt8.self) + elementSize * 256
        let f16 = decodedBlock.assumingMemoryBound(to: UInt16.self)
        for i in 0..<nI {
          f16[i] = palette[Int(u8[i])]
        }
      }
    default:
      return 0
    }
    decodedSize[0] = elementSize * numberOfElements
    return 1
  }

private let fpzipEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard let data = data, let dimensions = dimensions, let encoded = encoded,
      let encodedSize = encodedSize, dimensionCount > 0
    else { return 0 }
    guard let fpz = fpzip_write_to_buffer(encoded, encodedSize[0]) else { return 0 }
    defer { fpzip_write_close(fpz) }
    fpz.pointee.type = Int32(FPZIP_TYPE_FLOAT)
    switch dataType {
    case Int32(CCV_64F):
      fpz.pointee.type = Int32(FPZIP_TYPE_DOUBLE)
      fpz.pointee.prec = 64
    case Int32(CCV_32F):
      fpz.pointee.type = Int32(FPZIP_TYPE_FLOAT)
      fpz.pointee.prec = 32
    case Int32(CCV_16F):
      fpz.pointee.type = Int32(FPZIP_TYPE_FLOAT)
      fpz.pointee.prec = 19  // Since we have to retain all exponetial components (i.e. 8, in Float16, that is 5), we need to specify a precision at least 19.
    default:
      return 0
    }
    fpz.pointee.nx = dimensions[Int(dimensionCount) - 1]
    fpz.pointee.ny = dimensionCount >= 2 ? dimensions[Int(dimensionCount) - 2] : 1
    fpz.pointee.nz = dimensionCount >= 3 ? dimensions[Int(dimensionCount) - 3] : 1
    if dimensionCount == 4 {
      fpz.pointee.nf = dimensions[0]
    } else if dimensionCount > 4 {
      var remainingCount = dimensions[Int(dimensionCount) - 4]
      for i in 4..<Int(dimensionCount) {
        remainingCount *= dimensions[Int(dimensionCount) - i - 1]
      }
      fpz.pointee.nf = remainingCount
    }
    fpzip_write_header(fpz)
    var inputData = data
    let f32: UnsafeMutablePointer<Float32>?
    if dataType == Int32(CCV_16F) {
      // Need to do the conversion
      let len =
        Int(fpz.pointee.nx) * Int(fpz.pointee.ny) * Int(fpz.pointee.nz) * Int(fpz.pointee.nf)
      let f32p = UnsafeMutablePointer<Float32>.allocate(capacity: len)
      ccv_half_precision_to_float(
        UnsafeMutableRawPointer(mutating: data).assumingMemoryBound(to: UInt16.self), f32p, len)
      inputData = UnsafeRawPointer(f32p)
      f32 = f32p
    } else {
      f32 = nil
    }
    let outbytes = fpzip_write(fpz, inputData)
    if let f32 = f32 {
      f32.deallocate()
    }
    guard outbytes > 0 && outbytes <= dataSize else { return 0 }
    identifier?[0] = 0xf7217
    encodedSize[0] = outbytes
    return 1
  }

private let fpzipDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    guard identifier == 0xf7217 else { return 0 }
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard let data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    guard let fpz = fpzip_read_from_buffer(data) else { return 0 }
    defer { fpzip_read_close(fpz) }
    guard fpzip_read_header(fpz) != 0 else { return 0 }
    let truncatedCount: Int
    let truncatedLength: Int
    let elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      guard fpz.pointee.type == Int32(FPZIP_TYPE_DOUBLE) else { return 0 }
      elementSize = MemoryLayout<Double>.size
      truncatedCount = decodedSize[0] / MemoryLayout<Double>.size
      truncatedLength = truncatedCount * MemoryLayout<Double>.size
    case Int32(CCV_32F):
      guard fpz.pointee.type == Int32(FPZIP_TYPE_FLOAT) else { return 0 }
      elementSize = MemoryLayout<Float>.size
      truncatedCount = decodedSize[0] / MemoryLayout<Float>.size
      truncatedLength = truncatedCount * MemoryLayout<Float>.size
    case Int32(CCV_16F):
      guard fpz.pointee.type == Int32(FPZIP_TYPE_FLOAT) else { return 0 }
      elementSize = MemoryLayout<Float>.size
      truncatedCount = decodedSize[0] / MemoryLayout<UInt16>.size
      truncatedLength = truncatedCount * MemoryLayout<UInt16>.size
    default:
      return 0
    }
    let len = Int(fpz.pointee.nx) * Int(fpz.pointee.ny) * Int(fpz.pointee.nz) * Int(fpz.pointee.nf)
    if decodedSize[0] < elementSize * len {
      let buffer = UnsafeMutableRawPointer.allocate(
        byteCount: elementSize * len, alignment: elementSize)
      defer { buffer.deallocate() }
      guard fpzip_read(fpz, buffer) != 0 else { return 0 }
      if dataType == Int32(CCV_16F) {
        ccv_float_to_half_precision(
          buffer.assumingMemoryBound(to: Float.self), buffer.assumingMemoryBound(to: UInt16.self),
          truncatedCount)
      }
      memcpy(decoded, buffer, truncatedLength)
      decodedSize[0] = truncatedLength
      return 1
    } else {
      // Decode directly, and then we do conversion, if it is CCV_16F.
      guard fpzip_read(fpz, decoded) != 0 else { return 0 }
      if dataType == Int32(CCV_16F) {
        ccv_float_to_half_precision(
          decoded.assumingMemoryBound(to: Float.self), decoded.assumingMemoryBound(to: UInt16.self),
          len)
      }
      decodedSize[0] = elementSize * len
    }
    return 1
  }

private let zipEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
    identifier
    in
    guard let data = data, let dimensions = dimensions, let encoded = encoded,
      let encodedSize = encodedSize, dimensionCount > 0
    else { return 0 }
    guard zip(data: data, dataSize: dataSize, zippedData: encoded, zippedDataSize: encodedSize)
    else {
      return 0
    }
    identifier?[0] = 0x217
    return 1
  }

private let zipDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
    decodedSize
    in
    guard identifier == 0x217 else { return 0 }
    guard let data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    guard
      unzip(data: data, dataSize: dataSize, unzippedData: decoded, unzippedDataSize: decodedSize)
    else { return 0 }
    return 1
  }

func truncatedBits(_ number: UInt16, bitCount: UInt16) -> UInt16 {
  guard bitCount > 0 else { return number }
  let mask: UInt16 = (1 << bitCount) - 1
  let discard = number & mask
  let threshold: UInt16 = 1 << (bitCount - 1)
  var shifted = number >> bitCount
  if discard > threshold || (discard == threshold && (shifted & 1) == 1) {
    shifted += 1  // Round to even
  }
  return shifted
}

// The ezm7 format consists of:
// |-- zipped exponents size (Int32) --|-- zipped exponents --|-- float without exponent --|
// Each float without exponent is an 8-bit chunk of data:
// |-- sign bit --|-- truncated mantissa (7 bits) --|
// By putting the exponent into its own byte, it seems to make it much easier for zip to compress
// it well. As for the sign bit and mantissa, they have so far been uncompressible
private let ezm7Encode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
    identifier
    in
    guard let data = data, let dimensions = dimensions, let encoded = encoded,
      let encodedSize = encodedSize, dimensionCount > 0
    else { return 0 }
    guard dataType == Int32(CCV_16F) else { return 0 }
    var floatCount = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      floatCount *= Int(dimensions[i])
    }
    let floatBytesBuffer = data.assumingMemoryBound(to: UInt16.self)
    let floatsWithoutExp = UnsafeMutablePointer<UInt8>.allocate(capacity: floatCount)
    let exponents = UnsafeMutablePointer<UInt8>.allocate(capacity: floatCount)
    defer {
      exponents.deallocate()
      floatsWithoutExp.deallocate()
    }
    for i in 0..<floatCount {
      let floatBytes = floatBytesBuffer[i]
      let exponent = UInt8((floatBytes >> 10) & ((1 << 5) - 1))
      let signBit = UInt8(floatBytes >> 15)
      let mantissa = floatBytes & ((1 << 10) - 1)
      var truncatedMantissa = UInt8(truncatedBits(mantissa, bitCount: 3))
      if (truncatedMantissa & (1 << 7)) != 0 {
        // If rounding would cause overflow, just round down instead
        truncatedMantissa = UInt8(mantissa >> 3)
      }
      exponents[i] = UInt8(exponent)
      floatsWithoutExp[i] = (signBit << 7) | truncatedMantissa
    }
    guard encodedSize[0] > 4 else { return 0 }
    var zippedDataSize = encodedSize[0] - 4
    guard
      zip(
        data: exponents,
        dataSize: floatCount,
        zippedData: encoded.advanced(by: 4),
        zippedDataSize: &zippedDataSize)
    else { return 0 }
    encoded.assumingMemoryBound(to: Int32.self)[0] = Int32(zippedDataSize)
    guard 4 + zippedDataSize + floatCount <= encodedSize[0] else { return 0 }
    memcpy(encoded.advanced(by: 4 + zippedDataSize), floatsWithoutExp, floatCount)
    identifier?[0] = 0x511
    encodedSize[0] =
      4 /* for compressed exponents size */
      + zippedDataSize /* exponents */
      + floatCount /* floats without exponent */
    return 1
  }

private let ezm7Decode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    guard dataType == Int32(CCV_16F) else { return 0 }
    guard identifier == 0x511 else { return 0 }
    guard let data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    var floatCount = Int(dimensions[0])
    for i in 1..<Int(dimensionCount) {
      floatCount *= Int(dimensions[i])
    }
    floatCount = min(floatCount, decodedSize[0] / 2)
    let exponentZipSize = Int(data.assumingMemoryBound(to: Int32.self)[0])
    guard dataSize >= 4 + exponentZipSize + floatCount else { return 0 }
    let exponentZipData = data.advanced(by: MemoryLayout<Int32>.size)
    let exponentBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: floatCount)
    defer { exponentBuffer.deallocate() }
    var unzippedDataSize = floatCount
    guard
      unzip(
        data: exponentZipData,
        dataSize: exponentZipSize,
        unzippedData: exponentBuffer,
        unzippedDataSize: &unzippedDataSize)
    else { return 0 }
    let floatsWithoutExp = exponentZipData.advanced(by: exponentZipSize).assumingMemoryBound(
      to: UInt8.self)
    let decodedAsInts = decoded.assumingMemoryBound(to: UInt16.self)
    for i in 0..<floatCount {
      let floatWithoutExp = UInt16(floatsWithoutExp[i])
      let signBit = floatWithoutExp >> 7
      let mantissa = (floatWithoutExp & 0x7f) << 3
      let exponent = UInt16(exponentBuffer[i])
      decodedAsInts[i] = (signBit << 15) | (exponent << 10) | mantissa
    }

    return 1
  }

#if canImport(Compression)
  import Compression

  func zip(
    data: UnsafeRawPointer, dataSize: Int, zippedData: UnsafeMutableRawPointer,
    zippedDataSize: UnsafeMutablePointer<Int>
  ) -> Bool {
    let outputSize = compression_encode_buffer(
      zippedData.assumingMemoryBound(to: UInt8.self), zippedDataSize[0],
      data.assumingMemoryBound(to: UInt8.self), dataSize, nil, COMPRESSION_ZLIB)
    guard outputSize > 0 else { return false }
    zippedDataSize[0] = outputSize
    return true
  }

  private func unzip(
    data: UnsafeRawPointer, dataSize: Int, unzippedData: UnsafeMutableRawPointer,
    unzippedDataSize: UnsafeMutablePointer<Int>
  ) -> Bool {
    let nextIn = data.assumingMemoryBound(to: UInt8.self)
    let nextOut = unzippedData.assumingMemoryBound(to: UInt8.self)
    var stream = compression_stream(
      dst_ptr: nextOut, dst_size: unzippedDataSize[0], src_ptr: nextIn, src_size: dataSize,
      state: nil)
    var status = compression_stream_init(&stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB)
    guard status != COMPRESSION_STATUS_ERROR else { return false }
    defer { compression_stream_destroy(&stream) }
    stream.src_ptr = nextIn
    stream.src_size = dataSize
    stream.dst_ptr = nextOut
    stream.dst_size = unzippedDataSize[0]
    repeat {
      status = compression_stream_process(&stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))
      guard status != COMPRESSION_STATUS_ERROR else { return false }
    } while status == COMPRESSION_STATUS_OK && stream.dst_size > 0
    unzippedDataSize[0] = unzippedDataSize[0] - stream.dst_size
    return true
  }

#else

  private func zip(
    data: UnsafeRawPointer, dataSize: Int, zippedData: UnsafeMutableRawPointer,
    zippedDataSize: UnsafeMutablePointer<Int>
  ) -> Bool {
    var stream = z_stream()
    let streamSize = Int32(MemoryLayout<z_stream>.size)
    let result = deflateInit2_(
      &stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 9, Z_DEFAULT_STRATEGY, ZLIB_VERSION,
      streamSize)
    defer { deflateEnd(&stream) }
    guard result == Z_OK else { return false }
    let chunkSize = 0x8000_0000
    var availableSize = dataSize
    var outputSize = 0
    var availableOutputSize = zippedDataSize[0]
    var flush = Z_NO_FLUSH
    var nextIn = UnsafeMutablePointer<UInt8>(mutating: data.assumingMemoryBound(to: UInt8.self))
    var nextOut = zippedData.assumingMemoryBound(to: UInt8.self)
    repeat {
      let bufferInputSize = availableSize > chunkSize ? chunkSize : availableSize
      stream.next_in = nextIn
      stream.avail_in = UInt32(bufferInputSize)
      flush = availableSize > chunkSize ? Z_NO_FLUSH : Z_FINISH
      repeat {
        stream.next_out = nextOut
        let bufferOutputSize = availableOutputSize > chunkSize ? chunkSize : availableOutputSize
        stream.avail_out = UInt32(bufferOutputSize)
        guard deflate(&stream, flush) >= Z_OK else { return false }
        let thisOutputSize = bufferOutputSize - Int(stream.avail_out)
        nextOut = nextOut.advanced(by: thisOutputSize)
        outputSize += thisOutputSize
        availableOutputSize -= thisOutputSize
      } while stream.avail_out == 0
      nextIn = nextIn.advanced(by: bufferInputSize)
      availableSize -= bufferInputSize
    } while flush != Z_FINISH
    zippedDataSize[0] = outputSize
    return true
  }

  private func unzip(
    data: UnsafeRawPointer, dataSize: Int, unzippedData: UnsafeMutableRawPointer,
    unzippedDataSize: UnsafeMutablePointer<Int>
  ) -> Bool {
    var stream = z_stream()
    let streamSize = Int32(MemoryLayout<z_stream>.size)
    var result = inflateInit2_(&stream, -MAX_WBITS, ZLIB_VERSION, streamSize)
    defer { inflateEnd(&stream) }
    guard result == Z_OK else { return false }
    let chunkSize = 0x8000_0000
    var availableSize = dataSize
    var outputSize = 0
    var availableOutputSize = unzippedDataSize[0]
    var nextIn = UnsafeMutablePointer<UInt8>(mutating: data.assumingMemoryBound(to: UInt8.self))
    var nextOut = unzippedData.assumingMemoryBound(to: UInt8.self)
    repeat {
      let bufferInputSize = availableSize > chunkSize ? chunkSize : availableSize
      stream.next_in = nextIn
      stream.avail_in = UInt32(bufferInputSize)
      repeat {
        stream.next_out = nextOut
        let bufferOutputSize = availableOutputSize > chunkSize ? chunkSize : availableOutputSize
        stream.avail_out = UInt32(bufferOutputSize)
        result = inflate(&stream, Z_NO_FLUSH)
        guard result != Z_NEED_DICT && result != Z_DATA_ERROR && result != Z_MEM_ERROR else {
          return false
        }
        let thisOutputSize = bufferOutputSize - Int(stream.avail_out)
        nextOut = nextOut.advanced(by: thisOutputSize)
        outputSize += thisOutputSize
        availableOutputSize -= thisOutputSize
      } while stream.avail_out == 0 && availableOutputSize > 0
      nextIn = nextIn.advanced(by: bufferInputSize)
      availableSize -= bufferInputSize
    } while result != Z_STREAM_END && availableOutputSize > 0
    return true
  }
#endif

private let q4pAndEzm7Encode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard
      q4pEncode(
        data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
        identifier) == 0
    else { return 1 }
    return ezm7Encode(
      data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
      identifier)
  }

private let q5pAndEzm7Encode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard
      q5pEncode(
        data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
        identifier) == 0
    else { return 1 }
    return ezm7Encode(
      data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
      identifier)
  }

private let q6pAndEzm7Encode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard
      q6pEncode(
        data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
        identifier) == 0
    else { return 1 }
    return ezm7Encode(
      data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
      identifier)
  }

private let q7pAndEzm7Encode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard
      q7pEncode(
        data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
        identifier) == 0
    else { return 1 }
    return ezm7Encode(
      data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
      identifier)
  }

private let q8pAndEzm7Encode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard
      q8pEncode(
        data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
        identifier) == 0
    else { return 1 }
    return ezm7Encode(
      data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
      identifier)
  }

private let fpzipAndZipEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    // Floating point to use fpzip
    if dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F) {
      return fpzipEncode(
        data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
        identifier)
    }
    return zipEncode(
      data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
      identifier)
  }

private let uDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    switch identifier {
    case 0xf7217:
      return fpzipDecode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    case 0x217:
      return zipDecode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    case 0x511:
      return ezm7Decode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    case 0x8a1e4b:
      return q4pDecode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    case 0x8a1e5b:
      return q5pDecode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    case 0x8a1e6b:
      return q6pDecode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    case 0x8a1e7b:
      return q7pDecode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    case 0x8a1e8b:
      return q8pDecode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    default:
      return 0
    }
  }

extension DynamicGraph {

  final class _Store {
    let sqlite: UnsafeMutableRawPointer
    let flags: Store.OpenFlag
    init(sqlite: OpaquePointer, flags: Store.OpenFlag) {
      self.sqlite = UnsafeMutableRawPointer(sqlite)
      self.flags = flags
    }
    deinit {
      // If the database is opened with WAL mode, this makes sure everything write back to the main
      // database, much easier to operate without worrying the data left in the wal log.
      if flags.contains(.truncateWhenClose) {
        sqlite3_wal_checkpoint_v2(OpaquePointer(sqlite), nil, SQLITE_CHECKPOINT_TRUNCATE, nil, nil)
      }
      sqlite3_close(OpaquePointer(sqlite))
    }
  }

  /**
   * A key-value based parameter store.
   */
  public struct Store {
    public struct OpenFlag: OptionSet {
      public let rawValue: Int
      public init(rawValue: Int) {
        self.rawValue = rawValue
      }
      public static let truncateWhenClose = OpenFlag(rawValue: 1 << 0)
      public static let readOnly = OpenFlag(rawValue: 1 << 1)
    }
    public struct Codec: OptionSet {
      public let rawValue: Int
      public init(rawValue: Int) {
        self.rawValue = rawValue
      }
      public static let fpzip = Codec(rawValue: 1 << 0)
      public static let zip = Codec(rawValue: 1 << 1)
      public static let ezm7 = Codec(rawValue: 1 << 2)
      public static let q4p = Codec(rawValue: 1 << 3)
      public static let q5p = Codec(rawValue: 1 << 4)
      public static let q6p = Codec(rawValue: 1 << 5)
      public static let q7p = Codec(rawValue: 1 << 6)
      public static let q8p = Codec(rawValue: 1 << 7)
      var encode:
        (
          @convention(c) (
            UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
            UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
          ) -> Int32
        )?
      {
        if contains(.ezm7) && contains(.q4p) {
          return q4pAndEzm7Encode  // Prefer q4p, if it is longer (because 16 palette), use ezm7.
        } else if contains(.ezm7) && contains(.q5p) {
          return q5pAndEzm7Encode  // Prefer q5p, if it is longer (because 32 palette), use ezm7.
        } else if contains(.ezm7) && contains(.q6p) {
          return q6pAndEzm7Encode  // Prefer q6p, if it is longer (because 64 palette), use ezm7.
        } else if contains(.ezm7) && contains(.q7p) {
          return q7pAndEzm7Encode  // Prefer q7p, if it is longer (because 256 palette), use ezm7.
        } else if contains(.ezm7) && contains(.q8p) {
          return q8pAndEzm7Encode  // Prefer q8p, if it is longer (because 256 palette), use ezm7.
        } else if contains(.ezm7) {
          // .ezm7 is not supported with other lossless formats
          guard self == .ezm7 else { return nil }
          return ezm7Encode
        } else if contains(.q4p) {
          // .q4p is not supported with other lossless formats
          guard self == .q4p else { return nil }
          return q4pEncode
        } else if contains(.q5p) {
          // .q5p is not supported with other lossless formats
          guard self == .q5p else { return nil }
          return q5pEncode
        } else if contains(.q6p) {
          // .q6p is not supported with other lossless formats
          guard self == .q6p else { return nil }
          return q6pEncode
        } else if contains(.q7p) {
          // .q7p is not supported with other lossless formats
          guard self == .q7p else { return nil }
          return q7pEncode
        } else if contains(.q8p) {
          // .q8p is not supported with other lossless formats
          guard self == .q8p else { return nil }
          return q8pEncode
        } else if contains(.fpzip) && contains(.zip) {
          return fpzipAndZipEncode
        } else if contains(.fpzip) {
          return fpzipEncode
        } else if contains(.zip) {
          return zipEncode
        }
        return nil
      }
      var decode:
        (
          @convention(c) (
            UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32,
            UnsafeMutableRawPointer?, UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
          ) -> Int32
        )?
      {
        guard !isEmpty else {
          return nil
        }
        switch self {
        case .ezm7:
          return ezm7Decode
        case .q4p:
          return q4pDecode
        case .q5p:
          return q5pDecode
        case .q6p:
          return q6pDecode
        case .q7p:
          return q7pDecode
        case .q8p:
          return q8pDecode
        case .fpzip:
          return fpzipDecode
        case .zip:
          return zipDecode
        default:
          return uDecode
        }
      }
    }
    private let graph: DynamicGraph
    private let store: _Store

    /**
     * Read a type-erased tensor from the store.
     *
     * - Parameter key: The key corresponding to that particular tensor.
     */
    public func read(_ key: String, codec: Codec = []) -> NNC.AnyTensor? {
      var underlying: UnsafeMutablePointer<ccv_nnc_tensor_t>? = nil
      let result: Int32
      if codec.isEmpty {
        result = ccv_nnc_tensor_read(store.sqlite, key, nil, nil, &underlying)
      } else {
        var option = ccv_nnc_tensor_io_option_t()
        option.decode = codec.decode
        result = ccv_nnc_tensor_read(store.sqlite, key, nil, &option, &underlying)
      }
      guard result == CCV_IO_FINAL else { return nil }
      let anyTensor = AnyTensorStorage(underlying!)
      return anyTensor.toAnyTensor()
    }

    /**
     * Retrieve codec for a particular key. It must be a tensor to make sense of this.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular tensor.
     * - Returns the codec, if the tensor doesn't exist, return nil.
     */
    public func codec(for key: String) -> Codec? {
      var selectCodec: OpaquePointer? = nil
      sqlite3_prepare_v2(
        OpaquePointer(store.sqlite), "SELECT type FROM tensors WHERE name=?1", -1, &selectCodec, nil
      )
      let SQLITE_TRANSIENT = unsafeBitCast(
        OpaquePointer(bitPattern: -1), to: sqlite3_destructor_type.self)
      sqlite3_bind_text(selectCodec, 1, key, -1, SQLITE_TRANSIENT)
      let codec: Codec?
      if sqlite3_step(selectCodec) == SQLITE_ROW {
        let type = sqlite3_column_int64(selectCodec, 0)
        let identifier = (type >> 32) & 0xffff_ffff
        switch identifier {
        case 0x217:
          codec = .zip
        case 0xf7217:
          codec = .fpzip
        case 0x511:
          codec = .ezm7
        case 0x8a1e4b:
          codec = .q4p
        case 0x8a1e5b:
          codec = .q5p
        case 0x8a1e6b:
          codec = .q6p
        case 0x8a1e7b:
          codec = .q7p
        case 0x8a1e8b:
          codec = .q8p
        default:
          codec = []
        }
      } else {
        codec = nil
      }
      sqlite3_finalize(selectCodec)
      return codec
    }

    /**
     * Read a tensor from the store into tensor variable from dynamic graph.
     *
     * - Parameters:
     *   - key: The key corresponding to that particular tensor.
     *   - variable: The tensor variable to be initialized with.
     * - Returns whether we successfully initialized the tensor variable.
     */
    @discardableResult
    public func read(_ key: String, variable: DynamicGraph_Any, codec: Codec = []) -> Bool {
      switch variable {
      case let tensor as DynamicGraph.AnyTensor:
        assert(tensor.graph === graph)
        let _graph = graph.cGraph
        let _tensor = tensor._tensor
        let raw = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)
        if raw != nil {
          var underlying = raw
          let result: Int32
          if codec.isEmpty {
            result = ccv_nnc_tensor_read(store.sqlite, key, nil, nil, &underlying)
          } else {
            var option = ccv_nnc_tensor_io_option_t()
            option.decode = codec.decode
            result = ccv_nnc_tensor_read(store.sqlite, key, nil, &option, &underlying)
          }
          if result == CCV_IO_FINAL {
            assert(underlying == raw)
          }
          return result == CCV_IO_FINAL
        }
        var underlying: UnsafeMutablePointer<ccv_nnc_tensor_t>? = nil
        let result: Int32
        if codec.isEmpty {
          result = ccv_nnc_tensor_read(store.sqlite, key, nil, nil, &underlying)
        } else {
          var option = ccv_nnc_tensor_io_option_t()
          option.decode = codec.decode
          result = ccv_nnc_tensor_read(store.sqlite, key, nil, &option, &underlying)
        }
        guard result == CCV_IO_FINAL else { return false }
        let anyTensor = AnyTensorStorage(underlying!)
        ccv_nnc_tensor_variable_set(_graph, _tensor, underlying)
        // Retain the tensor until we freed the variable.
        ccv_nnc_tensor_variable_destructor_hook(
          _graph, _tensor,
          { _, _, ctx in
            // No longer need to retain the tensor.
            Unmanaged<NNC.AnyTensorStorage>.fromOpaque(ctx!).release()
          }, Unmanaged.passRetained(anyTensor).toOpaque())
      case let group as DynamicGraph.AnyGroup:
        for (i, tensor) in group.untyped.enumerated() {
          guard read("\(key)(\(i))", variable: tensor) else {
            return false
          }
        }
      default:
        fatalError("Cannot recognize the variable")
      }
      return true
    }
    public enum ModelReaderResult {
      /// Continue to load parameter with the given name.
      case `continue`(String)
      /// The parameter is loaded, no futher operation need.
      case final(NNC.AnyTensor)
      /// Nothing is loaded.
      case fail
    }
    class ModelReaderHelper {
      let reader: (String, DataType, TensorFormat, TensorShape) -> ModelReaderResult
      let sqlite: UnsafeMutableRawPointer
      init(
        reader: @escaping (String, DataType, TensorFormat, TensorShape) -> ModelReaderResult,
        sqlite: UnsafeMutableRawPointer
      ) {
        self.reader = reader
        self.sqlite = sqlite
      }
    }
    /**
     * Read parameters into a given model.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model.
     *   - model: The model to be initialized with parameters from a given key.
     *   - reader: You can customize your reader to load parameter with a different name etc.
     */
    public func read(
      _ key: String, model: Model, codec: Codec = [],
      reader: ((String, DataType, TensorFormat, TensorShape) -> ModelReaderResult)? = nil
    ) {
      guard let reader = reader else {
        if codec.isEmpty {
          ccv_cnnp_model_read(store.sqlite, key, nil, model.cModel)
        } else {
          var option = ccv_nnc_tensor_io_option_t()
          option.decode = codec.decode
          ccv_cnnp_model_read(store.sqlite, key, &option, model.cModel)
        }
        return
      }
      let readerHelper = ModelReaderHelper(reader: reader, sqlite: store.sqlite)
      ccv_cnnp_model_set_io(
        model.cModel,
        { (handle, name, dir, options, tensorOut) -> Int32 in
          let readerHelper = Unmanaged<ModelReaderHelper>.fromOpaque(handle!).takeUnretainedValue()
          let cTensorOut = tensorOut!.pointee
          let params = cTensorOut!.pointee.info
          let result = readerHelper.reader(
            name.map { String(cString: $0) } ?? "", DataType.from(cTensorParams: params),
            TensorFormat.from(cTensorParams: params), TensorShape(dims: params.dim))
          switch result {
          case .final(let tensor):
            precondition(tensor.kind == .CPU)
            let dataSize = ccv_nnc_tensor_data_size(tensor.cTensor.pointee.info)
            ccv_nnc_tensor_swap(cTensorOut, name, dir, tensor.cTensor.pointee.data.ptr, dataSize)
            return Int32(CCV_IO_FINAL)
          case .continue(let name):
            return ccv_nnc_tensor_read(readerHelper.sqlite, name, dir, options, tensorOut)
          case .fail:
            return Int32(CCV_IO_ERROR)
          }
        }, nil)
      let unmanaged = Unmanaged.passRetained(readerHelper)
      if codec.isEmpty {
        ccv_cnnp_model_read(unmanaged.toOpaque(), key, nil, model.cModel)
      } else {
        var option = ccv_nnc_tensor_io_option_t()
        option.decode = codec.decode
        ccv_cnnp_model_read(unmanaged.toOpaque(), key, &option, model.cModel)
      }
      ccv_cnnp_model_set_io(model.cModel, nil, nil)
      unmanaged.release()
    }
    /**
     * Read parameters into a given model builder.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model.
     *   - model: The model builder to be initialized with parameters from a given key.
     *   - reader: You can customize your reader to load parameter with a different name etc.
     */
    public func read(
      _ key: String, model: AnyModelBuilder, codec: Codec = [],
      reader: ((String, DataType, TensorFormat, TensorShape) -> ModelReaderResult)? = nil
    ) {
      model.read(key, from: store, codec: codec, reader: reader)
    }

    /**
     * Write a tensor to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular tensor.
     *   - tensor: The tensor to be persisted.
     */
    public func write(_ key: String, tensor: NNC.AnyTensor, codec: Codec = []) {
      if codec.isEmpty {
        ccv_nnc_tensor_write(tensor.cTensor, store.sqlite, key, nil)
      } else {
        var option = ccv_nnc_tensor_io_option_t()
        option.encode = codec.encode
        ccv_nnc_tensor_write(tensor.cTensor, store.sqlite, key, &option)
      }
    }
    /**
     * Write a tensor variable to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular tensor.
     *   - variable: The tensor variable to be persisted.
     */
    public func write(_ key: String, variable: DynamicGraph_Any, codec: Codec = []) {
      switch variable {
      case let tensor as DynamicGraph.AnyTensor:
        assert(tensor.graph === graph)
        let _graph = graph.cGraph
        let _tensor = tensor._tensor
        let raw = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)!
        if codec.isEmpty {
          ccv_nnc_tensor_write(raw, store.sqlite, key, nil)
        } else {
          var option = ccv_nnc_tensor_io_option_t()
          option.encode = codec.encode
          ccv_nnc_tensor_write(raw, store.sqlite, key, &option)
        }
      case let group as DynamicGraph.AnyGroup:
        for (i, tensor) in group.untyped.enumerated() {
          write("\(key)(\(i))", variable: tensor)
        }
      default:
        fatalError("Cannot recognize the variable")
      }
    }
    /**
     * Write a model to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model.
     *   - model: The model where its parameters to be persisted.
     */
    public func write(_ key: String, model: Model, codec: Codec = []) {
      if codec.isEmpty {
        ccv_cnnp_model_write(model.cModel, store.sqlite, key, nil)
      } else {
        var option = ccv_nnc_tensor_io_option_t()
        option.encode = codec.encode
        ccv_cnnp_model_write(model.cModel, store.sqlite, key, &option)
      }
    }
    /**
     * Write a model builder to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model builder.
     *   - model builder: The model where its parameters to be persisted.
     */
    public func write(_ key: String, model: AnyModelBuilder, codec: Codec = []) {
      write(key, model: model.model!)
    }

    init(_ store: _Store, graph: DynamicGraph) {
      self.store = store
      self.graph = graph
    }

    /**
     * Retrieve a list of all tensors in this file. This reads from the disk
     * and could take some time to finish.
     */
    public var keys: [String] {
      var stmt: OpaquePointer? = nil
      sqlite3_prepare_v2(OpaquePointer(store.sqlite), "SELECT name FROM tensors", -1, &stmt, nil)
      guard let stmt = stmt else { return [] }
      var keys = [String]()
      while SQLITE_ROW == sqlite3_step(stmt) {
        guard let cString = sqlite3_column_text(stmt, 0) else { continue }
        keys.append(String(cString: cString))
      }
      sqlite3_finalize(stmt)
      return keys
    }

    /**
     * Remove one tensor by its key.
     */
    public func remove(_ key: String) {
      var deletion: OpaquePointer? = nil
      sqlite3_prepare_v2(
        OpaquePointer(store.sqlite), "DELETE FROM tensors WHERE name=?1", -1, &deletion, nil)
      let SQLITE_TRANSIENT = unsafeBitCast(
        OpaquePointer(bitPattern: -1), to: sqlite3_destructor_type.self)
      sqlite3_bind_text(deletion, 1, key, -1, SQLITE_TRANSIENT)
      sqlite3_step(deletion)
      sqlite3_finalize(deletion)
    }

    /**
     * Remove all tensors from the store. This also vacuums the store to minimize its size.
     */
    public func removeAll() {
      sqlite3_exec(OpaquePointer(store.sqlite), "DELETE FROM tensors", nil, nil, nil)
      sqlite3_exec(OpaquePointer(store.sqlite), "VACUUM", nil, nil, nil)
    }

  }

  /**
   * Open the store from a file.
   *
   * - Parameters:
   *   - filePath: The file path for the store.
   *   - flags: The flags for the opening store. Default to truncateWhenClose.
   *   - procedure: When the store is open, you can access it from this closure.
   * - Returns: Wether this store can be successfully open or not.
   */
  @discardableResult
  public func openStore(
    _ filePath: String, flags: Store.OpenFlag = .truncateWhenClose,
    procedure: (_ store: Store) throws -> Void
  ) rethrows -> Bool {
    var _sqlite: OpaquePointer? = nil
    if flags.contains(.readOnly) {
      if sqlite3_libversion_number() >= 3_022_000 {
        sqlite3_open_v2(filePath, &_sqlite, SQLITE_OPEN_READONLY, nil)
      } else {  // At least if it is readOnly, we won't create the file.
        sqlite3_open_v2(filePath, &_sqlite, SQLITE_OPEN_READWRITE, nil)
      }
    } else {
      sqlite3_open_v2(filePath, &_sqlite, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil)
    }
    guard let sqlite = _sqlite else { return false }
    sqlite3_busy_timeout(sqlite, 30_000)  // This is essential to have real-world usages.
    let store = Store(_Store(sqlite: sqlite, flags: flags), graph: self)
    try procedure(store)
    return true
  }

}
