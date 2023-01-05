import C_nnc
import NNC
#if canImport(lib_nnc_mps_compat) && canImport(CoreML)
import lib_nnc_mps_compat
import CoreML

extension Tensor where Element: MLShapedArrayScalar {
  public init(_ shapedArray: MLShapedArray<Element>) {
    var shapedArray = shapedArray
    let (pointer, shape) = shapedArray.withUnsafeMutableShapedBufferPointer { pointer, shape, _ in
      return (pointer.baseAddress!, shape)
    }
    // All these are fragile, since at this point, there is no guarantee that the pointer
    // will be valid. However, if you are doing everything right, it should be.
    self.init(
      .CPU, format: .NCHW, shape: TensorShape(shape), unsafeMutablePointer: pointer,
      bindLifetimeOf: shapedArray)
  }
}

extension MLShapedArray where Scalar: TensorNumeric {
  public init(_ tensor: Tensor<Scalar>) {
    let cTensor = tensor.cTensor
    switch tensor.kind {
    case .CPU:
      let storage = Unmanaged.passRetained(tensor.storage)
      if tensor.isTensorView {
        self.init(bytesNoCopy: cTensor.pointee.data.u8, shape: Array(tensor.shape), strides: Array(tensor.strides), deallocator: .custom({ _, _ in
          storage.release()
        }))
      } else {
        self.init(bytesNoCopy: cTensor.pointee.data.u8, shape: Array(tensor.shape), deallocator: .custom({ _, _ in
          storage.release()
        }))
      }
    case .GPU(_):
      let buffer = mpgetbuffer(cTensor)!
      let contents = buffer.contents().assumingMemoryBound(to: UInt8.self)
      let unmanaged = Unmanaged.passRetained(buffer)
      if tensor.isTensorView {
        self.init(bytesNoCopy: contents + Int(cTensor.pointee.dataof), shape: Array(tensor.shape), strides: Array(tensor.strides), deallocator: .custom({ _, _ in
          unmanaged.release()
        }))
      } else {
        self.init(bytesNoCopy: contents + Int(cTensor.pointee.dataof), shape: Array(tensor.shape), deallocator: .custom({ _, _ in
          unmanaged.release()
        }))
      }
    }
  }
}
#endif
