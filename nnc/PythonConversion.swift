import C_nnc
import NNC
import PythonKit

private let np = Python.import("numpy")
private let ctypes = Python.import("ctypes")

extension Tensor where Element: NumpyScalarCompatible {
  /**
   * Initialize a tensor from numpy object.
   */
  public init?(numpy numpyArray: PythonObject) {
    // Check if input is a `numpy.ndarray` instance.
    guard Python.isinstance(numpyArray, np.ndarray) == true else {
      return nil
    }
    // Check if the dtype of the `ndarray` is compatible with the `Element`
    // type.
    guard Element.numpyScalarTypes.contains(numpyArray.dtype) else {
      return nil
    }
    let pyShape = numpyArray.__array_interface__["shape"]
    guard let shape = [Int](pyShape) else { return nil }
    precondition(shape.count <= CCV_NNC_MAX_DIM_ALLOC)
    // Make sure that the array is contiguous in memory. This does a copy if
    // the array is not already contiguous in memory.
    let contiguousNumpyArray = np.ascontiguousarray(numpyArray)
    guard
      let ptrVal =
        UInt(contiguousNumpyArray.__array_interface__["data"].tuple2.0)
    else {
      return nil
    }
    guard let pointer = UnsafeMutablePointer<Element>(bitPattern: ptrVal) else {
      fatalError("numpy.ndarray data pointer was nil")
    }
    self.init(
      .CPU, format: .NCHW, dimensions: shape, unsafeMutablePointer: pointer, keepAlive: numpyArray)
  }
}

extension Tensor where Element: NumpyScalarCompatible {
  /**
   * Make a numpy object from a typed tensor.
   */
  public func makeNumpyArray() -> PythonObject {
    precondition(!isTensorView)
    return withUnsafeBytes { bytes in
      let data = ctypes.cast(Int(bitPattern: bytes.baseAddress), ctypes.POINTER(Element.ctype))
      let ndarray = np.ctypeslib.as_array(data, shape: PythonObject(tupleContentsOf: dimensions))
      return np.copy(ndarray)
    }
  }
}
