import C_nnc
import NNC
import PythonKit

private let np = Python.import("numpy")
private let ctypes = Python.import("ctypes")

extension Tensor where Element: NumpyScalarCompatible {

  /// Cannot create a tensor from numpy array.
  public enum NumpyScalarCompatibleError: Error {
    /// The PythonObject is not a numpy array.
    case notNumpy
    /// The numpy array type doesn't match the expected output type.
    case noDataConversion(PythonObject, Any.Type)
    /// Cannot find shape information from numpy array.
    case noShape
    /// Cannot find data pointer from the numpy array.
    case noPointer
  }
  /**
   * Initialize a tensor from numpy object. This doesn't copy the data over, rather, we simply
   * keep the original numpyArray alive. That's also why we don't support any data conversion.
   */
  public init(numpy numpyArray: PythonObject) throws {
    // Check if input is a `numpy.ndarray` instance.
    guard Python.isinstance(numpyArray, np.ndarray) == true else {
      throw NumpyScalarCompatibleError.notNumpy
    }
    // Check if the dtype of the `ndarray` is compatible with the `Element`
    // type.
    guard Element.numpyScalarTypes.contains(numpyArray.dtype) else {
      throw NumpyScalarCompatibleError.noDataConversion(numpyArray.dtype, Element.self)
    }
    let pyShape = numpyArray.__array_interface__["shape"]
    guard let shape = [Int](pyShape) else {
      throw NumpyScalarCompatibleError.noShape
    }
    precondition(shape.count <= CCV_NNC_MAX_DIM_ALLOC)
    // Make sure that the array is contiguous in memory. This does a copy if
    // the array is not already contiguous in memory.
    let contiguousNumpyArray = np.ascontiguousarray(numpyArray)
    guard
      let ptrVal =
        UInt(contiguousNumpyArray.__array_interface__["data"].tuple2.0)
    else {
      throw NumpyScalarCompatibleError.noPointer
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

extension Tensor: PythonConvertible where Element: NumpyScalarCompatible {
  public var pythonObject: PythonObject {
    makeNumpyArray()
  }
}
