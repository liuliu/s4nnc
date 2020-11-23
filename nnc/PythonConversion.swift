import PythonKit
import NNC
import C_nnc

private let np = Python.import("numpy")
private let ctypes = Python.import("ctypes")

extension Tensor where Element: NumpyScalarCompatible {
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
    guard let shape = Array<Int>(pyShape) else { return nil }
    // Make sure that the array is contiguous in memory. This does a copy if
    // the array is not already contiguous in memory.
    let contiguousNumpyArray = np.ascontiguousarray(numpyArray)
    guard let ptrVal =
        UInt(contiguousNumpyArray.__array_interface__["data"].tuple2.0) else {
            return nil
    }
    guard let ptr = UnsafePointer<Element>(bitPattern: ptrVal) else {
      fatalError("numpy.ndarray data pointer was nil")
    }
    let underlying = ccv_nnc_tensor_new(ptr,
      toCTensorParams(.CPU, dataType: Element.dataType, format: .NCHW, dimensions: shape),
      0)!
    let anyTensor = _AnyTensor(underlying, original: numpyArray)
    self.init(anyTensor)
  }
}

public extension Tensor where Element: NumpyScalarCompatible {
  func makeNumpyArray() -> PythonObject {
    precondition(!isTensorView)
    return withUnsafeBytes { bytes in
      let data = ctypes.cast(Int(bitPattern: bytes.baseAddress), ctypes.POINTER(Element.ctype))
      let ndarray = np.ctypeslib.as_array(data, shape: PythonObject(tupleContentsOf: dimensions))
      return np.copy(ndarray)
    }
  }
}
