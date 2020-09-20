import C_nnc

public enum DeviceKind {
  case CPU
  case GPU(Int)
}

public enum TensorFormat {
  case NHWC
  case NCHW
  case CHWN
}

public enum TensorDimensionFormat {
  case NHWC(Int, Int, Int, Int)
  case NCHW(Int, Int, Int, Int)
  case CHWN(Int, Int, Int, Int)
}

public enum DataType {
  case Float64
  case Int64
  case Float32
  case Int32
  case Float16
  case UInt8
}

public protocol TensorNumeric: Numeric {
  static var dataType: DataType { get }
}

extension Float64: TensorNumeric {
  public static var dataType: DataType { .Float64 }
}

extension Int64: TensorNumeric {
  public static var dataType: DataType { .Int64 }
}

extension Float32: TensorNumeric {
  public static var dataType: DataType { .Float32 }
}

extension Int32: TensorNumeric {
  public static var dataType: DataType { .Int32 }
}

extension Float16: TensorNumeric {
  public static var dataType: DataType { .Float16 }
}

extension UInt8: TensorNumeric {
  public static var dataType: DataType { .UInt8 }
}

public final class Tensor <Element: TensorNumeric> {

  let _tensor: UnsafeMutablePointer<ccv_nnc_tensor_t>
  private let owned: Bool

  init(_ tensor: UnsafeMutablePointer<ccv_nnc_tensor_t>, owned: Bool = true) {
    self.owned = owned
    _tensor = tensor
  }

  private init(_ kind: DeviceKind, dataType: DataType, format: TensorFormat, dimensions: [Int]) {
    owned = true
    _tensor = ccv_nnc_tensor_new(nil,
      toCTensorParams(kind, dataType: dataType, format: format, dimensions: dimensions),
      0)!
  }

  private convenience init(_ kind: DeviceKind, _ dataType: DataType, _ dimensionFormat: TensorDimensionFormat) {
    switch dimensionFormat {
    case let .NHWC(n, h, w, c):
      self.init(kind, dataType: dataType, format: .NHWC, dimensions: [n, h, w, c])
    case let .NCHW(n, c, h, w):
      self.init(kind, dataType: dataType, format: .NCHW, dimensions: [n, c, h, w])
    case let .CHWN(c, h, w, n):
      self.init(kind, dataType: dataType, format: .CHWN, dimensions: [c, h, w, n])
    }
  }

  public convenience init(_ kind: DeviceKind, format: TensorFormat, dimensions: [Int]) {
    self.init(kind, dataType: Element.dataType, format: format, dimensions: dimensions)
  }

  public convenience init(_ kind: DeviceKind, _ dimensionFormat: TensorDimensionFormat) {
    self.init(kind, Element.dataType, dimensionFormat)
  }

  deinit {
    if owned {
      ccv_nnc_tensor_free(_tensor)
    }
  }
}

func toCTensorParams(_ kind: DeviceKind, dataType: DataType, format: TensorFormat, dimensions: [Int]) -> ccv_nnc_tensor_param_t {
  var params = ccv_nnc_tensor_param_t()
  switch kind {
    case .CPU:
      params.type = Int32(CCV_TENSOR_CPU_MEMORY)
    case .GPU(let ordinal):
      params.type = Int32(CCV_TENSOR_GPU_MEMORY | (ordinal << 8))
  }
  switch dataType {
    case .Float64:
      params.datatype = Int32(CCV_64F)
    case .Int64:
      params.datatype = Int32(CCV_64S)
    case .Float32:
      params.datatype = Int32(CCV_32F)
    case .Int32:
      params.datatype = Int32(CCV_32S)
    case .Float16:
      params.datatype = Int32(CCV_16F)
    case .UInt8:
      params.datatype = Int32(CCV_8U)
  }
  switch format {
    case .NHWC:
      params.format = Int32(CCV_TENSOR_FORMAT_NHWC)
    case .NCHW:
      params.format = Int32(CCV_TENSOR_FORMAT_NCHW)
    case .CHWN:
      params.format = Int32(CCV_TENSOR_FORMAT_CHWN)
  }
  assert(dimensions.count <= CCV_NNC_MAX_DIM_ALLOC)
  switch dimensions.count {
    case 1:
      params.dim = (Int32(dimensions[0]), 0, 0, 0, 0, 0, 0, 0)
    case 2:
      params.dim = (Int32(dimensions[0]), Int32(dimensions[1]), 0, 0, 0, 0, 0, 0)
    case 3:
      params.dim = (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), 0, 0, 0, 0, 0)
    case 4:
      params.dim = (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), 0, 0, 0, 0)
    case 5:
      params.dim = (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), Int32(dimensions[4]), 0, 0, 0)
    case 6:
      params.dim = (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), Int32(dimensions[4]), Int32(dimensions[5]), 0, 0)
    case 7:
      params.dim = (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), 0)
    case 8:
      params.dim = (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]))
    default:
      break
  }
  return params
}
