import C_nnc

public enum DeviceKind {
  case CPU
  case GPU(Int)

  static func from(cTensorParams: ccv_nnc_tensor_param_t) -> DeviceKind {
    let type = Int(cTensorParams.type)
    if (type & CCV_TENSOR_CPU_MEMORY) == CCV_TENSOR_CPU_MEMORY {
      return .CPU
    } else {
      assert((type & CCV_TENSOR_GPU_MEMORY) == CCV_TENSOR_GPU_MEMORY)
      let ordinal = (type & 0xfff00) >> 8
      return .GPU(ordinal)
    }
  }
}

public enum TensorFormat {
  case NHWC
  case NCHW
  case CHWN
}

public enum TensorDimensionFormat {
  case C(Int) // Assuming NCHW
  case NC(Int, Int) // Assuming NCHW
  case HWC(Int, Int, Int) // Assuming NHWC
  case CHW(Int, Int, Int) // Assuming NCHW
  case NHWC(Int, Int, Int, Int)
  case NCHW(Int, Int, Int, Int)
  case CHWN(Int, Int, Int, Int)

  var format: TensorFormat {
    switch self {
      case .C:
        return .NCHW
      case .NC:
        return .NCHW
      case .HWC:
        return .NHWC
      case .CHW:
        return .NCHW
      case .NHWC:
        return .NHWC
      case .NCHW:
        return .NCHW
      case .CHWN:
        return .CHWN
    }
  }

  var dimensions: [Int] {
    switch self {
      case let .C(c):
        assert(c > 0)
        return [c]
      case let .NC(n, c):
        assert(n > 0)
        assert(c > 0)
        return [n, c]
      case let .HWC(h, w, c):
        assert(h > 0)
        assert(w > 0)
        assert(c > 0)
        return [h, w, c]
      case let .CHW(c, h, w):
        assert(c > 0)
        assert(h > 0)
        assert(w > 0)
        return [c, h, w]
      case let .NHWC(n, h, w, c):
        assert(n > 0)
        assert(h > 0)
        assert(w > 0)
        assert(c > 0)
        return [n, h, w, c]
      case let .NCHW(n, c, h, w):
        assert(n > 0)
        assert(c > 0)
        assert(h > 0)
        assert(w > 0)
        return [n, c, h, w]
      case let .CHWN(c, h, w, n):
        assert(c > 0)
        assert(h > 0)
        assert(w > 0)
        assert(n > 0)
        return [c, h, w, n]
    }
  }
}

public enum DataType {
  case Float64
  case Int64
  case Float32
  case Int32
  case Float16
  case UInt8

  static func from(cTensorParams: ccv_nnc_tensor_param_t) -> DataType {
    switch Int(cTensorParams.datatype) {
      case CCV_64F:
        return .Float64
      case CCV_64S:
        return .Int64
      case CCV_32F:
        return .Float32
      case CCV_32S:
        return .Int32
      case CCV_16F:
        return .Float16
      case CCV_8U:
        return .UInt8
      default:
        fatalError("unspecified datatype")
    }
  }
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

public final class _AnyTensor {
  let _tensor: UnsafeMutablePointer<ccv_nnc_tensor_t>
  fileprivate let original: Any?

  init(_ tensor: UnsafeMutablePointer<ccv_nnc_tensor_t>, original: Any? = nil) {
    self.original = original
    _tensor = tensor
  }

  deinit {
    guard original == nil else { return }
    ccv_nnc_tensor_free(_tensor)
  }
}

public protocol AnyTensor {
  var underlying: _AnyTensor { get }
}

public extension AnyTensor {
  var dataType: DataType {
    DataType.from(cTensorParams: underlying._tensor.pointee.info)
  }

  var kind: DeviceKind {
    DeviceKind.from(cTensorParams: underlying._tensor.pointee.info)
  }

  var dimensions: [Int] {
    let dim = underlying._tensor.pointee.info.dim
    if dim.0 == 0 {
      return []
    } else if dim.1 == 0 {
      return [Int(dim.0)]
    } else if dim.2 == 0 {
      return [Int(dim.0), Int(dim.1)]
    } else if dim.3 == 0 {
      return [Int(dim.0), Int(dim.1), Int(dim.2)]
    } else if dim.4 == 0 {
      return [Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3)]
    } else if dim.5 == 0 {
      return [Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3), Int(dim.4)]
    } else if dim.6 == 0 {
      return [Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3), Int(dim.4), Int(dim.5)]
    } else if dim.7 == 0 {
      return [Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3), Int(dim.4), Int(dim.5), Int(dim.6)]
    } else {
      return [Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3), Int(dim.4), Int(dim.5), Int(dim.6), Int(dim.7)]
    }
  }

  var isTensorView: Bool {
    let type = Int(underlying._tensor.pointee.type)
    return (type & CCV_TENSOR_VIEW) == CCV_TENSOR_VIEW
  }

  var increments: [Int] {
    guard isTensorView else {
      return dimensions
    }
    let inc = UnsafeMutableRawPointer(underlying._tensor).bindMemory(to: ccv_nnc_tensor_view_t.self, capacity: 1).pointee.inc
    if inc.0 == 0 {
      return dimensions
    } else if inc.1 == 0 {
      return [Int(inc.0)]
    } else if inc.2 == 0 {
      return [Int(inc.0), Int(inc.1)]
    } else if inc.3 == 0 {
      return [Int(inc.0), Int(inc.1), Int(inc.2)]
    } else if inc.4 == 0 {
      return [Int(inc.0), Int(inc.1), Int(inc.2), Int(inc.3)]
    } else if inc.5 == 0 {
      return [Int(inc.0), Int(inc.1), Int(inc.2), Int(inc.3), Int(inc.4)]
    } else if inc.6 == 0 {
      return [Int(inc.0), Int(inc.1), Int(inc.2), Int(inc.3), Int(inc.4), Int(inc.5)]
    } else if inc.7 == 0 {
      return [Int(inc.0), Int(inc.1), Int(inc.2), Int(inc.3), Int(inc.4), Int(inc.5), Int(inc.6)]
    } else {
      return [Int(inc.0), Int(inc.1), Int(inc.2), Int(inc.3), Int(inc.4), Int(inc.5), Int(inc.6), Int(inc.7)]
    }
  }
}

public struct Tensor <Element: TensorNumeric>: AnyTensor {

  private var _tensor: _AnyTensor

  public var underlying: _AnyTensor { _tensor }

  private init(_ kind: DeviceKind, dataType: DataType, format: TensorFormat, dimensions: [Int]) {
    let underlying = ccv_nnc_tensor_new(nil,
      toCTensorParams(kind, dataType: dataType, format: format, dimensions: dimensions),
      0)!
    _tensor = _AnyTensor(underlying)
  }

  private init(_ kind: DeviceKind, _ dataType: DataType, _ dimensionFormat: TensorDimensionFormat) {
    self.init(kind, dataType: dataType, format: dimensionFormat.format, dimensions: dimensionFormat.dimensions)
  }

  init(_ tensor: _AnyTensor) {
    _tensor = tensor
  }

  public init(_ tensor: AnyTensor) {
    assert(tensor.dataType == Element.dataType)
    _tensor = tensor.underlying
  }

  public init(_ kind: DeviceKind, format: TensorFormat, dimensions: [Int]) {
    self.init(kind, dataType: Element.dataType, format: format, dimensions: dimensions)
  }

  public init(_ kind: DeviceKind, _ dimensionFormat: TensorDimensionFormat) {
    self.init(kind, Element.dataType, dimensionFormat)
  }

  public subscript(indices: Int...) -> Element {
    get {
      assert(increments.count == indices.count)
      let increments = self.increments
      let count = increments.reduce(1, *)
      let tensor = _tensor._tensor
      let pointer = tensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
      var offset = 0
      for (i, increment) in increments.enumerated() {
        offset *= increment
        offset += indices[i]
      }
      return (pointer + offset).pointee
    }
    set(v) {
      assert(increments.count == indices.count)
      let increments = self.increments
      let count = increments.reduce(1, *)
      if !isKnownUniquelyReferenced(&_tensor) {
        // Make a copy (copy-on-write).
        var input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = _tensor._tensor
        var output = ccv_nnc_tensor_new(nil, _tensor._tensor.pointee.info, 0)
        ccv_nnc_cmd_exec(ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, ccv_nnc_cmd_param_t(), 0),
          ccv_nnc_no_hint, 0, &input, 1, &output, 1, nil)
        _tensor = _AnyTensor(output!)
      }
      // We need to deal with GPU memory.
      let tensor = _tensor._tensor
      let pointer = tensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
      var offset = 0
      for (i, increment) in increments.enumerated() {
        offset *= increment
        offset += indices[i]
      }
      (pointer + offset).pointee = v
    }
  }
}

func toCDimensions(_ dimensions: [Int]?) -> (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) {
  guard let dimensions = dimensions else {
    return (0, 0, 0, 0, 0, 0, 0, 0)
  }
  assert(dimensions.count <= CCV_NNC_MAX_DIM_ALLOC)
  switch dimensions.count {
    case 1:
      return (Int32(dimensions[0]), 0, 0, 0, 0, 0, 0, 0)
    case 2:
      return (Int32(dimensions[0]), Int32(dimensions[1]), 0, 0, 0, 0, 0, 0)
    case 3:
      return (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), 0, 0, 0, 0, 0)
    case 4:
      return (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), 0, 0, 0, 0)
    case 5:
      return (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), Int32(dimensions[4]), 0, 0, 0)
    case 6:
      return (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), Int32(dimensions[4]), Int32(dimensions[5]), 0, 0)
    case 7:
      return (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), 0)
    case 8:
      return (Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]))
    default:
    return (0, 0, 0, 0, 0, 0, 0, 0)
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
  params.dim = toCDimensions(dimensions)
  return params
}
