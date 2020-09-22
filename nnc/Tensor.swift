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

public class AnyTensor {
  let _tensor: UnsafeMutablePointer<ccv_nnc_tensor_t>
  fileprivate let owner: Any?

  init(_ tensor: UnsafeMutablePointer<ccv_nnc_tensor_t>, owner: Any? = nil) {
    self.owner = owner
    _tensor = tensor
  }

  deinit {
    guard owner == nil else { return }
    ccv_nnc_tensor_free(_tensor)
  }

  public var dataType: DataType {
    switch Int(_tensor.pointee.info.datatype) {
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

  public var kind: DeviceKind {
    let type = Int(_tensor.pointee.info.type)
    if (type & CCV_TENSOR_CPU_MEMORY) == CCV_TENSOR_CPU_MEMORY {
      return .CPU
    } else {
      assert((type & CCV_TENSOR_GPU_MEMORY) == CCV_TENSOR_GPU_MEMORY)
      let ordinal = (type & 0xfff00) >> 8
      return .GPU(ordinal)
    }
  }

  public var dimensions: [Int] {
    let dim = _tensor.pointee.info.dim
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

  public var isTensorView: Bool {
    let type = Int(_tensor.pointee.type)
    return (type & CCV_TENSOR_VIEW) == CCV_TENSOR_VIEW
  }

  public var increments: [Int] {
    guard isTensorView else {
      return dimensions
    }
    let inc = UnsafeMutableRawPointer(_tensor).bindMemory(to: ccv_nnc_tensor_view_t.self, capacity: 1).pointee.inc
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

public final class Tensor <Element: TensorNumeric>: AnyTensor {

  private convenience init(_ kind: DeviceKind, dataType: DataType, format: TensorFormat, dimensions: [Int]) {
    let tensor = ccv_nnc_tensor_new(nil,
      toCTensorParams(kind, dataType: dataType, format: format, dimensions: dimensions),
      0)!
    self.init(tensor)
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

  public convenience init(_ tensor: AnyTensor) {
    assert(tensor.dataType == Element.dataType)
    self.init(tensor._tensor, owner: tensor) // We cannot free it, the owner it is the other tensor.
  }

  public convenience init(_ kind: DeviceKind, format: TensorFormat, dimensions: [Int]) {
    self.init(kind, dataType: Element.dataType, format: format, dimensions: dimensions)
  }

  public convenience init(_ kind: DeviceKind, _ dimensionFormat: TensorDimensionFormat) {
    self.init(kind, Element.dataType, dimensionFormat)
  }

  public subscript(indices: Int...) -> Element {
    get {
      assert(increments.count == indices.count)
      let increments = self.increments
      let count = increments.reduce(1, *)
      let pointer = _tensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
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
      let pointer = _tensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
      var offset = 0
      for (i, increment) in increments.enumerated() {
        offset *= increment
        offset += indices[i]
      }
      (pointer + offset).pointee = v
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
