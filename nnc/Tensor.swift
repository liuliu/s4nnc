import C_nnc
import Foundation

public final class CmdParamsFactory {
  public static let factory = CmdParamsFactory()
  init() {
    ccv_nnc_init()
  }
  public func newParams() -> ccv_nnc_cmd_param_t {
    return ccv_nnc_cmd_param_t()
  }
  func sink() {
  }
}

/// The kind of devices the tensor resides on.
public enum DeviceKind {
  case CPU
  case GPU(Int)

  public static func from(cTensorParams: ccv_nnc_tensor_param_t) -> DeviceKind {
    let type = Int(cTensorParams.type)
    if (type & CCV_TENSOR_CPU_MEMORY) == CCV_TENSOR_CPU_MEMORY {
      return .CPU
    } else {
      assert((type & CCV_TENSOR_GPU_MEMORY) == CCV_TENSOR_GPU_MEMORY)
      let ordinal = (type & 0xfff00) >> 8
      return .GPU(ordinal)
    }
  }

  /**
   * GPU device related information.
   */
  public enum GPUs {
    /**
     * Number of available GPU devices.
     */
    public static var count: Int {
      Int(ccv_nnc_device_count(Int32(CCV_STREAM_CONTEXT_GPU)))
    }

    /**
     * Remap GPU devices. It supports at most 64 devices remapping. Call this method
     * again would clear up the old remap.
     */
    public static func permute(_ ordinals: Int...) {
      Array(ordinals.map { Int32($0) }).withUnsafeBytes {
        ccv_nnc_set_device_permutation(
          Int32(CCV_STREAM_CONTEXT_GPU), $0.baseAddress?.assumingMemoryBound(to: Int32.self),
          Int32(ordinals.count))
      }
    }
  }

  @usableFromInline
  var toC: Int32 {
    switch self {
    case .CPU:
      return Int32(CCV_TENSOR_CPU_MEMORY)
    case .GPU(let ordinal):
      return Int32(CCV_TENSOR_GPU_MEMORY | (ordinal << 8))
    }
  }
}

extension DeviceKind: Equatable {
  public static func == (lhs: Self, rhs: Self) -> Bool {
    if case .CPU = lhs, case .CPU = rhs {
      return true
    } else if case .GPU(let lv) = lhs, case .GPU(let rv) = rhs {
      return lv == rv
    }
    return false
  }
}

/// Tensor arrangements.
public enum TensorFormat {
  case NHWC
  case NCHW
  case CHWN

  public static func from(cTensorParams: ccv_nnc_tensor_param_t) -> TensorFormat {
    switch Int(cTensorParams.format) {
    case CCV_TENSOR_FORMAT_NCHW:
      return .NCHW
    case CCV_TENSOR_FORMAT_NHWC:
      return .NHWC
    case CCV_TENSOR_FORMAT_CHWN:
      return .CHWN
    default:
      fatalError("unspecified format")
    }
  }

  @usableFromInline
  var toC: Int32 {
    switch self {
    case .NHWC:
      return Int32(CCV_TENSOR_FORMAT_NHWC)
    case .NCHW:
      return Int32(CCV_TENSOR_FORMAT_NCHW)
    case .CHWN:
      return Int32(CCV_TENSOR_FORMAT_CHWN)
    }
  }
}

/// Tensor shape (or dimensions)
public struct TensorShape {
  public var dims:
    (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
  public init(
    dims: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
  ) {
    self.dims = dims
  }
}

extension TensorShape: ExpressibleByArrayLiteral {
  public init(_ elements: [Int]) {
    dims = toCDimensions(elements)
  }
  public init(arrayLiteral elements: Int...) {
    dims = toCDimensions(elements)
  }
}

extension TensorShape: CustomStringConvertible {
  public var description: String {
    return Array(self).description
  }
}

extension TensorShape: Equatable {
  public static func == (lhs: Self, rhs: Self) -> Bool {
    return lhs.dims.0 == rhs.dims.0 && lhs.dims.1 == rhs.dims.1 && lhs.dims.2 == rhs.dims.2
      && lhs.dims.3 == rhs.dims.3 && lhs.dims.4 == rhs.dims.4 && lhs.dims.5 == rhs.dims.5
      && lhs.dims.6 == rhs.dims.6 && lhs.dims.7 == rhs.dims.7 && lhs.dims.8 == rhs.dims.8
      && lhs.dims.9 == rhs.dims.9 && lhs.dims.10 == rhs.dims.10 && lhs.dims.11 == rhs.dims.11
  }
}

extension Array where Element == Int {
  public init(_ shape: TensorShape) {
    self = fromCDimensions(shape.dims)
  }
}

extension TensorShape: RandomAccessCollection {
  public typealias Element = Int
  public typealias Index = Int
  public typealias Indices = Range<Index>
  public typealias SubSequence = TensorShape
  public var startIndex: Index { 0 }
  public var endIndex: Index {
    var dims = self.dims
    return Int(
      withUnsafePointer(to: &dims) {
        ccv_nnc_tensor_nd(UnsafeRawPointer($0).assumingMemoryBound(to: Int32.self))
      })
  }
  public var indices: Indices { startIndex..<endIndex }
  public func formIndex(after i: inout Index) { i += 1 }
  public func formIndex(before i: inout Index) { i -= 1 }
  public subscript(position: Index) -> Element {
    get {
      switch position {
      case 0:
        return Int(dims.0)
      case 1:
        return Int(dims.1)
      case 2:
        return Int(dims.2)
      case 3:
        return Int(dims.3)
      case 4:
        return Int(dims.4)
      case 5:
        return Int(dims.5)
      case 6:
        return Int(dims.6)
      case 7:
        return Int(dims.7)
      case 8:
        return Int(dims.8)
      case 9:
        return Int(dims.9)
      case 10:
        return Int(dims.10)
      case 11:
        return Int(dims.11)
      default:
        fatalError("Cannot find index \(position) for dims \(dims)")
      }
    }
    set {
      switch position {
      case 0:
        dims.0 = Int32(newValue)
      case 1:
        dims.1 = Int32(newValue)
      case 2:
        dims.2 = Int32(newValue)
      case 3:
        dims.3 = Int32(newValue)
      case 4:
        dims.4 = Int32(newValue)
      case 5:
        dims.5 = Int32(newValue)
      case 6:
        dims.6 = Int32(newValue)
      case 7:
        dims.7 = Int32(newValue)
      case 8:
        dims.8 = Int32(newValue)
      case 9:
        dims.9 = Int32(newValue)
      case 10:
        dims.10 = Int32(newValue)
      case 11:
        dims.11 = Int32(newValue)
      default:
        fatalError("Cannot find index \(position) for dims \(dims)")
      }
    }
  }
  public subscript(indices: Indices) -> SubSequence {
    get {
      precondition(indices.lowerBound >= 0 && indices.upperBound <= CCV_NNC_MAX_DIM_ALLOC)
      assert(indices.upperBound > indices.lowerBound)
      var shape = self
      withUnsafeMutablePointer(to: &shape.dims) {
        let dims = UnsafeMutableRawPointer($0).assumingMemoryBound(to: Int32.self)
        let length = indices.upperBound - indices.lowerBound
        memmove(dims, dims + indices.lowerBound, MemoryLayout<Int32>.size * length)
        for i in length..<Int(CCV_NNC_MAX_DIM_ALLOC) {
          dims[i] = 0
        }
      }
      return shape
    }
    set {
      precondition(indices.lowerBound >= 0 && indices.upperBound <= CCV_NNC_MAX_DIM_ALLOC)
      assert(indices.upperBound > indices.lowerBound)
      let length = indices.upperBound - indices.lowerBound
      var newValue = newValue
      withUnsafeMutablePointer(to: &dims) { to in
        let to = UnsafeMutableRawPointer(to).assumingMemoryBound(to: Int32.self)
        withUnsafePointer(to: &newValue.dims) { from in
          let from = UnsafeRawPointer(from).assumingMemoryBound(to: Int32.self)
          let _ = memmove(to + indices.lowerBound, from, MemoryLayout<Int32>.size * length)
        }
      }
    }
  }
}

/// Tensor shapes and its arrangements.
public enum TensorShapeFormat {
  case C(Int)  // Assuming NCHW
  case NC(Int, Int)  // Assuming NCHW
  case WC(Int, Int)  // Assuming NHWC
  case HWC(Int, Int, Int)  // Assuming NHWC
  case CHW(Int, Int, Int)  // Assuming NCHW
  case NHWC(Int, Int, Int, Int)
  case NCHW(Int, Int, Int, Int)
  case CHWN(Int, Int, Int, Int)

  @usableFromInline
  var format: TensorFormat {
    switch self {
    case .C:
      return .NCHW
    case .NC:
      return .NCHW
    case .WC:
      return .NHWC
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

  @usableFromInline
  var shape: TensorShape {
    switch self {
    case let .C(c):
      assert(c > 0)
      return [c]
    case let .NC(n, c):
      assert(n > 0)
      assert(c > 0)
      return [n, c]
    case let .WC(w, c):
      assert(w > 0)
      assert(c > 0)
      return [w, c]
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

/// Data types for a given tensor.
public enum DataType {
  case Float64
  case Int64
  case Float32
  case Int32
  case Float16
  case UInt8

  public static func from(cTensorParams: ccv_nnc_tensor_param_t) -> DataType {
    switch Int(cTensorParams.datatype & 0xff000) {
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
    case CCV_QX:
      switch Int((cTensorParams.datatype & 0xff) << 12) {
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
    default:
      fatalError("unspecified datatype")
    }
  }

  @usableFromInline
  var toC: Swift.Int32 {
    switch self {
    case .Float64:
      return Swift.Int32(CCV_64F)
    case .Int64:
      return Swift.Int32(CCV_64S)
    case .Float32:
      return Swift.Int32(CCV_32F)
    case .Int32:
      return Swift.Int32(CCV_32S)
    case .Float16:
      return Swift.Int32(CCV_16F)
    case .UInt8:
      return Swift.Int32(CCV_8U)
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

#if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
  extension Float16: TensorNumeric {
    public static var dataType: DataType { .Float16 }
  }
#else
  private typealias Float16 = UInt16
  extension Float16: TensorNumeric {
    public static var dataType: DataType { .Float16 }
  }
#endif

extension UInt8: TensorNumeric {
  public static var dataType: DataType { .UInt8 }
}

public final class AnyTensorStorage {
  fileprivate let cTensor: UnsafeMutablePointer<ccv_nnc_tensor_t>
  fileprivate let original: Any?
  private let selfOwned: Bool

  init(
    _ cTensor: UnsafeMutablePointer<ccv_nnc_tensor_t>, original: Any? = nil, selfOwned: Bool = true
  ) {
    self.original = original
    self.selfOwned = selfOwned
    self.cTensor = cTensor
  }

  deinit {
    guard selfOwned else { return }
    ccv_nnc_tensor_free(cTensor)
  }

  @usableFromInline
  var dataType: DataType {
    DataType.from(cTensorParams: cTensor.pointee.info)
  }

  @usableFromInline
  func copy() -> AnyTensorStorage {
    var input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = cTensor
    var output = ccv_nnc_tensor_new(nil, cTensor.pointee.info, 0)
    ccv_nnc_cmd_exec(
      ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0),
      ccv_nnc_no_hint, 0, &input, 1, &output, 1, nil)
    return AnyTensorStorage(output!)
  }

  @usableFromInline
  var shape: TensorShape {
    TensorShape(dims: cTensor.pointee.info.dim)
  }

  @usableFromInline
  var strides: TensorShape {
    let type = Int(cTensor.pointee.type)
    guard (type & CCV_TENSOR_VIEW) == CCV_TENSOR_VIEW else {
      var strides = TensorShape(dims: cTensor.pointee.info.dim)
      var stride = 1
      for i in (0..<strides.count).reversed() {
        let oldStride = strides[i]
        strides[i] = stride
        stride *= oldStride
      }
      return strides
    }
    return TensorShape(
      dims: UnsafeMutableRawPointer(cTensor).bindMemory(
        to: ccv_nnc_tensor_view_t.self, capacity: 1
      ).pointee.stride)
  }

  @usableFromInline
  subscript<Element: TensorNumeric>(indices: [Int], type: Element.Type) -> Element {
    get {
      let strides = self.strides
      assert(strides.count == indices.count)
      let count = strides[0] * shape[0]
      let pointer = cTensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
      var offset = 0
      for (i, stride) in strides.enumerated() {
        offset += indices[i] * stride
      }
      return (pointer + offset).pointee
    }
    set(v) {
      let strides = self.strides
      assert(strides.count == indices.count)
      let count = strides[0] * shape[0]
      // We need to deal with GPU memory.
      let pointer = cTensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
      var offset = 0
      for (i, stride) in strides.enumerated() {
        offset += indices[i] * stride
      }
      (pointer + offset).pointee = v
    }
  }
}

extension AnyTensorStorage {
  @usableFromInline
  subscript<Element: TensorNumeric>(ranges: [Range<Int>], type: Element.Type) -> AnyTensorStorage {
    get {
      // This is a restricted form a reshape.
      let cTensorParams = cTensor.pointee.info
      let device = DeviceKind.from(cTensorParams: cTensorParams)
      let format = TensorFormat.from(cTensorParams: cTensorParams)
      let shape = self.shape
      assert(shape.count == ranges.count)
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < shape[i])
        assert(range.upperBound > 0 && range.upperBound <= shape[i])
      }
      let offset = ranges.map { $0.lowerBound }
      let dimensions = ranges.map { $0.count }
      var cOffset = toCDimensions(offset)
      let strides = self.strides
      var cStrides = strides.dims
      let newt = withUnsafePointer(to: &cOffset) {
        cOffset -> UnsafeMutablePointer<ccv_nnc_tensor_view_t> in
        let cOffset = UnsafeRawPointer(cOffset).assumingMemoryBound(to: Int32.self)
        return withUnsafePointer(to: &cStrides) {
          cStrides -> UnsafeMutablePointer<ccv_nnc_tensor_view_t> in
          let cStrides = UnsafeRawPointer(cStrides).assumingMemoryBound(to: Int32.self)
          return ccv_nnc_tensor_view_new(
            cTensor,
            toCTensorParams(
              device, dataType: Element.dataType, format: format, dimensions: dimensions), cOffset,
            cStrides)!
        }
      }
      return newt.withMemoryRebound(to: ccv_nnc_tensor_t.self, capacity: 1) {
        AnyTensorStorage($0, original: self)
      }
    }
    set(v) {
      let cTensorParams = cTensor.pointee.info
      let device = DeviceKind.from(cTensorParams: cTensorParams)
      // Use the format of the input to make sure we don't do unnecessary format conversions.
      let vFormat = TensorFormat.from(cTensorParams: v.cTensor.pointee.info)
      let shape = self.shape
      assert(shape.count == ranges.count)
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < shape[i])
        assert(range.upperBound > 0 && range.upperBound <= shape[i])
      }
      let offset = ranges.map { $0.lowerBound }
      let dimensions = ranges.map { $0.count }
      var cOffset = toCDimensions(offset)
      let strides = self.strides
      var cStrides = strides.dims
      var newt = withUnsafePointer(to: &cOffset) { cOffset -> ccv_nnc_tensor_view_t in
        let cOffset = UnsafeRawPointer(cOffset).assumingMemoryBound(to: Int32.self)
        return withUnsafePointer(to: &cStrides) { cStrides -> ccv_nnc_tensor_view_t in
          let cStrides = UnsafeRawPointer(cStrides).assumingMemoryBound(to: Int32.self)
          return ccv_nnc_tensor_view(
            cTensor,
            toCTensorParams(
              device, dataType: Element.dataType, format: vFormat, dimensions: dimensions), cOffset,
            cStrides)
        }
      }
      let inputDim = fromCDimensions(v.cTensor.pointee.info.dim)
      for (i, dimension) in dimensions.enumerated() {
        assert(dimension == inputDim[i])
      }
      var input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = v.cTensor
      withUnsafeMutablePointer(to: &newt) { newt in
        var output: UnsafeMutablePointer<ccv_nnc_tensor_t>? = UnsafeMutableRawPointer(newt)
          .bindMemory(to: ccv_nnc_tensor_t.self, capacity: 1)
        ccv_nnc_cmd_exec(
          ccv_nnc_cmd(
            CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0),
          ccv_nnc_no_hint, 0, &input, 1, &output, 1, nil)
      }
      withExtendedLifetime(v) {}
    }
  }
}

extension AnyTensorStorage {
  @usableFromInline
  subscript<Element: TensorNumeric>(indices: [Int], range: Range<Int>, type: Element.Type)
    -> AnyTensorStorage
  {
    get {
      // This is a restricted form a reshape.
      let cTensorParams = cTensor.pointee.info
      let device = DeviceKind.from(cTensorParams: cTensorParams)
      let format = TensorFormat.from(cTensorParams: cTensorParams)
      let shape = self.shape
      assert(shape.count == indices.count + 1)
      let dimensions = Array(repeating: 1, count: indices.count) + [range.count]
      let strides = self.strides
      let count = strides[0] * shape[0]
      let pointer = cTensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
      assert(range.lowerBound >= 0 && range.lowerBound < shape[indices.count])
      assert(range.upperBound > 0 && range.upperBound <= shape[indices.count])
      var offset = 0
      if indices.count > 0 {
        for (i, stride) in strides.prefix(indices.count).enumerated() {
          offset += indices[i] * stride
        }
      }
      offset += range.lowerBound * strides[indices.count]
      let newt = ccv_nnc_tensor_new(
        pointer + offset,
        toCTensorParams(device, dataType: Element.dataType, format: format, dimensions: dimensions),
        0)!
      return AnyTensorStorage(newt, original: self)
    }
    set(v) {
      let cTensorParams = cTensor.pointee.info
      let device = DeviceKind.from(cTensorParams: cTensorParams)
      // Use the format of the input to make sure we don't do unnecessary format conversions.
      let vFormat = TensorFormat.from(cTensorParams: v.cTensor.pointee.info)
      assert(device == DeviceKind.from(cTensorParams: v.cTensor.pointee.info))
      let shape = self.shape
      assert(shape.count == indices.count + 1)
      let strides = self.strides
      let count = strides[0] * shape[0]
      let pointer = cTensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
      assert(range.lowerBound >= 0 && range.lowerBound < shape[indices.count])
      assert(range.upperBound > 0 && range.upperBound <= shape[indices.count])
      let inputDim = TensorShape(dims: v.cTensor.pointee.info.dim)
      for d in inputDim {
        assert(d == 1 || d == range.count)
      }
      assert(inputDim.reduce(1, *) == range.count)
      var offset = 0
      if indices.count > 0 {
        for (i, stride) in strides.prefix(indices.count).enumerated() {
          offset += indices[i] * stride
        }
      }
      offset += range.lowerBound * strides[indices.count]
      if case .CPU = device {  // If it is CPU, do direct copy.
        memcpy(
          pointer + offset, v.cTensor.pointee.data.u8, MemoryLayout<Element>.size * range.count)
        return
      }
      // Otherwise, mostly use the source tensor format and do a data transfer.
      var input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = v.cTensor
      var newt = ccv_nnc_tensor(
        pointer + offset,
        toCTensorParams(device, dataType: Element.dataType, format: vFormat, shape: inputDim),
        0)
      withUnsafeMutablePointer(to: &newt) { newt in
        var output: UnsafeMutablePointer<ccv_nnc_tensor_t>? = newt
        ccv_nnc_cmd_exec(
          ccv_nnc_cmd(
            CCV_NNC_DATA_TRANSFER_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0),
          ccv_nnc_no_hint, 0, &input, 1, &output, 1, nil)
      }
      withExtendedLifetime(v) {}
    }
  }
}

/// A type-erased tensor.
public protocol AnyTensor {
  var storage: AnyTensorStorage { get }
  var cTensor: UnsafeMutablePointer<ccv_nnc_tensor_t> { get }
  init(from: AnyTensor)
}

extension AnyTensor {
  @inlinable
  public var dataType: DataType {
    DataType.from(cTensorParams: cTensor.pointee.info)
  }

  @inlinable
  public var kind: DeviceKind {
    DeviceKind.from(cTensorParams: cTensor.pointee.info)
  }

  @inlinable
  public var format: TensorFormat {
    TensorFormat.from(cTensorParams: cTensor.pointee.info)
  }

  @inlinable
  public var shape: TensorShape {
    TensorShape(dims: cTensor.pointee.info.dim)
  }

  @inlinable
  public var isTensorView: Bool {
    let type = Int(cTensor.pointee.type)
    return (type & CCV_TENSOR_VIEW) == CCV_TENSOR_VIEW
  }

  @inlinable
  public var isContiguous: Bool {
    let type = Int(cTensor.pointee.type)
    guard (type & CCV_TENSOR_VIEW) == CCV_TENSOR_VIEW else {
      return true
    }
    let cTensorView = UnsafeRawPointer(cTensor).assumingMemoryBound(to: ccv_nnc_tensor_view_t.self)
    return cTensorView.pointee.contiguous == 1
  }

  @inlinable
  public var strides: TensorShape {
    guard isTensorView else {
      var strides = shape
      var stride = 1
      for i in (0..<strides.count).reversed() {
        let oldStride = strides[i]
        strides[i] = stride
        stride *= oldStride
      }
      return strides
    }
    return TensorShape(
      dims: UnsafeMutableRawPointer(cTensor).bindMemory(
        to: ccv_nnc_tensor_view_t.self, capacity: 1
      ).pointee.stride)
  }
}

extension Tensor {
  @inlinable
  public func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
    let count = strides[0] * shape[0] * MemoryLayout<Element>.size
    return try body(UnsafeRawBufferPointer(start: cTensor.pointee.data.u8, count: count))
  }
  @inlinable
  public mutating func withUnsafeMutableBytes<R>(
    _ body: (UnsafeMutableRawBufferPointer) throws -> R
  ) rethrows -> R {
    let count = strides[0] * shape[0] * MemoryLayout<Element>.size
    return try body(UnsafeMutableRawBufferPointer(start: cTensor.pointee.data.u8, count: count))
  }
}

/// Basic tensor type.
public struct Tensor<Element: TensorNumeric>: AnyTensor {

  public var storage: AnyTensorStorage { _storage }
  public var cTensor: UnsafeMutablePointer<ccv_nnc_tensor_t> { _storage.cTensor }

  @usableFromInline
  var _storage: AnyTensorStorage

  private init(_ kind: DeviceKind, dataType: DataType, format: TensorFormat, shape: TensorShape) {
    let cTensor = ccv_nnc_tensor_new(
      nil,
      toCTensorParams(kind, dataType: dataType, format: format, shape: shape),
      0)!
    _storage = AnyTensorStorage(cTensor)
  }

  private init(_ kind: DeviceKind, _ dataType: DataType, _ dimensionFormat: TensorShapeFormat) {
    self.init(
      kind, dataType: dataType, format: dimensionFormat.format,
      shape: dimensionFormat.shape)
  }

  @usableFromInline
  init(_ tensor: AnyTensorStorage) {
    _storage = tensor
  }

  /**
   * Create a typed tensor from a type-erased tensor.
   *
   * - Parameter tensor: A type-erased tensor.
   */
  public init(_ tensor: AnyTensor) {
    precondition(tensor.dataType == Element.dataType)
    _storage = tensor.storage
  }

  /**
   * Convert from a different type tensor to this tensor. If the given tensor is not contiguous,
   * this method will make it contiguous.
   * - Parameter tensor: A type-erased tensor.
   */
  public init(from tensor: AnyTensor) {
    if tensor.dataType == Element.dataType {
      if tensor.isContiguous {
        _storage = tensor.storage
      } else {
        _storage = tensor.storage.copy()
      }
    } else {
      var cTensor = ccv_nnc_tensor_new(
        nil,
        toCTensorParams(
          tensor.kind, dataType: Element.dataType, format: tensor.format,
          shape: tensor.shape),
        0)
      var input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = tensor.cTensor
      ccv_nnc_cmd_exec(
        ccv_nnc_cmd(
          CCV_NNC_DATATYPE_CONVERSION_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0),
        ccv_nnc_no_hint, 0, &input, 1, &cTensor, 1, nil)
      _storage = AnyTensorStorage(cTensor!)
    }
  }

  public init(_ kind: DeviceKind, format: TensorFormat, shape: TensorShape) {
    self.init(kind, dataType: Element.dataType, format: format, shape: shape)
  }

  /**
   * Create a new uninitialized tensor.
   *
   * - Parameters:
   *   - kind: Which device this new tensor is on.
   *   - dimensionFormat: The format and shape of the new tensor.
   */
  public init(_ kind: DeviceKind, _ dimensionFormat: TensorShapeFormat) {
    self.init(kind, Element.dataType, dimensionFormat)
  }

  public init<S: Sequence>(
    _ sequence: S, kind: DeviceKind, format: TensorFormat, shape: TensorShape
  )
  where S.Element == Element {
    self.init(kind, format: format, shape: shape)
    let cArray = ContiguousArray(sequence)
    cArray.withUnsafeBytes { bytes -> Void in
      var tensor = ccv_nnc_tensor(
        bytes.baseAddress,
        toCTensorParams(.CPU, dataType: Element.dataType, format: format, shape: shape), 0
      )
      let cmd = ccv_nnc_cmd(
        CCV_NNC_DATA_TRANSFER_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0)
      var _output: UnsafeMutablePointer<ccv_nnc_tensor_t>? = cTensor
      withUnsafeMutablePointer(to: &tensor) { input in
        var _input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = input
        ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, nil)
      }
    }
  }

  /**
   * Create a new tensor and initialize with content from a sequence.
   *
   * - Parameters:
   *   - sequence: The sequence to initialize the new tensor with.
   *   - dimensionFormat: The format and shape of the new tensor.
   */
  public init<S: Sequence>(
    _ sequence: S, _ kind: DeviceKind, _ dimensionFormat: TensorShapeFormat
  )
  where S.Element == Element {
    self.init(
      sequence, kind: kind, format: dimensionFormat.format, shape: dimensionFormat.shape)
  }

  public init(
    _ kind: DeviceKind, format: TensorFormat, shape: TensorShape,
    unsafeMutablePointer: UnsafeMutablePointer<Element>, bindLifetimeOf: Any
  ) {
    let cTensor = ccv_nnc_tensor_new(
      unsafeMutablePointer,
      toCTensorParams(kind, dataType: Element.dataType, format: format, shape: shape),
      0)!
    self.init(AnyTensorStorage(cTensor, original: bindLifetimeOf))
  }

  @inlinable
  public subscript(indices: Int...) -> Element {
    get {
      return _storage[indices, Element.self]
    }
    set(v) {
      guard case .CPU = kind else {
        fatalError("cannot modify non-CPU tensor")
      }
      if !isKnownUniquelyReferenced(&_storage) {
        // Make a copy (copy-on-write).
        _storage = _storage.copy()
      }
      _storage[indices, Element.self] = v
    }
  }

  @inlinable
  public subscript(ranges: Range<Int>...) -> Tensor<Element> {
    get {
      return Tensor<Element>(_storage[ranges, Element.self])
    }
    set(v) {
      if !isKnownUniquelyReferenced(&_storage) {
        // Make a copy (copy-on-write).
        _storage = _storage.copy()
      }
      _storage[ranges, Element.self] = v._storage
    }
  }

  @usableFromInline
  subscript(indices: [Int], range: Range<Int>) -> Tensor<Element> {
    get {
      return Tensor<Element>(_storage[indices, range, Element.self])
    }
    set(v) {
      if !isKnownUniquelyReferenced(&_storage) {
        // Make a copy (copy-on-write).
        _storage = _storage.copy()
      }
      _storage[indices, range, Element.self] = v._storage
    }
  }

  @usableFromInline
  subscript(indices: [Int], range: UnboundedRange) -> Tensor<Element> {
    get {
      let shape = self.shape
      return Tensor<Element>(_storage[indices, 0..<shape[indices.count], Element.self])
    }
    set(v) {
      if !isKnownUniquelyReferenced(&_storage) {
        // Make a copy (copy-on-write).
        _storage = _storage.copy()
      }
      let shape = self.shape
      _storage[indices, 0..<shape[indices.count], Element.self] = v._storage
    }
  }
}

extension Tensor: ExpressibleByArrayLiteral {
  public init(_ elements: [Element]) {
    let wrapped = Wrapped(elements)
    let newt = wrapped.value.withUnsafeBytes {
      ccv_nnc_tensor_new(
        $0.baseAddress,
        toCTensorParams(
          .CPU, dataType: Element.dataType, format: .NCHW, shape: [elements.count]), 0)!
    }
    self.init(AnyTensorStorage(newt, original: wrapped))
  }
  public init(arrayLiteral elements: Element...) {
    self.init(elements)
  }
}

extension Array where Element: TensorNumeric {
  public init(_ tensor: Tensor<Element>) {
    let count = tensor.strides[0] * tensor.shape[0]
    let pointer = tensor.cTensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
    self.init(UnsafeBufferPointer(start: pointer, count: count))
  }
}

extension Tensor {
  @inlinable
  public subscript(range: Range<Int>) -> Tensor<Element> {
    get { self[[], range] }
    set { self[[], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, range: Range<Int>) -> Tensor<Element> {
    get { self[[i0], range] }
    set { self[[i0], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, range: Range<Int>) -> Tensor<Element> {
    get { self[[i0, i1], range] }
    set { self[[i0, i1], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, range: Range<Int>) -> Tensor<Element> {
    get { self[[i0, i1, i2], range] }
    set { self[[i0, i1, i2], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: Range<Int>) -> Tensor<Element> {
    get { self[[i0, i1, i2, i3], range] }
    set { self[[i0, i1, i2, i3], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: Range<Int>) -> Tensor<
    Element
  > {
    get { self[[i0, i1, i2, i3, i4], range] }
    set { self[[i0, i1, i2, i3, i4], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: Range<Int>)
    -> Tensor<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5], range] }
    set { self[[i0, i1, i2, i3, i4, i5], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, range: Range<Int>)
    -> Tensor<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5, i6], range] }
    set { self[[i0, i1, i2, i3, i4, i5, i6], range] = newValue }
  }
}

extension Tensor {
  @inlinable
  public subscript(range: UnboundedRange) -> Tensor<Element> {
    get { self }
    set { self[[], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, range: UnboundedRange) -> Tensor<Element> {
    get { self[[i0], range] }
    set { self[[i0], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, range: UnboundedRange) -> Tensor<Element> {
    get { self[[i0, i1], range] }
    set { self[[i0, i1], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, range: UnboundedRange) -> Tensor<Element> {
    get { self[[i0, i1, i2], range] }
    set { self[[i0, i1, i2], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: UnboundedRange) -> Tensor<Element> {
    get { self[[i0, i1, i2, i3], range] }
    set { self[[i0, i1, i2, i3], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: UnboundedRange) -> Tensor<
    Element
  >
  {
    get { self[[i0, i1, i2, i3, i4], range] }
    set { self[[i0, i1, i2, i3, i4], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: UnboundedRange)
    -> Tensor<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5], range] }
    set { self[[i0, i1, i2, i3, i4, i5], range] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int,
    range: UnboundedRange
  )
    -> Tensor<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5, i6], range] }
    set { self[[i0, i1, i2, i3, i4, i5, i6], range] = newValue }
  }
}

extension Tensor {

  /**
   * Move this tensor from CPU to GPU.
   *
   * - Parameters:
   *   - ordinal: Which GPU the new tensor will reside.
   *   - streamContext: Run the operation on the given stream context.
   * - Returns: A new tensor on GPU.
   */
  public func toGPU(_ ordinal: Int = 0, streamContext: StreamContext? = nil) -> Self {
    guard kind != .GPU(ordinal) else { return self }
    var _output = ccv_nnc_tensor_new(
      nil,
      toCTensorParams(.GPU(ordinal), dataType: dataType, format: format, shape: shape),
      0)
    let cmd = ccv_nnc_cmd(
      CCV_NNC_DATA_TRANSFER_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0)
    let _streamContext = streamContext?._stream
    var _input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = cTensor
    ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, _streamContext)
    return Self(AnyTensorStorage(_output!))
  }

  /**
   * Move this tensor from GPU to CPU.
   *
   * - Parameters:
   *   - streamContext: Run the operation on the given stream context.
   * - Returns: A new tensor on CPU.
   */
  public func toCPU(streamContext: StreamContext? = nil) -> Self {
    guard kind != .CPU else { return self }
    var _output = ccv_nnc_tensor_new(
      nil,
      toCTensorParams(.CPU, dataType: dataType, format: format, shape: shape),
      0)
    let cmd = ccv_nnc_cmd(
      CCV_NNC_DATA_TRANSFER_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0)
    let _streamContext = streamContext?._stream
    var _input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = cTensor
    ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, _streamContext)
    return Self(AnyTensorStorage(_output!))
  }

}

extension Tensor {

  /**
   * Make an explicit copy of the existing tensor. Raw tensors conforms to copy-on-write
   * semantics. But sometimes, it is better to have explicit copy for better memory management
   * hygiene. For example, if the tensor is obtained with .rawValue from a DynamicGraph
   * variable, keeping that tensor alive would keep the whole computation graph available.
   * Making an explicit copy would break that chain.
   *
   * - Returns: A new tensor copied from the existing one.
   */
  public func copied() -> Self {
    return Self(storage.copy())
  }

  /**
   * Only make explicit copy if the original is not contiguous in memory.
   *
   * - Returns: A new tensor copied from the existing one.
   */
  public func contiguous() -> Self {
    guard !isContiguous else { return self }
    return Self(storage.copy())
  }

}

extension Tensor {

  public func reshaped(
    format: TensorFormat, shape: TensorShape, offset: TensorShape? = nil,
    strides: TensorShape? = nil
  ) -> Self {
    let deviceKind = self.kind
    if isTensorView
      && (shape.count != self.shape.count || (strides != nil && strides != self.strides))
    {
      // Check if this is permuted, if it is, we cannot reshape. Need to copied first.
      let strides = self.strides
      for i in 1..<strides.count {
        precondition(
          strides[i - 1] >= strides[i],
          "The tensor is permuted, cannot reshape to \(shape), try .copied() before reshape.")
      }
    }
    guard let strides = strides else {
      precondition(offset == nil)  // Cannot have no strides but offset.
      let newt = ccv_nnc_tensor_new(
        cTensor.pointee.data.ptr,
        toCTensorParams(deviceKind, dataType: Element.dataType, format: format, shape: shape),
        0)!
      return Self(AnyTensorStorage(newt, original: _storage))
    }
    var cOffset = offset?.dims ?? (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    var cStrides = strides.dims
    let newt = withUnsafePointer(to: &cOffset) {
      cOffset -> UnsafeMutablePointer<ccv_nnc_tensor_view_t> in
      let cOffset = UnsafeRawPointer(cOffset).assumingMemoryBound(to: Int32.self)
      return withUnsafePointer(to: &cStrides) {
        cStrides -> UnsafeMutablePointer<ccv_nnc_tensor_view_t> in
        let cStrides = UnsafeRawPointer(cStrides).assumingMemoryBound(to: Int32.self)
        return ccv_nnc_tensor_view_new(
          cTensor,
          toCTensorParams(
            deviceKind, dataType: Element.dataType, format: format, shape: shape), cOffset,
          cStrides)!
      }
    }
    let anyTensor = newt.withMemoryRebound(to: ccv_nnc_tensor_t.self, capacity: 1) {
      AnyTensorStorage($0, original: _storage)
    }
    return Self(anyTensor)
  }

  /**
   * Create a new tensor pointing to the same memory region but with different sizes.
   *
   * - Parameters:
   *   - shapeFormat: New format and shape for the tensor.
   *   - offset: Whether offset on each shape.
   *   - strides: The strides on each shape.
   * - Returns: The new tensor with different format but the same memory content.
   */
  public func reshaped(
    _ shapeFormat: TensorShapeFormat, offset: TensorShape? = nil, strides: TensorShape? = nil
  ) -> Self {
    return reshaped(
      format: shapeFormat.format, shape: shapeFormat.shape, offset: offset,
      strides: strides)
  }

}

extension Tensor {

  /**
   * Create a new tensor with dimensions permuted.
   *
   * - Parameters:
   *   - indices: The indices for dimensions from the original tensor. For example, a [2, 3, 4] tensor with [2, 0, 1] indices will permute to a [4, 2, 3] tensor.
   * - Returns: The new tensor with dimensions permuted.
   */
  public func permuted(_ indices: Int...) -> Self {
    let deviceKind = self.kind
    let shape = self.shape
    let strides = self.strides
    var newShape = shape
    var newStrides = strides
    for (i, index) in indices.enumerated() {
      newShape[i] = shape[index]
      newStrides[i] = strides[index]
    }
    var cOffset:
      (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) = (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      )
    var cStrides = newStrides.dims
    let newt = withUnsafePointer(to: &cOffset) {
      cOffset -> UnsafeMutablePointer<ccv_nnc_tensor_view_t> in
      let cOffset = UnsafeRawPointer(cOffset).assumingMemoryBound(to: Int32.self)
      return withUnsafePointer(to: &cStrides) {
        cStrides -> UnsafeMutablePointer<ccv_nnc_tensor_view_t> in
        let cStrides = UnsafeRawPointer(cStrides).assumingMemoryBound(to: Int32.self)
        return ccv_nnc_tensor_view_new(
          cTensor,
          toCTensorParams(
            deviceKind, dataType: Element.dataType, format: format, shape: newShape), cOffset,
          cStrides)!
      }
    }
    let anyTensor = newt.withMemoryRebound(to: ccv_nnc_tensor_t.self, capacity: 1) {
      AnyTensorStorage($0, original: _storage)
    }
    return Self(anyTensor)
  }
}

extension Collection where Element == Tensor<Float64> {
  public func reshaped(
    format: TensorFormat, shape: TensorShape, offset: TensorShape? = nil,
    strides: TensorShape? = nil
  ) -> [Element] {
    return map {
      $0.reshaped(format: format, shape: shape, offset: offset, strides: strides)
    }
  }
  /**
   * Create new tensors pointing to the same memory region but with different sizes.
   *
   * - Parameters:
   *   - shapeFormat: New format and shape for the tensor.
   *   - offset: Whether offset on each shape.
   *   - strides: The strides on each shape.
   * - Returns: The new tensors with different format but the same memory content.
   */
  public func reshaped(
    _ shapeFormat: TensorShapeFormat, offset: TensorShape? = nil, strides: TensorShape? = nil
  ) -> [Element] {
    return map { $0.reshaped(shapeFormat, offset: offset, strides: strides) }
  }
}

extension Collection where Element == Tensor<Int64> {
  public func reshaped(
    format: TensorFormat, shape: TensorShape, offset: TensorShape? = nil,
    strides: TensorShape? = nil
  ) -> [Element] {
    return map {
      $0.reshaped(format: format, shape: shape, offset: offset, strides: strides)
    }
  }
  /**
   * Create new tensors pointing to the same memory region but with different sizes.
   *
   * - Parameters:
   *   - shapeFormat: New format and shape for the tensor.
   *   - offset: Whether offset on each shape.
   *   - strides: The strides on each shape.
   * - Returns: The new tensors with different format but the same memory content.
   */
  public func reshaped(
    _ shapeFormat: TensorShapeFormat, offset: TensorShape? = nil, strides: TensorShape? = nil
  ) -> [Element] {
    return map { $0.reshaped(shapeFormat, offset: offset, strides: strides) }
  }
}

extension Collection where Element == Tensor<Float32> {
  public func reshaped(
    format: TensorFormat, shape: TensorShape, offset: TensorShape? = nil,
    strides: TensorShape? = nil
  ) -> [Element] {
    return map {
      $0.reshaped(format: format, shape: shape, offset: offset, strides: strides)
    }
  }
  /**
   * Create new tensors pointing to the same memory region but with different sizes.
   *
   * - Parameters:
   *   - dimensionFormat: New format and shape for the tensor.
   *   - offset: Whether offset on each shape.
   *   - strides: The strides on each shape.
   * - Returns: The new tensors with different format but the same memory content.
   */
  public func reshaped(
    _ shapeFormat: TensorShapeFormat, offset: TensorShape? = nil, strides: TensorShape? = nil
  ) -> [Element] {
    return map { $0.reshaped(shapeFormat, offset: offset, strides: strides) }
  }
}

extension Collection where Element == Tensor<Int32> {
  public func reshaped(
    format: TensorFormat, shape: TensorShape, offset: TensorShape? = nil,
    strides: TensorShape? = nil
  ) -> [Element] {
    return map {
      $0.reshaped(format: format, shape: shape, offset: offset, strides: strides)
    }
  }
  /**
   * Create new tensors pointing to the same memory region but with different sizes.
   *
   * - Parameters:
   *   - shapeFormat: New format and shape for the tensor.
   *   - offset: Whether offset on each shape.
   *   - strides: The strides on each shape.
   * - Returns: The new tensors with different format but the same memory content.
   */
  public func reshaped(
    _ shapeFormat: TensorShapeFormat, offset: TensorShape? = nil, strides: TensorShape? = nil
  ) -> [Element] {
    return map { $0.reshaped(shapeFormat, offset: offset, strides: strides) }
  }
}

#if !((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
  extension Collection where Element == Tensor<Float16> {
    public func reshaped(
      format: TensorFormat, shape: TensorShape, offset: TensorShape? = nil,
      strides: TensorShape? = nil
    ) -> [Element] {
      return map {
        $0.reshaped(format: format, shape: shape, offset: offset, strides: strides)
      }
    }
    /**
   * Create new tensors pointing to the same memory region but with different sizes.
   *
   * - Parameters:
   *   - shapeFormat: New format and shape for the tensor.
   *   - offset: Whether offset on each shape.
   *   - strides: The strides on each shape.
   * - Returns: The new tensors with different format but the same memory content.
   */
    public func reshaped(
      _ shapeFormat: TensorShapeFormat, offset: TensorShape? = nil, strides: TensorShape? = nil
    ) -> [Element] {
      return map { $0.reshaped(shapeFormat, offset: offset, strides: strides) }
    }
  }
#endif

extension Collection where Element == Tensor<UInt8> {
  public func reshaped(
    format: TensorFormat, shape: TensorShape, offset: TensorShape? = nil,
    strides: TensorShape? = nil
  ) -> [Element] {
    return map {
      $0.reshaped(format: format, shape: shape, offset: offset, strides: strides)
    }
  }
  /**
   * Create new tensors pointing to the same memory region but with different sizes.
   *
   * - Parameters:
   *   - shapeFormat: New format and shape for the tensor.
   *   - offset: Whether offset on each shape.
   *   - strides: The strides on each shape.
   * - Returns: The new tensors with different format but the same memory content.
   */
  public func reshaped(
    _ shapeFormat: TensorShapeFormat, offset: TensorShape? = nil, strides: TensorShape? = nil
  ) -> [Element] {
    return map { $0.reshaped(shapeFormat, offset: offset, strides: strides) }
  }
}

extension AnyTensorStorage {

  #if ((os(macOS) || (os(iOS) && targetEnvironment(macCatalyst))) && (arch(i386) || arch(x86_64)))
    func toAnyTensor() -> AnyTensor {
      switch dataType {
      case .Float64:
        return Tensor<Float64>(self)
      case .Int64:
        return Tensor<Int64>(self)
      case .Float32:
        return Tensor<Float32>(self)
      case .Int32:
        return Tensor<Int32>(self)
      case .Float16:
        return Tensor<Float16>(self)
      case .UInt8:
        return Tensor<UInt8>(self)
      }
    }

    func toTensor<Element>(_ type: Element.Type) -> Element {
      switch dataType {
      case .Float64:
        return unsafeBitCast(Tensor<Float64>(self), to: Element.self)
      case .Int64:
        return unsafeBitCast(Tensor<Int64>(self), to: Element.self)
      case .Float32:
        return unsafeBitCast(Tensor<Float32>(self), to: Element.self)
      case .Int32:
        return unsafeBitCast(Tensor<Int32>(self), to: Element.self)
      case .Float16:
        return unsafeBitCast(Tensor<Float16>(self), to: Element.self)
      case .UInt8:
        return unsafeBitCast(Tensor<UInt8>(self), to: Element.self)
      }
    }
  #else
    func toAnyTensor() -> AnyTensor {
      switch dataType {
      case .Float64:
        return Tensor<Float64>(self)
      case .Int64:
        return Tensor<Int64>(self)
      case .Float32:
        return Tensor<Float32>(self)
      case .Int32:
        return Tensor<Int32>(self)
      case .Float16:
        return Tensor<Float16>(self)
      case .UInt8:
        return Tensor<UInt8>(self)
      }
    }

    func toTensor<Element>(_ type: Element.Type) -> Element {
      switch dataType {
      case .Float64:
        return unsafeBitCast(Tensor<Float64>(self), to: Element.self)
      case .Int64:
        return unsafeBitCast(Tensor<Int64>(self), to: Element.self)
      case .Float32:
        return unsafeBitCast(Tensor<Float32>(self), to: Element.self)
      case .Int32:
        return unsafeBitCast(Tensor<Int32>(self), to: Element.self)
      case .Float16:
        return unsafeBitCast(Tensor<Float16>(self), to: Element.self)
      case .UInt8:
        return unsafeBitCast(Tensor<UInt8>(self), to: Element.self)
      }
    }
  #endif

}

extension Tensor: CustomStringConvertible {
  public var description: String {
    return
      "Tensor<\(dataType)>(kind: .\(kind), format: .\(format), shape: \(shape), strides: \(strides))"
  }
}

@inlinable
func toCDimensions(_ dimensions: [Int]?) -> (
  Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32
) {
  guard let dimensions = dimensions else {
    return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  }
  assert(dimensions.count <= CCV_NNC_MAX_DIM_ALLOC)
  switch dimensions.count {
  case 1:
    return (Int32(dimensions[0]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  case 2:
    return (Int32(dimensions[0]), Int32(dimensions[1]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  case 3:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), 0, 0, 0, 0, 0, 0, 0, 0, 0
    )
  case 4:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), 0, 0,
      0, 0, 0, 0, 0, 0
    )
  case 5:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), 0, 0, 0, 0, 0, 0, 0
    )
  case 6:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), 0, 0, 0, 0, 0, 0
    )
  case 7:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), 0, 0, 0, 0, 0
    )
  case 8:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]), 0, 0,
      0, 0
    )
  case 9:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]),
      Int32(dimensions[8]), 0, 0, 0
    )
  case 10:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]),
      Int32(dimensions[8]), Int32(dimensions[9]), 0, 0
    )
  case 11:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]),
      Int32(dimensions[8]), Int32(dimensions[9]), Int32(dimensions[10]), 0
    )
  case 12:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]),
      Int32(dimensions[8]), Int32(dimensions[9]), Int32(dimensions[10]), Int32(dimensions[11])
    )
  default:
    return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  }
}

@inlinable
func toCDimensionsArray(_ dimensions: [Int]?) -> [Int32] {
  guard let dimensions = dimensions else {
    return [0, 0, 0, 0, 0, 0, 0, 0]
  }
  assert(dimensions.count <= CCV_NNC_MAX_DIM_ALLOC)
  switch dimensions.count {
  case 1:
    return [Int32(dimensions[0]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  case 2:
    return [Int32(dimensions[0]), Int32(dimensions[1]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  case 3:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
  case 4:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), 0, 0,
      0, 0, 0, 0, 0, 0,
    ]
  case 5:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), 0, 0, 0, 0, 0, 0, 0,
    ]
  case 6:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), 0, 0, 0, 0, 0, 0,
    ]
  case 7:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), 0, 0, 0, 0, 0,
    ]
  case 8:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]), 0, 0,
      0, 0,
    ]
  case 9:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]),
      Int32(dimensions[8]), 0, 0, 0,
    ]
  case 10:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]),
      Int32(dimensions[8]), Int32(dimensions[9]), 0, 0,
    ]
  case 11:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]),
      Int32(dimensions[8]), Int32(dimensions[9]), Int32(dimensions[10]), 0,
    ]
  case 12:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]),
      Int32(dimensions[8]), Int32(dimensions[9]), Int32(dimensions[10]), Int32(dimensions[11]),
    ]
  default:
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  }
}

@inlinable
func fromCDimensions(
  _ dim: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
) -> [Int] {
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
  } else if dim.8 == 0 {
    return [
      Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3), Int(dim.4), Int(dim.5), Int(dim.6),
      Int(dim.7),
    ]
  } else if dim.9 == 0 {
    return [
      Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3), Int(dim.4), Int(dim.5), Int(dim.6),
      Int(dim.7), Int(dim.8),
    ]
  } else if dim.10 == 0 {
    return [
      Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3), Int(dim.4), Int(dim.5), Int(dim.6),
      Int(dim.7), Int(dim.8), Int(dim.9),
    ]
  } else if dim.11 == 0 {
    return [
      Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3), Int(dim.4), Int(dim.5), Int(dim.6),
      Int(dim.7), Int(dim.8), Int(dim.9), Int(dim.10),
    ]
  } else {
    return [
      Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3), Int(dim.4), Int(dim.5), Int(dim.6),
      Int(dim.7), Int(dim.8), Int(dim.9), Int(dim.10), Int(dim.11),
    ]
  }
}

@inlinable
func toCTensorParams(
  _ kind: DeviceKind, dataType: DataType, format: TensorFormat, dimensions: [Int]
) -> ccv_nnc_tensor_param_t {
  var params = ccv_nnc_tensor_param_t()
  params.type = kind.toC
  params.datatype = dataType.toC
  params.format = format.toC
  params.dim = toCDimensions(dimensions)
  return params
}

@inlinable
func toCTensorParams(
  _ kind: DeviceKind, dataType: DataType, format: TensorFormat, shape: TensorShape
) -> ccv_nnc_tensor_param_t {
  var params = ccv_nnc_tensor_param_t()
  params.type = kind.toC
  params.datatype = dataType.toC
  params.format = format.toC
  params.dim = shape.dims
  return params
}

extension Tensor: CustomDebugStringConvertible {
  public var debugDescription: String {
    let tensor = kind == .CPU ? self : self.toCPU()
    guard let cString = ccv_nnc_tensor_format_new(tensor.cTensor) else {
      return description
    }
    let debugDescription = description + " " + String(cString: cString)
    free(cString)
    return debugDescription
  }
}

extension Tensor {
  public func data(using codec: DynamicGraph.Store.Codec = []) -> Data {
    let tensor = kind == .CPU ? self : toCPU()
    var codec = codec
    codec.subtract([.externalData, .jit, .externalOnDemand])
    let encode = codec.encode
    return withExtendedLifetime(tensor) {
      let cTensor = tensor.cTensor
      let dataSize = ccv_nnc_tensor_data_size_without_padding(cTensor.pointee.info)
      guard
        let workspace = malloc(
          dataSize + MemoryLayout<ccv_nnc_tensor_param_t>.size + MemoryLayout<UInt32>.size)
      else {
        return Data()
      }
      var identifier: UInt32 = 0
      guard let encode = encode else {
        memcpy(workspace, &identifier, MemoryLayout<UInt32>.size)
        memcpy(
          workspace + MemoryLayout<UInt32>.size, &cTensor.pointee.info,
          MemoryLayout<ccv_nnc_tensor_param_t>.size)
        memcpy(
          workspace + MemoryLayout<UInt32>.size + MemoryLayout<ccv_nnc_tensor_param_t>.size,
          cTensor.pointee.data.u8, dataSize)
        return Data(
          bytesNoCopy: workspace,
          count: dataSize + MemoryLayout<ccv_nnc_tensor_param_t>.size + MemoryLayout<UInt32>.size,
          deallocator: .free)
      }
      var params = cTensor.pointee.info
      var encodedSize = dataSize
      var dim = params.dim
      guard
        (withUnsafePointer(to: &dim) {
          let dim = UnsafeRawPointer($0).assumingMemoryBound(to: Int32.self)
          return encode(
            cTensor.pointee.data.u8, dataSize, cTensor.pointee.info.datatype, dim,
            ccv_nnc_tensor_nd(dim), nil,
            workspace + MemoryLayout<ccv_nnc_tensor_param_t>.size + MemoryLayout<UInt32>.size,
            &encodedSize, &params, &identifier)
        }) != 0
      else {
        identifier = 0
        memcpy(workspace, &identifier, MemoryLayout<UInt32>.size)
        memcpy(
          workspace + MemoryLayout<UInt32>.size, &cTensor.pointee.info,
          MemoryLayout<ccv_nnc_tensor_param_t>.size)
        memcpy(
          workspace + MemoryLayout<UInt32>.size + MemoryLayout<ccv_nnc_tensor_param_t>.size,
          cTensor.pointee.data.u8, dataSize)
        return Data(
          bytesNoCopy: workspace,
          count: dataSize + MemoryLayout<ccv_nnc_tensor_param_t>.size + MemoryLayout<UInt32>.size,
          deallocator: .free)
      }
      memcpy(workspace, &identifier, MemoryLayout<UInt32>.size)
      memcpy(
        workspace + MemoryLayout<UInt32>.size, &cTensor.pointee.info,
        MemoryLayout<ccv_nnc_tensor_param_t>.size)
      let bytes = realloc(
        workspace,
        encodedSize + MemoryLayout<ccv_nnc_tensor_param_t>.size + MemoryLayout<UInt32>.size)!
      return Data(
        bytesNoCopy: bytes,
        count: encodedSize + MemoryLayout<ccv_nnc_tensor_param_t>.size + MemoryLayout<UInt32>.size,
        deallocator: .free)
    }
  }

  public init?(data: Data, using codec: DynamicGraph.Store.Codec) {
    guard
      let storage =
        (data.withUnsafeBytes { (unsafePointer: UnsafeRawBufferPointer) -> AnyTensorStorage? in
          guard let baseAddress = unsafePointer.baseAddress else { return nil }
          guard
            unsafePointer.count >= MemoryLayout<UInt32>.size
              + MemoryLayout<ccv_nnc_tensor_param_t>.size
          else {
            return nil
          }
          let identifier = unsafePointer.load(as: UInt32.self)
          let params = unsafePointer.loadUnaligned(
            fromByteOffset: MemoryLayout<UInt32>.size, as: ccv_nnc_tensor_param_t.self)
          guard identifier != 0 else {
            let dataSize = ccv_nnc_tensor_data_size_without_padding(params)
            guard
              unsafePointer.count >= MemoryLayout<UInt32>.size
                + MemoryLayout<ccv_nnc_tensor_param_t>.size + dataSize
            else {
              return nil
            }
            let cTensor = ccv_nnc_tensor_new(nil, params, 0)!
            memcpy(
              cTensor.pointee.data.u8,
              baseAddress + MemoryLayout<UInt32>.size + MemoryLayout<ccv_nnc_tensor_param_t>.size,
              dataSize)
            return AnyTensorStorage(cTensor, original: nil)
          }
          var codec = codec
          codec.subtract([.externalData, .jit, .externalOnDemand])
          guard let decode = codec.decode else {
            return nil
          }
          let dataSize =
            unsafePointer.count - MemoryLayout<UInt32>.size
            - MemoryLayout<ccv_nnc_tensor_param_t>.size
          var dim = params.dim
          var cTensor = ccv_nnc_tensor_new(nil, params, 0)
          guard
            (withUnsafePointer(to: &dim) {
              let dim = UnsafeRawPointer($0).assumingMemoryBound(to: Int32.self)
              var decodedSize = ccv_nnc_tensor_data_size_without_padding(params)
              return decode(
                baseAddress + MemoryLayout<UInt32>.size + MemoryLayout<ccv_nnc_tensor_param_t>.size,
                dataSize, params.datatype, dim, ccv_nnc_tensor_nd(dim), identifier, nil, params,
                &cTensor, cTensor?.pointee.data.u8, &decodedSize)
            }) != 0
          else {
            ccv_nnc_tensor_free(cTensor)
            return nil
          }
          return AnyTensorStorage(cTensor!, original: nil)
        })
    else {
      return nil
    }
    self.init(storage)
  }
}
