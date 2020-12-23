import C_nnc

final class CmdParamsFactory {
  static let factory = CmdParamsFactory()
  init() {
    ccv_nnc_init()
  }
  func newParams() -> ccv_nnc_cmd_param_t {
    return ccv_nnc_cmd_param_t()
  }
  func sink() {
  }
}

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

  public enum GPUInfo {
    public static var count: Int {
      Int(ccv_nnc_device_count(Int32(CCV_STREAM_CONTEXT_GPU)))
    }
  }
}

public enum TensorFormat {
  case NHWC
  case NCHW
  case CHWN

  static func from(cTensorParams: ccv_nnc_tensor_param_t) -> TensorFormat {
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

public enum TensorDimensionFormat {
  case C(Int)  // Assuming NCHW
  case NC(Int, Int)  // Assuming NCHW
  case HWC(Int, Int, Int)  // Assuming NHWC
  case CHW(Int, Int, Int)  // Assuming NCHW
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

extension Float16: TensorNumeric {
  public static var dataType: DataType { .Float16 }
}

extension UInt8: TensorNumeric {
  public static var dataType: DataType { .UInt8 }
}

public final class _AnyTensor {
  let _tensor: UnsafeMutablePointer<ccv_nnc_tensor_t>
  fileprivate let original: Any?
  private let selfOwned: Bool

  init(
    _ tensor: UnsafeMutablePointer<ccv_nnc_tensor_t>, original: Any? = nil, selfOwned: Bool = true
  ) {
    self.original = original
    self.selfOwned = selfOwned
    _tensor = tensor
  }

  deinit {
    guard original == nil else { return }
    guard selfOwned else { return }
    ccv_nnc_tensor_free(_tensor)
  }

  var dataType: DataType {
    DataType.from(cTensorParams: _tensor.pointee.info)
  }

  func copy() -> _AnyTensor {
    var input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = _tensor
    var output = ccv_nnc_tensor_new(nil, _tensor.pointee.info, 0)
    ccv_nnc_cmd_exec(
      ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0),
      ccv_nnc_no_hint, 0, &input, 1, &output, 1, nil)
    return _AnyTensor(output!)
  }

  var increments: [Int] {
    let type = Int(_tensor.pointee.type)
    guard (type & CCV_TENSOR_VIEW) == CCV_TENSOR_VIEW else {
      return fromCDimensions(_tensor.pointee.info.dim)
    }
    let inc = UnsafeMutableRawPointer(_tensor).bindMemory(
      to: ccv_nnc_tensor_view_t.self, capacity: 1
    ).pointee.inc
    return fromCDimensions(inc)
  }

  subscript<Element: TensorNumeric>(indices: [Int], type: Element.Type) -> Element {
    get {
      let increments = self.increments
      assert(increments.count == indices.count)
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
      let increments = self.increments
      assert(increments.count == indices.count)
      let count = increments.reduce(1, *)
      // We need to deal with GPU memory.
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

extension _AnyTensor {
  subscript<Element: TensorNumeric>(ranges: [Range<Int>], type: Element.Type) -> _AnyTensor {
    get {
      // This is a restricted form a reshape.
      let cTensorParams = _tensor.pointee.info
      let device = DeviceKind.from(cTensorParams: cTensorParams)
      let format = TensorFormat.from(cTensorParams: cTensorParams)
      let increments = self.increments
      assert(increments.count == ranges.count)
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < increments[i])
        assert(range.upperBound > 0 && range.upperBound <= increments[i])
      }
      let offset = ranges.map { $0.lowerBound }
      let dimensions = ranges.map { $0.count }
      var cOffset = toCDimensions(offset)
      var cIncrements = toCDimensions(increments)
      let newt = withUnsafePointer(to: &cOffset.0) { cOffset in
        withUnsafePointer(to: &cIncrements.0) { cIncrements in
          ccv_nnc_tensor_view_new(
            _tensor,
            toCTensorParams(
              device, dataType: Element.dataType, format: format, dimensions: dimensions), cOffset,
            cIncrements)!
        }
      }
      return newt.withMemoryRebound(to: ccv_nnc_tensor_t.self, capacity: 1) {
        _AnyTensor($0, original: self)
      }
    }
    set(v) {
      let cTensorParams = _tensor.pointee.info
      let device = DeviceKind.from(cTensorParams: cTensorParams)
      let format = TensorFormat.from(cTensorParams: cTensorParams)
      let increments = self.increments
      assert(increments.count == ranges.count)
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < increments[i])
        assert(range.upperBound > 0 && range.upperBound <= increments[i])
      }
      let offset = ranges.map { $0.lowerBound }
      let dimensions = ranges.map { $0.count }
      var cOffset = toCDimensions(offset)
      var cIncrements = toCDimensions(increments)
      var newt = withUnsafePointer(to: &cOffset.0) { cOffset in
        withUnsafePointer(to: &cIncrements.0) { cIncrements in
          ccv_nnc_tensor_view(
            _tensor,
            toCTensorParams(
              device, dataType: Element.dataType, format: format, dimensions: dimensions), cOffset,
            cIncrements)
        }
      }
      let inputDim = fromCDimensions(v._tensor.pointee.info.dim)
      for (i, dimension) in dimensions.enumerated() {
        assert(dimension == inputDim[i])
      }
      var input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = v._tensor
      withUnsafeMutablePointer(to: &newt) { newt in
        var output: UnsafeMutablePointer<ccv_nnc_tensor_t>? = UnsafeMutableRawPointer(newt)
          .bindMemory(to: ccv_nnc_tensor_t.self, capacity: 1)
        ccv_nnc_cmd_exec(
          ccv_nnc_cmd(
            CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0),
          ccv_nnc_no_hint, 0, &input, 1, &output, 1, nil)
      }
    }
  }
}

extension _AnyTensor {
  subscript<Element: TensorNumeric>(indices: [Int], range: Range<Int>, type: Element.Type)
    -> [Element]
  {
    get {
      let increments = self.increments
      assert(increments.count == indices.count + 1)
      let count = increments.reduce(1, *)
      let pointer = _tensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
      assert(range.lowerBound >= 0 && range.lowerBound < increments[indices.count])
      assert(range.upperBound > 0 && range.upperBound <= increments[indices.count])
      var offset = 0
      for (i, increment) in increments.prefix(indices.count).enumerated() {
        offset *= increment
        offset += indices[i]
      }
      offset *= increments[indices.count]
      offset += range.lowerBound
      return Array(UnsafeBufferPointer(start: pointer + offset, count: range.count))
    }
    set(v) {
      let increments = self.increments
      assert(increments.count == indices.count + 1)
      let count = increments.reduce(1, *)
      let pointer = _tensor.pointee.data.ptr.bindMemory(to: Element.self, capacity: count)
      assert(range.lowerBound >= 0 && range.lowerBound < increments[indices.count])
      assert(range.upperBound > 0 && range.upperBound <= increments[indices.count])
      assert(range.count == v.count)
      var offset = 0
      for (i, increment) in increments.prefix(indices.count).enumerated() {
        offset *= increment
        offset += indices[i]
      }
      offset *= increments[indices.count]
      offset += range.lowerBound
      v.withUnsafeBytes { bytes -> Void in
        memcpy(pointer + offset, bytes.baseAddress!, bytes.count)
      }
    }
  }
}

public protocol AnyTensor {
  var underlying: _AnyTensor { get }
}

extension AnyTensor {
  public var dataType: DataType {
    DataType.from(cTensorParams: underlying._tensor.pointee.info)
  }

  public var kind: DeviceKind {
    DeviceKind.from(cTensorParams: underlying._tensor.pointee.info)
  }

  public var format: TensorFormat {
    TensorFormat.from(cTensorParams: underlying._tensor.pointee.info)
  }

  public var dimensions: [Int] {
    fromCDimensions(underlying._tensor.pointee.info.dim)
  }

  public var isTensorView: Bool {
    let type = Int(underlying._tensor.pointee.type)
    return (type & CCV_TENSOR_VIEW) == CCV_TENSOR_VIEW
  }

  public var increments: [Int] {
    guard isTensorView else {
      return dimensions
    }
    let inc = UnsafeMutableRawPointer(underlying._tensor).bindMemory(
      to: ccv_nnc_tensor_view_t.self, capacity: 1
    ).pointee.inc
    return fromCDimensions(inc)
  }
}

extension Tensor {
  public func withUnsafeBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
    let count = increments.reduce(MemoryLayout<Element>.size, *)
    return try body(UnsafeRawBufferPointer(start: underlying._tensor.pointee.data.u8, count: count))
  }
}

public struct Tensor<Element: TensorNumeric>: AnyTensor {

  private var _tensor: _AnyTensor

  public var underlying: _AnyTensor { _tensor }

  private init(_ kind: DeviceKind, dataType: DataType, format: TensorFormat, dimensions: [Int]) {
    let underlying = ccv_nnc_tensor_new(
      nil,
      toCTensorParams(kind, dataType: dataType, format: format, dimensions: dimensions),
      0)!
    _tensor = _AnyTensor(underlying)
  }

  private init(_ kind: DeviceKind, _ dataType: DataType, _ dimensionFormat: TensorDimensionFormat) {
    self.init(
      kind, dataType: dataType, format: dimensionFormat.format,
      dimensions: dimensionFormat.dimensions)
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

  public init<S: Sequence>(_ sequence: S, format: TensorFormat, dimensions: [Int])
  where S.Element == Element {
    self.init(.CPU, format: format, dimensions: dimensions)
    let cArray = ContiguousArray(sequence)
    cArray.withUnsafeBytes { bytes -> Void in
      memcpy(_tensor._tensor.pointee.data.u8, bytes.baseAddress!, bytes.count)
    }
  }

  public init<S: Sequence>(_ sequence: S, _ dimensionFormat: TensorDimensionFormat)
  where S.Element == Element {
    self.init(sequence, format: dimensionFormat.format, dimensions: dimensionFormat.dimensions)
  }

  public init(
    _ kind: DeviceKind, format: TensorFormat, dimensions: [Int],
    unsafeMutablePointer: UnsafeMutablePointer<Element>, keepAlive: Any
  ) {
    let underlying = ccv_nnc_tensor_new(
      unsafeMutablePointer,
      toCTensorParams(kind, dataType: Element.dataType, format: format, dimensions: dimensions),
      0)!
    self.init(_AnyTensor(underlying, original: keepAlive))
  }

  public subscript(indices: Int...) -> Element {
    get {
      return _tensor[indices, Element.self]
    }
    set(v) {
      guard case .CPU = kind else {
        fatalError("cannot modify non-CPU tensor")
      }
      if !isKnownUniquelyReferenced(&_tensor) {
        // Make a copy (copy-on-write).
        _tensor = _tensor.copy()
      }
      _tensor[indices, Element.self] = v
    }
  }

  public subscript(ranges: Range<Int>...) -> Tensor<Element> {
    get {
      return Tensor<Element>(_tensor[ranges, Element.self])
    }
    set(v) {
      guard case .CPU = kind else {
        fatalError("cannot modify non-CPU tensor")
      }
      if !isKnownUniquelyReferenced(&_tensor) {
        // Make a copy (copy-on-write).
        _tensor = _tensor.copy()
      }
      _tensor[ranges, Element.self] = v._tensor
    }
  }

  private subscript(indices: [Int], range: Range<Int>) -> [Element] {
    get {
      return _tensor[indices, range, Element.self]
    }
    set(v) {
      guard case .CPU = kind else {
        fatalError("cannot modify non-CPU tensor")
      }
      if !isKnownUniquelyReferenced(&_tensor) {
        // Make a copy (copy-on-write).
        _tensor = _tensor.copy()
      }
      _tensor[indices, range, Element.self] = v
    }
  }
}

extension Tensor {
  public subscript(range: Range<Int>) -> [Element] {
    get { self[[], range] }
    set { self[[], range] = newValue }
  }

  public subscript(i0: Int, range: Range<Int>) -> [Element] {
    get { self[[i0], range] }
    set { self[[i0], range] = newValue }
  }

  public subscript(i0: Int, i1: Int, range: Range<Int>) -> [Element] {
    get { self[[i0, i1], range] }
    set { self[[i0, i1], range] = newValue }
  }

  public subscript(i0: Int, i1: Int, i2: Int, range: Range<Int>) -> [Element] {
    get { self[[i0, i1, i2], range] }
    set { self[[i0, i1, i2], range] = newValue }
  }

  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: Range<Int>) -> [Element] {
    get { self[[i0, i1, i2, i3], range] }
    set { self[[i0, i1, i2, i3], range] = newValue }
  }

  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: Range<Int>) -> [Element] {
    get { self[[i0, i1, i2, i3, i4], range] }
    set { self[[i0, i1, i2, i3, i4], range] = newValue }
  }

  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: Range<Int>)
    -> [Element]
  {
    get { self[[i0, i1, i2, i3, i4, i5], range] }
    set { self[[i0, i1, i2, i3, i4, i5], range] = newValue }
  }

  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, range: Range<Int>)
    -> [Element]
  {
    get { self[[i0, i1, i2, i3, i4, i5, i6], range] }
    set { self[[i0, i1, i2, i3, i4, i5, i6], range] = newValue }
  }
}

extension Tensor {

  public func toGPU(_ ordinal: Int = 0, streamContext: StreamContext? = nil) -> Self {
    var _output = ccv_nnc_tensor_new(
      nil,
      toCTensorParams(.GPU(ordinal), dataType: dataType, format: format, dimensions: dimensions),
      0)
    let cmd = ccv_nnc_cmd(
      CCV_NNC_DATA_TRANSFER_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0)
    let _streamContext = streamContext?._stream
    var _input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = underlying._tensor
    ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, _streamContext)
    return Self(_AnyTensor(_output!))
  }

  public func toCPU(streamContext: StreamContext? = nil) -> Self {
    var _output = ccv_nnc_tensor_new(
      nil,
      toCTensorParams(.CPU, dataType: dataType, format: format, dimensions: dimensions),
      0)
    let cmd = ccv_nnc_cmd(
      CCV_NNC_DATA_TRANSFER_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0)
    let _streamContext = streamContext?._stream
    var _input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = underlying._tensor
    ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, _streamContext)
    return Self(_AnyTensor(_output!))
  }

}

extension Tensor {

  public func reshape(
    format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil
  ) -> Self {
    let cTensorParams = _tensor._tensor.pointee.info
    let device = DeviceKind.from(cTensorParams: cTensorParams)
    guard let offset = offset, let increments = increments else {
      let newt = ccv_nnc_tensor_new(
        _tensor._tensor.pointee.data.ptr,
        toCTensorParams(device, dataType: Element.dataType, format: format, dimensions: dimensions),
        0)!
      return Self(_AnyTensor(newt, original: _tensor))
    }
    var cOffset = toCDimensions(offset)
    var cIncrements = toCDimensions(increments)
    let newt = withUnsafePointer(to: &cOffset.0) { cOffset in
      withUnsafePointer(to: &cIncrements.0) { cIncrements in
        ccv_nnc_tensor_view_new(
          _tensor._tensor,
          toCTensorParams(
            device, dataType: Element.dataType, format: format, dimensions: dimensions), cOffset,
          cIncrements)!
      }
    }
    let anyTensor = newt.withMemoryRebound(to: ccv_nnc_tensor_t.self, capacity: 1) {
      _AnyTensor($0, original: _tensor)
    }
    return Self(anyTensor)
  }

  public func reshape(
    _ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil
  ) -> Self {
    return reshape(
      format: dimensionFormat.format, dimensions: dimensionFormat.dimensions, offset: offset,
      increments: increments)
  }

}

extension Collection where Element == Tensor<Float64> {
  public func reshape(
    format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map {
      $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments)
    }
  }
  public func reshape(
    _ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map { $0.reshape(dimensionFormat, offset: offset, increments: increments) }
  }
}

extension Collection where Element == Tensor<Int64> {
  public func reshape(
    format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map {
      $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments)
    }
  }
  public func reshape(
    _ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map { $0.reshape(dimensionFormat, offset: offset, increments: increments) }
  }
}

extension Collection where Element == Tensor<Float32> {
  public func reshape(
    format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map {
      $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments)
    }
  }
  public func reshape(
    _ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map { $0.reshape(dimensionFormat, offset: offset, increments: increments) }
  }
}

extension Collection where Element == Tensor<Int32> {
  public func reshape(
    format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map {
      $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments)
    }
  }
  public func reshape(
    _ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map { $0.reshape(dimensionFormat, offset: offset, increments: increments) }
  }
}

extension Collection where Element == Tensor<Float16> {
  public func reshape(
    format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map {
      $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments)
    }
  }
  public func reshape(
    _ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map { $0.reshape(dimensionFormat, offset: offset, increments: increments) }
  }
}

extension Collection where Element == Tensor<UInt8> {
  public func reshape(
    format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map {
      $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments)
    }
  }
  public func reshape(
    _ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil
  ) -> [Element] {
    return map { $0.reshape(dimensionFormat, offset: offset, increments: increments) }
  }
}

extension _AnyTensor {

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

}

extension Tensor: CustomStringConvertible {
  public var description: String {
    return
      "Tensor<\(dataType)>(kind: .\(kind), format: .\(format), dimensions: \(dimensions), increments: \(increments))"
  }
}

func toCDimensions(_ dimensions: [Int]?) -> (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
{
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
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), 0, 0,
      0, 0
    )
  case 5:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), 0, 0, 0
    )
  case 6:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), 0, 0
    )
  case 7:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), 0
    )
  case 8:
    return (
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7])
    )
  default:
    return (0, 0, 0, 0, 0, 0, 0, 0)
  }
}

func toCDimensionsArray(_ dimensions: [Int]?) -> [Int32] {
  guard let dimensions = dimensions else {
    return [0, 0, 0, 0, 0, 0, 0, 0]
  }
  assert(dimensions.count <= CCV_NNC_MAX_DIM_ALLOC)
  switch dimensions.count {
  case 1:
    return [Int32(dimensions[0]), 0, 0, 0, 0, 0, 0, 0]
  case 2:
    return [Int32(dimensions[0]), Int32(dimensions[1]), 0, 0, 0, 0, 0, 0]
  case 3:
    return [Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), 0, 0, 0, 0, 0]
  case 4:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]), 0, 0,
      0, 0,
    ]
  case 5:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), 0, 0, 0,
    ]
  case 6:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), 0, 0,
    ]
  case 7:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), 0,
    ]
  case 8:
    return [
      Int32(dimensions[0]), Int32(dimensions[1]), Int32(dimensions[2]), Int32(dimensions[3]),
      Int32(dimensions[4]), Int32(dimensions[5]), Int32(dimensions[6]), Int32(dimensions[7]),
    ]
  default:
    return [0, 0, 0, 0, 0, 0, 0, 0]
  }
}

func fromCDimensions(_ dim: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)) -> [Int] {
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
    return [
      Int(dim.0), Int(dim.1), Int(dim.2), Int(dim.3), Int(dim.4), Int(dim.5), Int(dim.6),
      Int(dim.7),
    ]
  }
}

func toCTensorParams(
  _ kind: DeviceKind, dataType: DataType, format: TensorFormat, dimensions: [Int]
) -> ccv_nnc_tensor_param_t {
  var params = ccv_nnc_tensor_param_t()
  switch kind {
  case .CPU:
    params.type = Int32(CCV_TENSOR_CPU_MEMORY)
  case .GPU(let ordinal):
    params.type = Int32(CCV_TENSOR_GPU_MEMORY | (ordinal << 8))
  }
  params.datatype = dataType.toC
  params.format = format.toC
  params.dim = toCDimensions(dimensions)
  return params
}
