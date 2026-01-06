#if canImport(C_nnc)
import C_nnc
#elseif canImport(C_swiftpm_nnc)
import C_swiftpm_nnc
#endif

/// A dynamic graph is a workspace for computations. All tensor variables can be tracked
/// from a dynamic graph.
public final class DynamicGraph {

  fileprivate class _AnyTensor {

    let graph: DynamicGraph
    let _tensor: ccv_nnc_tensor_variable_t
    let original: Any?

    init(
      graph: DynamicGraph, tensor: ccv_nnc_tensor_variable_t, requiresGrad: Bool = false,
      original: Any? = nil
    ) {
      self.graph = graph
      _tensor = tensor
      self.requiresGrad = requiresGrad
      self.original = original
    }

    deinit {
      ccv_nnc_tensor_variable_free(graph.cGraph, _tensor)
    }

    var requiresGrad: Bool
  }

  /**
   * A type-erased tensor variable.
   */
  public class AnyTensor {

    public var grad: AnyTensor? = nil

    fileprivate let underlying: _AnyTensor

    public var graph: DynamicGraph { underlying.graph }
    var _tensor: ccv_nnc_tensor_variable_t { underlying._tensor }

    public var requiresGrad: Bool {  // I would like to keep this as internal. Unfortunately, that is hard to do. Hence, mark with _.
      get {
        underlying.requiresGrad
      }
      set(v) {
        underlying.requiresGrad = v
        if v {
          precondition(!self.isConstant)
          graph.trackGrad(self)
        } else {
          graph.untrackGrad(ObjectIdentifier(self))
        }
      }
    }

    required init(
      graph: DynamicGraph, tensor: ccv_nnc_tensor_variable_t, requiresGrad: Bool = false,
      original: Any? = nil
    ) {
      underlying = _AnyTensor(
        graph: graph, tensor: tensor, requiresGrad: requiresGrad, original: original)
    }

    fileprivate init(_ underlying: _AnyTensor) {
      self.underlying = underlying
    }

    deinit {
      if requiresGrad {
        graph.untrackGrad(ObjectIdentifier(self))
      }
    }

    public required init(_ tensor: AnyTensor) {
      self.underlying = tensor.underlying
    }

    public var shape: TensorShape {
      let _graph = graph.cGraph
      let info = ccv_nnc_tensor_variable_params(_graph, _tensor)
      return TensorShape(dims: info.dim)
    }

    public var kind: DeviceKind {
      let _graph = graph.cGraph
      let info = ccv_nnc_tensor_variable_params(_graph, _tensor)
      return DeviceKind.from(cTensorParams: info)
    }

    public var format: TensorFormat {
      let _graph = graph.cGraph
      let info = ccv_nnc_tensor_variable_params(_graph, _tensor)
      return TensorFormat.from(cTensorParams: info)
    }

    public var strides: TensorShape {
      let _graph = graph.cGraph
      let _streamContext = graph.streamContext?._stream
      let cTensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, _streamContext)!
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

    var dataType: DataType {
      let _graph = graph.cGraph
      let info = ccv_nnc_tensor_variable_params(_graph, _tensor)
      return DataType.from(cTensorParams: info)
    }

    /**
     * A constant tensor can only be used as input, you cannot compute gradients
     * for a constant tensor.
     */
    public var isConstant: Bool {
      let _graph = graph.cGraph
      return ccv_nnc_tensor_variable_is_constant(_graph, _tensor) == 1
    }

    /**
     * Whether this tensor is contiguous in memory.
     */
    public var isContiguous: Bool {
      let _graph = graph.cGraph
      let _streamContext = graph.streamContext?._stream
      let cTensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, _streamContext)!
      let type = Int(cTensor.pointee.type)
      guard (type & CCV_TENSOR_VIEW) == CCV_TENSOR_VIEW else {
        return true
      }
      let cTensorView = UnsafeRawPointer(cTensor).assumingMemoryBound(
        to: ccv_nnc_tensor_view_t.self)
      return cTensorView.pointee.contiguous == 1
    }
  }

  /**
   * A typed tensor variable.
   */
  public final class Tensor<Element: TensorNumeric>: AnyTensor {
    // This is to help speed up, there is no need to have only one rawValue.
    private weak var _rawValue: NNC.AnyTensorStorage? = nil

    /**
     * Get the underlying tensor. If not available, create one.
     */
    public var rawValue: NNC.Tensor<Element> {
      if let rawValue = _rawValue {
        return NNC.Tensor<Element>(rawValue)
      }
      let _graph = graph.cGraph
      let _streamContext = graph.streamContext?._stream
      let tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, _streamContext)!
      let rawValue = NNC.AnyTensorStorage(tensor, original: self, selfOwned: false)  // To enforce copy-on-write syntax.
      _rawValue = rawValue
      return NNC.Tensor<Element>(rawValue)
    }

    // If we did type conversion, we need to hold a reference to its parent.
    public subscript(indices: Int...) -> Element {
      get {
        if let rawValue = _rawValue {
          return rawValue[indices, Element.self]
        }
        let _graph = graph.cGraph
        let _streamContext = graph.streamContext?._stream
        let tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, _streamContext)!
        let rawValue = NNC.AnyTensorStorage(tensor, original: self, selfOwned: false)  // To enforce copy-on-write syntax.
        _rawValue = rawValue
        return rawValue[indices, Element.self]
      }
      set(v) {
        if let rawValue = _rawValue {
          rawValue[indices, Element.self] = v
        }
        let _graph = graph.cGraph
        let _streamContext = graph.streamContext?._stream
        let tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, _streamContext)!
        let rawValue = NNC.AnyTensorStorage(tensor, original: self, selfOwned: false)  // To enforce copy-on-write syntax.
        _rawValue = rawValue
        rawValue[indices, Element.self] = v
      }
    }

    public var typeErased: AnyTensor { self }
  }

  public let cGraph: OpaquePointer
  var streamContext: StreamContext? = nil

  /**
   * The workspace size for executions. This is helpful when execute models, which will use this
   * to tune the optimal kernel.
   */
  public var workspaceSize: Int = 0

  public var maxConcurrency: StreamContext.Concurrency = .noLimit {
    didSet {
      ccv_nnc_dynamic_graph_set_max_concurrency(cGraph, Int32(maxConcurrency.rawValue))
    }
  }

  struct WeakAnyTensor {
    weak var value: AnyTensor?
  }
  var trackGrads = [ObjectIdentifier: WeakAnyTensor]()

  public init() {
    CmdParamsFactory.factory.sink()
    cGraph = ccv_nnc_dynamic_graph_new()
  }

  deinit {
    ccv_nnc_dynamic_graph_free(cGraph)
  }
}

extension DynamicGraph {
  func trackGrad(_ tensor: AnyTensor) {
    let weakAnyTensor = WeakAnyTensor(value: tensor)
    trackGrads[ObjectIdentifier(tensor)] = weakAnyTensor
  }
  func untrackGrad(_ objectIdentifier: ObjectIdentifier) {
    trackGrads[objectIdentifier] = nil
  }
  func gradients<S: Sequence>(for tensors: S) -> [AnyTensor] where S.Element: AnyTensor {
    let values = trackGrads.values.compactMap { $0.value }
    guard values.count > 0 else { return [] }
    var bitmask = [UInt64](repeating: 0, count: (values.count + 63) / 64)
    let sources: [ccv_nnc_tensor_variable_t?] = values.map { $0._tensor }
    let destinations: [ccv_nnc_tensor_variable_t?] = tensors.map { $0._tensor }
    bitmask.withUnsafeMutableBufferPointer { buffer in
      ccv_nnc_dynamic_graph_has_effect_to_tensor_variables(
        cGraph, sources, Int32(sources.count), destinations, Int32(destinations.count),
        buffer.baseAddress)
    }
    var gradients = [AnyTensor]()
    for (i, value) in values.enumerated() {
      if bitmask[i / 64] & 1 << (i & 63) != 0 {
        gradients.append(value)
      }
    }
    return gradients
  }
}

extension DynamicGraph {
  public enum LogLevel {
    /// No log output (the default).
    case none
    /// Verbose, show all computations and its values.
    case verbose
    /// Show all computations.
    case info
    /// Only show errors if encountered any.
    case error
  }
  /**
   * Set the log level on a dynamic graph.
   */
  public static var logLevel: LogLevel {
    get {
      let cliLevels = ccv_cli_get_output_levels()
      if (cliLevels & 1) != 0 {
        return .verbose
      } else if (cliLevels & 2) != 0 {
        return .info
      } else if (cliLevels & 4) != 0 {
        return .error
      }
      return .none
    }
    set(v) {
      switch v {
      case .none:
        ccv_cli_set_output_levels(0)
      case .verbose:
        ccv_cli_set_output_levels(ccv_cli_output_level_and_above(Int32(CCV_CLI_VERBOSE)))
      case .info:
        ccv_cli_set_output_levels(ccv_cli_output_level_and_above(Int32(CCV_CLI_INFO)))
      case .error:
        ccv_cli_set_output_levels(ccv_cli_output_level_and_above(Int32(CCV_CLI_ERROR)))
      }
    }
  }
}

extension DynamicGraph {
  /// Set the seed for global context.
  public static func setSeed(_ seed: UInt32) {
    ccv_nnc_stream_context_set_seed(nil, seed)
  }
}

extension DynamicGraph {
  /// Create a synchronization point with the underlying stream context.
  public func joined() {
    streamContext?.joined()
  }
}

extension DynamicGraph {
  /// The watermark for in-flight Metal operations.
  public static var queueWatermark: Int {
    set { ccv_nnc_set_queue_watermark(Int32(newValue)) }
    get { Int(ccv_nnc_queue_watermark()) }
  }
  /// Set whether to enable profiler or not.
  public static func setProfiler(_ on: Bool) {
    ccv_nnc_set_profiler(on ? 1 : 0)
  }
}

extension DynamicGraph {
  /// Statistics about the graph.
  public struct Statistics {
    /// How many variables (including constants) in this graph.
    public var variables: Int
    /// How many computation units in this graph.
    public var computations: Int
  }
  /**
   * Collect statistics from a dynamic graph. It computes how many variables and computations
   * are still tracked. If you have memory leaks, this is useful to track down that.
   */
  public var statistics: Statistics {
    let variables = ccv_nnc_dynamic_graph_bookkeeping_count(cGraph, Int32(CCV_NNC_SYMBOL_TENSOR))
    let computations = ccv_nnc_dynamic_graph_bookkeeping_count(
      cGraph, Int32(CCV_NNC_SYMBOL_GRAPH_EXEC))
    return Statistics(variables: Int(variables), computations: Int(computations))
  }
}

func == (lhs: DynamicGraph.WeakAnyTensor, rhs: DynamicGraph.WeakAnyTensor) -> Bool {
  return lhs.value === rhs.value
}

extension DynamicGraph.WeakAnyTensor: Hashable {
  func hash(into hasher: inout Hasher) {
    guard let value = value else { return }
    ObjectIdentifier(value).hash(into: &hasher)
  }
}

extension DynamicGraph.AnyTensor {

  /**
   * Create a new tensor variable with dimensions permuted.
   *
   * - Parameters:
   *   - indices: The indices for dimensions from the original tensor. For example, a [2, 3, 4] tensor with [2, 0, 1] indices will permute to a [4, 2, 3] tensor.
   * - Returns: The new tensor variable with dimensions permuted.
   */
  public func permuted(_ indices: Int...) -> Self {
    let _graph = graph.cGraph
    let cTensorParams = ccv_nnc_tensor_variable_params(_graph, _tensor)
    let device = DeviceKind.from(cTensorParams: cTensorParams)
    let dataType = DataType.from(cTensorParams: cTensorParams)
    let shape = self.shape
    var cOffset:
      (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) = (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      )
    var cStrides:
      (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) = (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      )
    let isAlias =
      (withUnsafeMutablePointer(to: &cOffset) { offset -> Int32 in
        let offset = UnsafeMutableRawPointer(offset).assumingMemoryBound(to: Int32.self)
        return withUnsafeMutablePointer(to: &cStrides) { strides -> Int32 in
          let strides = UnsafeMutableRawPointer(strides).assumingMemoryBound(to: Int32.self)
          return ccv_nnc_tensor_variable_alias_params(_graph, _tensor, offset, strides)
        }
      }) == 0
    let offset = TensorShape(dims: cOffset)
    var strides = TensorShape(dims: cStrides)
    if !isAlias || strides.count == 0 {
      var stride = 1
      for i in (0..<shape.count).reversed() {
        strides[i] = stride
        stride *= shape[i]
      }
    }
    var newOffset = offset
    var newShape = shape
    var newStrides = strides
    for (i, index) in indices.enumerated() {
      newShape[i] = shape[index]
      newOffset[i] = offset[index]
      newStrides[i] = strides[index]
    }
    cOffset = newOffset.dims
    cStrides = newStrides.dims
    let _alias = withUnsafePointer(to: &cOffset) { offset -> ccv_nnc_tensor_variable_t in
      let offset = UnsafeRawPointer(offset).assumingMemoryBound(to: Int32.self)
      return withUnsafePointer(to: &cStrides) { strides -> ccv_nnc_tensor_variable_t in
        let strides = UnsafeRawPointer(strides).assumingMemoryBound(to: Int32.self)
        return ccv_nnc_tensor_variable_alias_new(
          _graph, _tensor, offset, strides,
          toCTensorParams(
            device, dataType: dataType, format: format, shape: newShape))!
      }
    }
    return Self(graph: underlying.graph, tensor: _alias, original: self)
  }

  public func reshaped(
    format: TensorFormat, shape: TensorShape, offset: TensorShape? = nil,
    strides: TensorShape? = nil
  ) -> Self {
    var shape = shape
    if let first = shape.firstIndex(of: -1) {
      precondition(shape.filter { $0 == -1 }.count == 1)
      let numElements = self.shape.reduce(1, *)
      let known = shape.reduce(1) { $1 == -1 ? $0 : $0 * $1 }
      precondition(known > 0 && numElements % known == 0)
      shape[first] = numElements / known
    }
    let _graph = graph.cGraph
    let cTensorParams = ccv_nnc_tensor_variable_params(_graph, _tensor)
    let device = DeviceKind.from(cTensorParams: cTensorParams)
    let dataType = DataType.from(cTensorParams: cTensorParams)
    var cOffset = offset?.dims ?? (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    var cStrides = strides?.dims ?? (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    var oldStrides:
      (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) = (
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      )
    let isAlias =
      (withUnsafeMutablePointer(to: &oldStrides) { strides -> Int32 in
        let strides = UnsafeMutableRawPointer(strides).assumingMemoryBound(to: Int32.self)
        return ccv_nnc_tensor_variable_alias_params(_graph, _tensor, nil, strides)
      }) == 0
    if isAlias && (shape.count != self.shape.count || (strides != nil && strides != self.strides)) {
      // Check if strides not permuted. If it is permuted (and we shape to different sizes or have different strides), we need to first make a copy and then reshape again.
      let oldStrides = TensorShape(dims: oldStrides)
      if oldStrides.count > 0 {
        for i in 1..<oldStrides.count {
          // Otherwise we cannot reshape, need to first make a copy and then reshape.
          precondition(
            oldStrides[i - 1] >= oldStrides[i],
            "The tensor is permuted, cannot reshape to \(shape), try .copied() before reshape.")
        }
      }
    }
    let _alias = withUnsafePointer(to: &cOffset) { offset -> ccv_nnc_tensor_variable_t in
      let offset = UnsafeRawPointer(offset).assumingMemoryBound(to: Int32.self)
      return withUnsafePointer(to: &cStrides) { strides -> ccv_nnc_tensor_variable_t in
        let strides = UnsafeRawPointer(strides).assumingMemoryBound(to: Int32.self)
        return ccv_nnc_tensor_variable_alias_new(
          _graph, _tensor, offset, strides,
          toCTensorParams(
            device, dataType: dataType, format: format, shape: shape))!
      }
    }
    return Self(graph: underlying.graph, tensor: _alias, original: self)
  }

  /**
   * Create a new tensor representing the same variable but with different sizes.
   *
   * - Parameters:
   *   - shapeFormat: New format and shape for the tensor.
   *   - offset: Whether offset on each shape.
   *   - strides: The stride on each shape.
   * - Returns: The new tensor with different format but the same underlying variable.
   */
  public func reshaped(
    _ shapeFormat: TensorShapeFormat, offset: TensorShape? = nil, strides: TensorShape? = nil
  ) -> Self {
    return reshaped(
      format: shapeFormat.format, shape: shapeFormat.shape, offset: offset,
      strides: strides)
  }

}

extension DynamicGraph.AnyTensor: CustomStringConvertible {
  public var description: String {
    let _graph = graph.cGraph
    let cTensorParams = ccv_nnc_tensor_variable_params(_graph, _tensor)
    if cTensorParams.datatype == 0 {
      return "DynamicGraph.AutoTensor"
    } else {
      let dataType = DataType.from(cTensorParams: cTensorParams)
      let format = TensorFormat.from(cTensorParams: cTensorParams)
      let device = DeviceKind.from(cTensorParams: cTensorParams)
      let shape = fromCDimensions(cTensorParams.dim)
      return
        "DynamicGraph.Tensor<\(dataType)>(kind: .\(device), format: .\(format), shape: \(shape))"
    }
  }
}

extension DynamicGraph.AnyTensor: CustomDebugStringConvertible {
  public var debugDescription: String {
    let _graph = graph.cGraph
    let _streamContext = graph.streamContext?._stream
    guard var tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, _streamContext) else {
      return description
    }
    let cTensorParams = tensor.pointee.info
    var _output: UnsafeMutablePointer<ccv_nnc_tensor_t>? = nil
    if DeviceKind.from(cTensorParams: cTensorParams) != .CPU {
      var _input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = tensor
      tensor = ccv_nnc_tensor_new(
        nil,
        toCTensorParams(.CPU, dataType: dataType, format: format, shape: shape),
        0)
      _output = tensor
      let cmd = ccv_nnc_cmd(
        CCV_NNC_DATA_TRANSFER_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0)
      ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, _streamContext)
      // Need to wait the stream to be done so we can print current ones.
      ccv_nnc_stream_context_wait(_streamContext)
    }
    defer {
      if _output != nil {
        ccv_nnc_tensor_free(_output)
      }
    }
    guard let cString = ccv_nnc_tensor_format_new(tensor) else {
      return description
    }
    let debugDescription = description + " " + String(cString: cString)
    free(cString)
    return debugDescription
  }
}

extension DynamicGraph {

  /**
   * Create a placeholder variable. It doesn't have shape and can only
   * be used as output.
   */
  public func variable() -> AnyTensor {
    let _tensor = ccv_nnc_tensor_variable_new_impl(cGraph, ccv_nnc_tensor_auto)!
    let tensor = AnyTensor(graph: self, tensor: _tensor)
    return tensor
  }

  /**
   * Create a placeholder constant. It doesn't have shape and can only
   * be used as output.
   */
  public func constant() -> AnyTensor {
    let _tensor = ccv_nnc_tensor_constant_new_impl(cGraph, ccv_nnc_tensor_auto)!
    return AnyTensor(graph: self, tensor: _tensor)
  }

  /**
   * Create a new variable from an existing tensor.
   *
   * - Parameter tensor: The existing tensor.
   * - Returns: Created new tensor variable.
   */
  public func variable<Element: TensorNumeric>(_ tensor: NNC.Tensor<Element>) -> Tensor<Element> {
    let _tensor = ccv_nnc_tensor_variable_new_impl(cGraph, ccv_nnc_tensor_auto)!
    ccv_nnc_tensor_variable_set(cGraph, _tensor, tensor.cTensor)
    // Retain the tensor until we freed the variable.
    ccv_nnc_tensor_variable_destructor_hook(
      cGraph, _tensor,
      { _, _, ctx in
        // No longer need to retain the tensor.
        Unmanaged<NNC.AnyTensorStorage>.fromOpaque(ctx!).release()
      }, Unmanaged.passRetained(tensor.storage).toOpaque())
    let tensor = Tensor<Element>(graph: self, tensor: _tensor)
    return tensor
  }

  /**
   * Create a new constant from an existing tensor.
   *
   * - Parameter tensor: The existing tensor.
   * - Returns: Created new tensor constant.
   */
  public func constant<Element: TensorNumeric>(_ tensor: NNC.Tensor<Element>) -> Tensor<Element> {
    let _tensor = ccv_nnc_tensor_constant_new_impl(cGraph, ccv_nnc_tensor_auto)!
    ccv_nnc_tensor_variable_set(cGraph, _tensor, tensor.cTensor)
    // Retain the tensor until we freed the variable.
    ccv_nnc_tensor_variable_destructor_hook(
      cGraph, _tensor,
      { _, _, ctx in
        // No longer need to retain the tensor.
        Unmanaged<NNC.AnyTensorStorage>.fromOpaque(ctx!).release()
      }, Unmanaged.passRetained(tensor.storage).toOpaque())
    return Tensor<Element>(graph: self, tensor: _tensor)
  }

  /**
   * Create a grouped variable from an array of existing tensors.
   *
   * - Parameter tensors: The array of existing tensors.
   * - Returns: Newly created grouped variable.
   */
  public func variable<Element: TensorNumeric>(_ tensors: [NNC.Tensor<Element>])
    -> DynamicGraph.Group<
      Tensor<Element>
    >
  {
    precondition(tensors.count > 0)
    return DynamicGraph.Group(tensors.map { self.variable($0) })
  }

  /**
   * Create a grouped constant from an array of existing tensors.
   *
   * - Parameter tensors: The array of existing tensors.
   * - Returns: Newly created grouped constant.
   */
  public func constant<Element: TensorNumeric>(_ tensors: [NNC.Tensor<Element>])
    -> DynamicGraph.Group<
      Tensor<Element>
    >
  {
    precondition(tensors.count > 0)
    return DynamicGraph.Group(tensors.map { self.constant($0) })
  }

  public func variable<Element: TensorNumeric>(
    _ device: DeviceKind, format: TensorFormat, shape: TensorShape, of: Element.Type = Element.self
  ) -> Tensor<Element> {
    let _tensor = ccv_nnc_tensor_variable_new_impl(
      cGraph,
      toCTensorParams(device, dataType: Element.dataType, format: format, shape: shape))!
    return Tensor<Element>(graph: self, tensor: _tensor)
  }

  public func constant<Element: TensorNumeric>(
    _ device: DeviceKind, format: TensorFormat, shape: TensorShape, of: Element.Type = Element.self
  ) -> Tensor<Element> {
    let tensor = ccv_nnc_tensor_constant_new_impl(
      cGraph,
      toCTensorParams(device, dataType: Element.dataType, format: format, shape: shape))!
    return Tensor<Element>(graph: self, tensor: tensor)
  }

  public func variable<Element: TensorNumeric>(
    _ device: DeviceKind, _ shapeFormat: TensorShapeFormat, of: Element.Type = Element.self
  ) -> Tensor<Element> {
    return variable(device, format: shapeFormat.format, shape: shapeFormat.shape)
  }

  public func constant<Element: TensorNumeric>(
    _ device: DeviceKind, _ shapeFormat: TensorShapeFormat, of: Element.Type = Element.self
  ) -> Tensor<Element> {
    return constant(device, format: shapeFormat.format, shape: shapeFormat.shape)
  }

}

extension DynamicGraph {
  public func variable<T: DynamicGraph.TensorGroup>(like: T) -> T {
    let graph = like.graph
    switch like {
    case is DynamicGraph.AnyTensor:
      return graph.variable(
        like.kind, format: like.format, shape: like.shape, of: T.ElementNumeric.self)
        as! T
    case is DynamicGraph.AnyGroup:
      return DynamicGraph.Group(
        like.untyped.map {
          graph.variable(
            $0.kind, format: $0.format, shape: $0.shape, of: T.ElementNumeric.self)
        }) as! T
    default:
      fatalError("Cannot support the given type")
    }
  }
  public func constant<T: DynamicGraph.TensorGroup>(like: T) -> T {
    let graph = like.graph
    switch like {
    case is DynamicGraph.AnyTensor:
      return graph.constant(
        like.kind, format: like.format, shape: like.shape, of: T.ElementNumeric.self)
        as! T
    case is DynamicGraph.AnyGroup:
      return DynamicGraph.Group(
        like.untyped.map {
          graph.constant(
            $0.kind, format: $0.format, shape: $0.shape, of: T.ElementNumeric.self)
        }) as! T
    default:
      fatalError("Cannot support the given type")
    }
  }
}

extension DynamicGraph {
  /**
   * Turn off gradient tracking within the given closure. This may be useful during testing, we can
   * make more aggressive optimizations if the gradient tracking is off.
   */
  public func withNoGrad<Result>(_ closure: () throws -> Result) rethrows -> Result {
    let noGrad = ccv_nnc_dynamic_graph_set_no_grad(cGraph, 1)
    let result = try closure()
    if noGrad == 0 {  // Only set it back if we previously set it ourselves.
      ccv_nnc_dynamic_graph_set_no_grad(cGraph, 0)
    }
    return result
  }
}

extension DynamicGraph {
  /**
   * Perform operations on a given stream within the closure. Each operation can take a stream context
   * parameter, however, that often error-prune. This method make sure all operations within the closure
   * will be dispatched to the given stream context, making it easier to organize.
   */
  public func withStream<Result>(_ streamContext: StreamContext, _ closure: () throws -> Result)
    rethrows -> Result
  {
    self.streamContext = streamContext
    let result = try closure()
    self.streamContext = nil
    return result
  }
}

extension DynamicGraph {
  /**
   * Perform garbage collection. This will be done automatically when GPU memory at pressure. Do
   * it manually helps to identify actual memory leaks.
   */
  public func garbageCollect() {
    ccv_nnc_dynamic_graph_gc(cGraph)
  }
}

extension DynamicGraph {
  public struct EnableBits: OptionSet, CaseIterable {
    public let rawValue: UInt64
    public init(rawValue: UInt64) {
      self.rawValue = rawValue
    }
    /**
     * Disable mixing MPSMatrixMultiplication with MPSGraph.matrixMultiplication.
     */
    public static let disableMixedMPSGEMM = EnableBits(
      rawValue: UInt64(CCV_NNC_DISABLE_MIXED_MPS_GEMM))
    /**
     * Disable mixing MPSMatrixSoftMax with MPSGraph.softMax.
     */
    public static let disableMixedMPSSoftMax = EnableBits(
      rawValue: UInt64(CCV_NNC_DISABLE_MIXED_MPS_SOFTMAX))
    /**
     * Disable memory-mapped MTLBuffer.
     */
    public static let disableMmapMTLBuffer = EnableBits(
      rawValue: UInt64(CCV_NNC_DISABLE_MMAP_MTL_BUFFER))
    /**
     * Disable all MFA shaders.
     */
    public static let disableMFA = EnableBits(
      rawValue: UInt64(CCV_NNC_DISABLE_MFA))
    /**
     * Disable MFA GEMM shader.
     */
    public static let disableMFAGEMM = EnableBits(
      rawValue: UInt64(CCV_NNC_DISABLE_MFA_GEMM))
    /**
     * Disable MFA attention shader.
     */
    public static let disableMFAAttention = EnableBits(
      rawValue: UInt64(CCV_NNC_DISABLE_MFA_ATTENTION))
    /**
     * Disable MFA Neural Accelerators optimizations.
     */
    public static let disableMFANeuralAccelerators = EnableBits(
      rawValue: UInt64(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
    public static let allCases: [EnableBits] = [
      .disableMixedMPSGEMM, .disableMixedMPSSoftMax, .disableMmapMTLBuffer,
      .disableMFA, .disableMFAGEMM, .disableMFAAttention, .disableMFANeuralAccelerators,
    ]
  }

  /**
   * Set system-wide flag.
   */
  public static var flags: EnableBits {
    get { EnableBits(rawValue: ccv_nnc_flags()) }
    set {
      let oldValue = EnableBits(rawValue: ccv_nnc_flags())
      for flag in EnableBits.allCases {
        if oldValue.contains(flag) && !newValue.contains(flag) {
          ccv_nnc_disable_flag(flag.rawValue)
        } else if !oldValue.contains(flag) && newValue.contains(flag) {
          ccv_nnc_enable_flag(flag.rawValue)
        }
      }
    }
  }

  public struct BinaryArtifacts: Equatable {
    public var pathsToRead: [String]
    public var pathToWrite: String?
    init(pathsToRead: [String], pathToWrite: String?) {
      self.pathsToRead = pathsToRead
      self.pathToWrite = pathToWrite
    }
  }
  /**
   * Set binary artifacts that will be used to speed up the compilations.
   */
  public static var binaryArtifacts = BinaryArtifacts(pathsToRead: [], pathToWrite: nil) {
    didSet {
      guard binaryArtifacts != oldValue else { return }
      var pathsToRead: [UnsafePointer<CChar>?] = binaryArtifacts.pathsToRead.map { UnsafePointer(strdup($0)) }
      pathsToRead.withUnsafeMutableBufferPointer {
        ccv_nnc_set_binary_artifacts($0.baseAddress, Int32(binaryArtifacts.pathsToRead.count), binaryArtifacts.pathToWrite)
      }
      for pathToRead in pathsToRead {
        free(UnsafeMutablePointer(mutating: pathToRead))
      }
    }
  }
}
