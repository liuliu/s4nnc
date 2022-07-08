import C_nnc

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
      ccv_nnc_tensor_variable_free(graph._graph, _tensor)
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

    public var dimensions: [Int] {
      let _graph = graph._graph
      let info = ccv_nnc_tensor_variable_params(_graph, _tensor)
      return fromCDimensions(info.dim)
    }

    public var kind: DeviceKind {
      let _graph = graph._graph
      let info = ccv_nnc_tensor_variable_params(_graph, _tensor)
      return DeviceKind.from(cTensorParams: info)
    }

    public var format: TensorFormat {
      let _graph = graph._graph
      let info = ccv_nnc_tensor_variable_params(_graph, _tensor)
      return TensorFormat.from(cTensorParams: info)
    }

    public var increments: [Int] {
      let _graph = graph._graph
      let _streamContext = graph.streamContext?._stream
      let cTensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, _streamContext)!
      return fromCTensorIncrements(cTensor)
    }

    /**
     * A constant tensor can only be used as input, you cannot compute gradients
     * for a constant tensor.
     */
    public var isConstant: Bool {
      let _graph = graph._graph
      return ccv_nnc_tensor_variable_is_constant(_graph, _tensor) == 1
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
      let _graph = graph._graph
      let _streamContext = graph.streamContext?._stream
      let tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, _streamContext)!
      let rawValue = NNC.AnyTensorStorage(tensor, original: self)  // To enforce copy-on-write syntax.
      _rawValue = rawValue
      return NNC.Tensor<Element>(rawValue)
    }

    // If we did type conversion, we need to hold a reference to its parent.
    public subscript(indices: Int...) -> Element {
      get {
        if let rawValue = _rawValue {
          return rawValue[indices, Element.self]
        }
        let _graph = graph._graph
        let _streamContext = graph.streamContext?._stream
        let tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, _streamContext)!
        let rawValue = NNC.AnyTensorStorage(tensor, original: self)  // To enforce copy-on-write syntax.
        _rawValue = rawValue
        return rawValue[indices, Element.self]
      }
      set(v) {
        if let rawValue = _rawValue {
          rawValue[indices, Element.self] = v
        }
        let _graph = graph._graph
        let _streamContext = graph.streamContext?._stream
        let tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, _streamContext)!
        let rawValue = NNC.AnyTensorStorage(tensor, original: self)  // To enforce copy-on-write syntax.
        _rawValue = rawValue
        rawValue[indices, Element.self] = v
      }
    }
  }

  let _graph: OpaquePointer
  var streamContext: StreamContext? = nil

  struct WeakAnyTensor {
    weak var value: AnyTensor?
  }
  var trackGrads = [ObjectIdentifier: WeakAnyTensor]()

  public init() {
    CmdParamsFactory.factory.sink()
    _graph = ccv_nnc_dynamic_graph_new()
  }

  deinit {
    ccv_nnc_dynamic_graph_free(_graph)
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
        _graph, sources, Int32(sources.count), destinations, Int32(destinations.count),
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
    let variables = ccv_nnc_dynamic_graph_bookkeeping_count(_graph, Int32(CCV_NNC_SYMBOL_TENSOR))
    let computations = ccv_nnc_dynamic_graph_bookkeeping_count(
      _graph, Int32(CCV_NNC_SYMBOL_GRAPH_EXEC))
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

  public func reshaped(
    format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil
  ) -> Self {
    let _graph = graph._graph
    let cTensorParams = ccv_nnc_tensor_variable_params(_graph, _tensor)
    let device = DeviceKind.from(cTensorParams: cTensorParams)
    let dataType = DataType.from(cTensorParams: cTensorParams)
    var offset = toCDimensions(offset)
    var increments = toCDimensions(increments)
    let _alias = withUnsafePointer(to: &offset.0) { offset in
      withUnsafePointer(to: &increments.0) { increments in
        ccv_nnc_tensor_variable_alias_new(
          _graph, _tensor, offset, increments,
          toCTensorParams(
            device, dataType: dataType, format: format, dimensions: dimensions))!
      }
    }
    return Self(graph: underlying.graph, tensor: _alias, original: self)
  }

  /**
   * Create a new tensor representing the same variable but with different sizes.
   *
   * - Parameters:
   *   - dimensionFormat: New format and dimensions for the tensor.
   *   - offset: Whether offset on each dimensions.
   *   - increments: The step on each dimensions.
   * - Returns: The new tensor with different format but the same underlying variable.
   */
  public func reshaped(
    _ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil
  ) -> Self {
    return reshaped(
      format: dimensionFormat.format, dimensions: dimensionFormat.dimensions, offset: offset,
      increments: increments)
  }

}

extension DynamicGraph.AnyTensor: CustomStringConvertible {
  public var description: String {
    let _graph = graph._graph
    let cTensorParams = ccv_nnc_tensor_variable_params(_graph, _tensor)
    if cTensorParams.datatype == 0 {
      return "DynamicGraph.AutoTensor"
    } else {
      let dataType = DataType.from(cTensorParams: cTensorParams)
      let format = TensorFormat.from(cTensorParams: cTensorParams)
      let device = DeviceKind.from(cTensorParams: cTensorParams)
      let dimensions = fromCDimensions(cTensorParams.dim)
      return
        "DynamicGraph.Tensor<\(dataType)>(kind: .\(device), format: .\(format), dimensions: \(dimensions))"
    }
  }
}

extension DynamicGraph {

  /**
   * Create a placeholder variable. It doesn't have shape and can only
   * be used as output.
   */
  public func variable() -> AnyTensor {
    let _tensor = ccv_nnc_tensor_variable_new_impl(_graph, ccv_nnc_tensor_auto)!
    let tensor = AnyTensor(graph: self, tensor: _tensor)
    return tensor
  }

  /**
   * Create a placeholder constant. It doesn't have shape and can only
   * be used as output.
   */
  public func constant() -> AnyTensor {
    let _tensor = ccv_nnc_tensor_constant_new_impl(_graph, ccv_nnc_tensor_auto)!
    return AnyTensor(graph: self, tensor: _tensor)
  }

  /**
   * Create a new variable from an existing tensor.
   *
   * - Parameter tensor: The existing tensor.
   * - Returns: Created new tensor variable.
   */
  public func variable<Element: TensorNumeric>(_ tensor: NNC.Tensor<Element>) -> Tensor<Element> {
    let _tensor = ccv_nnc_tensor_variable_new_impl(_graph, ccv_nnc_tensor_auto)!
    ccv_nnc_tensor_variable_set(_graph, _tensor, tensor.cTensor)
    // Retain the tensor until we freed the variable.
    ccv_nnc_tensor_variable_destructor_hook(
      _graph, _tensor,
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
    let _tensor = ccv_nnc_tensor_constant_new_impl(_graph, ccv_nnc_tensor_auto)!
    ccv_nnc_tensor_variable_set(_graph, _tensor, tensor.cTensor)
    // Retain the tensor until we freed the variable.
    ccv_nnc_tensor_variable_destructor_hook(
      _graph, _tensor,
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
    _ device: DeviceKind, format: TensorFormat, dimensions: [Int], of: Element.Type = Element.self
  ) -> Tensor<Element> {
    let _tensor = ccv_nnc_tensor_variable_new_impl(
      _graph,
      toCTensorParams(device, dataType: Element.dataType, format: format, dimensions: dimensions))!
    return Tensor<Element>(graph: self, tensor: _tensor)
  }

  public func constant<Element: TensorNumeric>(
    _ device: DeviceKind, format: TensorFormat, dimensions: [Int], of: Element.Type = Element.self
  ) -> Tensor<Element> {
    let tensor = ccv_nnc_tensor_constant_new_impl(
      _graph,
      toCTensorParams(device, dataType: Element.dataType, format: format, dimensions: dimensions))!
    return Tensor<Element>(graph: self, tensor: tensor)
  }

  public func variable<Element: TensorNumeric>(
    _ device: DeviceKind, _ dimensionFormat: TensorDimensionFormat, of: Element.Type = Element.self
  ) -> Tensor<Element> {
    return variable(device, format: dimensionFormat.format, dimensions: dimensionFormat.dimensions)
  }

  public func constant<Element: TensorNumeric>(
    _ device: DeviceKind, _ dimensionFormat: TensorDimensionFormat, of: Element.Type = Element.self
  ) -> Tensor<Element> {
    return constant(device, format: dimensionFormat.format, dimensions: dimensionFormat.dimensions)
  }

}

extension DynamicGraph {
  public func variable<T: DynamicGraph.TensorGroup>(like: T) -> T {
    let graph = like.graph
    switch like {
    case is DynamicGraph.AnyTensor:
      return graph.variable(
        like.kind, format: like.format, dimensions: like.dimensions, of: T.ElementNumeric.self)
        as! T
    case is DynamicGraph.AnyGroup:
      return DynamicGraph.Group(
        (0..<like.untyped.count).map { _ in
          graph.variable(
            like.kind, format: like.format, dimensions: like.dimensions, of: T.ElementNumeric.self)
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
        like.kind, format: like.format, dimensions: like.dimensions, of: T.ElementNumeric.self)
        as! T
    case is DynamicGraph.AnyGroup:
      return DynamicGraph.Group(
        (0..<like.untyped.count).map { _ in
          graph.constant(
            like.kind, format: like.format, dimensions: like.dimensions, of: T.ElementNumeric.self)
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
    ccv_nnc_dynamic_graph_set_no_grad(_graph, 1)
    let result = try closure()
    ccv_nnc_dynamic_graph_set_no_grad(_graph, 0)
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
