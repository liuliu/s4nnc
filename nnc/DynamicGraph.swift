import C_nnc

public final class DynamicGraph {

  fileprivate class _AnyTensor {

    let graph: DynamicGraph
    let _tensor: ccv_nnc_tensor_variable_t

    init(graph: DynamicGraph, tensor: ccv_nnc_tensor_variable_t) {
      self.graph = graph
      _tensor = tensor
    }

    deinit {
      ccv_nnc_tensor_variable_free(graph._graph, _tensor)
    }
  }

  public class AnyTensor {

    internal(set) public var grad: AnyTensor? = nil

    fileprivate let underlying: _AnyTensor

    var graph: DynamicGraph { underlying.graph }
    var _tensor: ccv_nnc_tensor_variable_t { underlying._tensor }

    fileprivate init(graph: DynamicGraph, tensor: ccv_nnc_tensor_variable_t) {
      underlying = _AnyTensor(graph: graph, tensor: tensor)
    }

    fileprivate init(_ underlying: _AnyTensor) {
      self.underlying = underlying
    }

    public var dimensions: [Int] {
        let _graph = graph._graph
      let info = ccv_nnc_tensor_variable_params(_graph, _tensor)
      return fromCDimensions(info.dim)
    }
  }

  public final class Tensor<Element: TensorNumeric>: AnyTensor {
    private weak var _rawValue: nnc._AnyTensor? = nil

    public var rawValue: nnc.Tensor<Element> {
      if let rawValue = _rawValue {
        return nnc.Tensor<Element>(rawValue)
      }
      let _graph = graph._graph
      let tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)!
      let rawValue = nnc._AnyTensor(tensor, original: self) // To enforce copy-on-write syntax.
      _rawValue = rawValue
      return nnc.Tensor<Element>(rawValue)
    }

    // If we did type conversion, we need to hold a reference to its parent.
    public convenience init(_ tensor: AnyTensor) {
      self.init(tensor.underlying)
    }

    public subscript(indices: Int...) -> Element {
      get {
        if let rawValue = _rawValue {
          return rawValue[indices, Element.self]
        }
        let _graph = graph._graph
        let tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)!
        let rawValue = nnc._AnyTensor(tensor, original: self) // To enforce copy-on-write syntax.
        _rawValue = rawValue
        return rawValue[indices, Element.self]
      }
      set(v) {
        if let rawValue = _rawValue {
          rawValue[indices, Element.self] = v
        }
        let _graph = graph._graph
        let tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)!
        let rawValue = nnc._AnyTensor(tensor, original: self) // To enforce copy-on-write syntax.
        _rawValue = rawValue
        rawValue[indices, Element.self] = v
      }
    }
  }

  let _graph: OpaquePointer

  public init() {
    CmdParamsFactory.factory.sink()
    _graph = ccv_nnc_dynamic_graph_new()
  }

  deinit {
    ccv_nnc_dynamic_graph_free(_graph)
  }
}

public extension DynamicGraph {
  enum LogLevel {
    case none
    case verbose
    case info
    case error
  }
  var logLevel: LogLevel {
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

public extension DynamicGraph {
  struct Statistics {
    var variables: Int
    var computations: Int
  }
  var statistics: Statistics {
    let variables = ccv_nnc_dynamic_graph_bookkeeping_count(_graph, Int32(CCV_NNC_SYMBOL_TENSOR))
    let computations = ccv_nnc_dynamic_graph_bookkeeping_count(_graph, Int32(CCV_NNC_SYMBOL_GRAPH_EXEC))
    return Statistics(variables: Int(variables), computations: Int(computations))
  }
}

public func ==(lhs: DynamicGraph.AnyTensor, rhs: DynamicGraph.AnyTensor) -> Bool {
  return lhs === rhs
}

extension DynamicGraph.AnyTensor: Hashable {
  public func hash(into hasher: inout Hasher) {
    ObjectIdentifier(self).hash(into: &hasher)
  }
}

public extension DynamicGraph.Tensor {

  func reshape(format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    let _graph = graph._graph
    let cTensorParams = ccv_nnc_tensor_variable_params(_graph, _tensor)
    let device = DeviceKind.from(cTensorParams: cTensorParams)
    var offset = toCDimensions(offset)
    var increments = toCDimensions(increments)
    let _alias = withUnsafePointer(to: &offset.0) { offset in
      withUnsafePointer(to: &increments.0) { increments in
        ccv_nnc_tensor_variable_alias_new(_graph, _tensor, offset, increments,
          toCTensorParams(device, dataType: Element.dataType, format: format, dimensions: dimensions))!
      }
    }
    return Self(graph: underlying.graph, tensor: _alias)
  }

  func reshape(_ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return reshape(format: dimensionFormat.format, dimensions: dimensionFormat.dimensions, offset: offset, increments: increments)
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
      return "DynamicGraph.Tensor<\(dataType)>(kind: .\(device), format: .\(format), dimensions: \(dimensions))"
    }
  }
}

public extension DynamicGraph {

  func variable() -> AnyTensor {
    let _tensor = ccv_nnc_tensor_variable_new_impl(_graph, ccv_nnc_tensor_auto)!
    let tensor = AnyTensor(graph: self, tensor: _tensor)
    return tensor
  }

  func constant() -> AnyTensor {
    let _tensor = ccv_nnc_tensor_constant_new_impl(_graph, ccv_nnc_tensor_auto)!
    return AnyTensor(graph: self, tensor: _tensor)
  }

  func variable<Element: TensorNumeric>(_ tensor: nnc.Tensor<Element>) -> Tensor<Element> {
    let _tensor = ccv_nnc_tensor_variable_new_impl(_graph, ccv_nnc_tensor_auto)!
    ccv_nnc_tensor_variable_set(_graph, _tensor, tensor.underlying._tensor)
    // Retain the tensor until we freed the variable.
    ccv_nnc_tensor_variable_owner_hook(_graph, _tensor, { _, _, owner, ctx in
      guard owner == nil else { return }
      // No longer need to retain the tensor.
      Unmanaged<nnc._AnyTensor>.fromOpaque(ctx!).release()
    }, Unmanaged.passRetained(tensor.underlying).toOpaque())
    let tensor = Tensor<Element>(graph: self, tensor: _tensor)
    return tensor
  }

  func constant<Element: TensorNumeric>(_ tensor: nnc.Tensor<Element>) -> Tensor<Element> {
    let _tensor = ccv_nnc_tensor_constant_new_impl(_graph, ccv_nnc_tensor_auto)!
    ccv_nnc_tensor_variable_set(_graph, _tensor, tensor.underlying._tensor)
    // Retain the tensor until we freed the variable.
    ccv_nnc_tensor_variable_owner_hook(_graph, _tensor, { _, _, owner, ctx in
      guard owner == nil else { return }
      // No longer need to retain the tensor.
      Unmanaged<nnc._AnyTensor>.fromOpaque(ctx!).release()
    }, Unmanaged.passRetained(tensor.underlying).toOpaque())
    return Tensor<Element>(graph: self, tensor: _tensor)
  }

  func variable<Element: TensorNumeric>(_ device: DeviceKind, format: TensorFormat, dimensions: [Int]) -> Tensor<Element> {
    let _tensor = ccv_nnc_tensor_variable_new_impl(_graph,
      toCTensorParams(device, dataType: Element.dataType, format: format, dimensions: dimensions))!
    let tensor = Tensor<Element>(graph: self, tensor: _tensor)
    return tensor
  }

  func constant<Element: TensorNumeric>(_ device: DeviceKind, format: TensorFormat, dimensions: [Int]) -> Tensor<Element> {
    let tensor = ccv_nnc_tensor_constant_new_impl(_graph,
      toCTensorParams(device, dataType: Element.dataType, format: format, dimensions: dimensions))!
    return Tensor<Element>(graph: self, tensor: tensor)
  }

  func variable<Element: TensorNumeric>(_ device: DeviceKind, _ dimensionFormat: TensorDimensionFormat) -> Tensor<Element> {
    return variable(device, format: dimensionFormat.format, dimensions: dimensionFormat.dimensions)
  }

  func constant<Element: TensorNumeric>(_ device: DeviceKind, _ dimensionFormat: TensorDimensionFormat) -> Tensor<Element> {
    return constant(device, format: dimensionFormat.format, dimensions: dimensionFormat.dimensions)
  }

}

public extension DynamicGraph {
  func withNoGrad<Result>(_ closure: () throws -> Result) rethrows -> Result {
    ccv_nnc_dynamic_graph_set_no_grad(_graph, 1)
    let result = try closure()
    ccv_nnc_dynamic_graph_set_no_grad(_graph, 0)
    return result
  }
}
