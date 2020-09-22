import C_nnc

public final class DynamicGraph {

  public class AnyTensor {

    let graph: DynamicGraph
    let _tensor: ccv_nnc_tensor_variable_t

    private let owner: AnyTensor?

    init(graph: DynamicGraph, tensor: ccv_nnc_tensor_variable_t, owner: AnyTensor? = nil) {
      self.graph = graph
      _tensor = tensor
      self.owner = owner
    }

    deinit {
      guard owner == nil else { return }
      ccv_nnc_tensor_variable_free(graph._graph, _tensor)
    }
  }

  public final class Tensor<Element: TensorNumeric>: AnyTensor {
    public var rawValue: nnc.Tensor<Element> {
      let _graph = graph._graph
      let tensor = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)!
      return nnc.Tensor<Element>(tensor, owner: self)
    }

    // If we did type conversion, we need to hold a reference to its parent.
    public convenience init(_ tensor: AnyTensor) {
      self.init(graph: tensor.graph, tensor: tensor._tensor, owner: tensor)
    }
  }

  let _graph: OpaquePointer

  public init() {
    ccv_nnc_init()
    _graph = ccv_nnc_dynamic_graph_new()
  }

  deinit {
    ccv_nnc_dynamic_graph_free(_graph)
  }
}

public extension DynamicGraph {

  func variable() -> AnyTensor {
    let tensor = ccv_nnc_tensor_variable_new_impl(_graph, ccv_nnc_tensor_auto)!
    return AnyTensor(graph: self, tensor: tensor)
  }

  func constant() -> AnyTensor {
    let tensor = ccv_nnc_tensor_constant_new_impl(_graph, ccv_nnc_tensor_auto)!
    return AnyTensor(graph: self, tensor: tensor)
  }

  func variable<Element: TensorNumeric>(_ tensor: nnc.Tensor<Element>) -> Tensor<Element> {
    let _tensor = ccv_nnc_tensor_variable_new_impl(_graph, ccv_nnc_tensor_auto)!
    ccv_nnc_tensor_variable_set(_graph, _tensor, tensor._tensor)
    // Retain the tensor until we freed the variable.
    ccv_nnc_tensor_variable_owner_hook(_graph, _tensor, { _, _, owner, ctx in
      guard owner == nil else { return }
      // No longer need to retain the tensor.
      Unmanaged<nnc.AnyTensor>.fromOpaque(ctx!).release()
    }, Unmanaged.passRetained(tensor).toOpaque())
    return Tensor<Element>(graph: self, tensor: _tensor)
  }

  func constant<Element: TensorNumeric>(_ tensor: nnc.Tensor<Element>) -> Tensor<Element> {
    let _tensor = ccv_nnc_tensor_constant_new_impl(_graph, ccv_nnc_tensor_auto)!
    ccv_nnc_tensor_variable_set(_graph, _tensor, tensor._tensor)
    // Retain the tensor until we freed the variable.
    ccv_nnc_tensor_variable_owner_hook(_graph, _tensor, { _, _, owner, ctx in
      guard owner == nil else { return }
      // No longer need to retain the tensor.
      Unmanaged<nnc.AnyTensor>.fromOpaque(ctx!).release()
    }, Unmanaged.passRetained(tensor).toOpaque())
    return Tensor<Element>(graph: self, tensor: _tensor)
  }

  func variable<Element: TensorNumeric>(_ device: DeviceKind, format: TensorFormat, dimensions: [Int]) -> Tensor<Element> {
    let tensor = ccv_nnc_tensor_variable_new_impl(_graph,
      toCTensorParams(device, dataType: Element.dataType, format: format, dimensions: dimensions))!
    return Tensor<Element>(graph: self, tensor: tensor)
  }

  func constant<Element: TensorNumeric>(_ device: DeviceKind, format: TensorFormat, dimensions: [Int]) -> Tensor<Element> {
    let tensor = ccv_nnc_tensor_constant_new_impl(_graph,
      toCTensorParams(device, dataType: Element.dataType, format: format, dimensions: dimensions))!
    return Tensor<Element>(graph: self, tensor: tensor)
  }

  func variable<Element: TensorNumeric>(_ device: DeviceKind, _ dimensionFormat: TensorDimensionFormat) -> Tensor<Element> {
    switch dimensionFormat {
    case let .NHWC(n, h, w, c):
      return variable(device, format: .NHWC, dimensions: [n, h, w, c])
    case let .NCHW(n, c, h, w):
      return variable(device, format: .NCHW, dimensions: [n, c, h, w])
    case let .CHWN(c, h, w, n):
      return variable(device, format: .CHWN, dimensions: [c, h, w, n])
    }
  }

  func constant<Element: TensorNumeric>(_ device: DeviceKind, _ dimensionFormat: TensorDimensionFormat) -> Tensor<Element> {
    switch dimensionFormat {
    case let .NHWC(n, h, w, c):
      return constant(device, format: .NHWC, dimensions: [n, h, w, c])
    case let .NCHW(n, c, h, w):
      return constant(device, format: .NCHW, dimensions: [n, c, h, w])
    case let .CHWN(c, h, w, n):
      return constant(device, format: .CHWN, dimensions: [c, h, w, n])
    }
  }

}
