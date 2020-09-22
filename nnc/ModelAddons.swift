import C_nnc

public final class Mul: Model {
  public init(name: String = "") {
    super.init(ccv_cnnp_mul(name))
  }

  public func callAsFunction<Element: TensorNumeric>(_ inputs: DynamicGraph.Tensor<Element>...) -> DynamicGraph.Tensor<Element> {
    let outputs = self(inputs)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

public final class Add: Model {
  public init(name: String = "") {
    super.init(ccv_cnnp_add(name))
  }

  public func callAsFunction<Element: TensorNumeric>(_ inputs: DynamicGraph.Tensor<Element>...) -> DynamicGraph.Tensor<Element> {
    let outputs = self(inputs)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

public final class Dense: Model {
  public init(count: Int, noBias: Bool = false, name: String = "") {
    var params = ccv_cnnp_param_t()
    params.no_bias = noBias ? 1 : 0
    super.init(ccv_cnnp_dense(Int32(count), params, name))
  }

  public func callAsFunction<Element: TensorNumeric>(_ input: DynamicGraph.Tensor<Element>) -> DynamicGraph.Tensor<Element> {
    let outputs = self([input])
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}
