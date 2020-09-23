import C_nnc

public final class Sum: Model {
  public init(name: String = "") {
    super.init(ccv_cnnp_sum(name))
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

  public func callAsFunction<Element: TensorNumeric>(_ left: DynamicGraph.Tensor<Element>, _ right: DynamicGraph.Tensor<Element>) -> DynamicGraph.Tensor<Element> {
    let outputs = self([left, right])
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

public final class Mul: Model {
  public init(name: String = "") {
    super.init(ccv_cnnp_mul(name))
  }

  public func callAsFunction<Element: TensorNumeric>(_ left: DynamicGraph.Tensor<Element>, _ right: DynamicGraph.Tensor<Element>) -> DynamicGraph.Tensor<Element> {
    let outputs = self([left, right])
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

public final class Matmul: Model {
  public init(transposeA: (Int, Int) = (0, 0), transposeB: (Int, Int) = (0, 0), name: String = "") {
    let a = [Int32(transposeA.0), Int32(transposeA.1)]
    let b = [Int32(transposeB.0), Int32(transposeB.1)]
    super.init(ccv_cnnp_matmul(a, b, name))
  }

  public func callAsFunction<Element: TensorNumeric>(_ left: DynamicGraph.Tensor<Element>, _ right: DynamicGraph.Tensor<Element>) -> DynamicGraph.Tensor<Element> {
    let outputs = self([left, right])
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
