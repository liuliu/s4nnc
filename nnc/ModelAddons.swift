import C_nnc

public final class Sum: Model {
  public init(name: String = "") {
    super.init(ccv_cnnp_sum(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ inputs: T..., streamContext: StreamContext? = nil
  ) -> T {
    precondition(inputs.count >= 2)
    let outputs = self(
      inputs: inputs[0], Array(inputs.suffix(from: 1)), streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Add: Model {
  public init(leftScalar: Float = 1, rightScalar: Float = 1, name: String = "") {
    super.init(ccv_cnnp_add(leftScalar, rightScalar, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Mul: Model {
  public init(scalar: Float = 1, name: String = "") {
    super.init(ccv_cnnp_mul(scalar, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Matmul: Model {
  public init(transposeA: (Int, Int) = (0, 0), transposeB: (Int, Int) = (0, 0), name: String = "") {
    let a = [Int32(transposeA.0), Int32(transposeA.1)]
    let b = [Int32(transposeB.0), Int32(transposeB.1)]
    super.init(ccv_cnnp_matmul(a, b, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Dense: Model {
  public init(count: Int, noBias: Bool = false, name: String = "") {
    super.init(ccv_cnnp_dense(Int32(count), noBias ? 1 : 0, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Reshape: Model {
  public init(dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil, name: String = "")
  {
    var dimensions = toCDimensions(dimensions)
    var offset = toCDimensions(offset)
    var increments = toCDimensions(increments)
    let _model = withUnsafePointer(to: &dimensions.0) { dimensions in
      withUnsafePointer(to: &offset.0) { offset in
        withUnsafePointer(to: &increments.0) { increments in
          ccv_cnnp_reshape(dimensions, offset, increments, name)!
        }
      }
    }
    super.init(_model)
  }
}

extension Model.IO {
  public func reshape(_ dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil)
    -> Model.IO
  {
    return Reshape(dimensions: dimensions, offset: offset, increments: increments)(self)
  }
}

public final class RELU: Model {
  public init(name: String = "") {
    super.init(ccv_cnnp_relu(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Softmax: Model {
  public init(name: String = "") {
    super.init(ccv_cnnp_softmax(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Sigmoid: Model {
  public init(name: String = "") {
    super.init(ccv_cnnp_sigmoid(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Swish: Model {
  public init(name: String = "") {
    super.init(ccv_cnnp_swish(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Transpose: Model {
  public init(_ axisA: Int, _ axisB: Int, name: String = "") {
    super.init(ccv_cnnp_transpose(Int32(axisA), Int32(axisB), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension Model.IO {
  public func transpose(_ axisA: Int, _ axisB: Int) -> Model.IO {
    return Transpose(axisA, axisB)(self)
  }
}

public final class MaskedFill: Model {
  public init(equalTo: Float, fillWith: Float, name: String = "") {
    super.init(ccv_cnnp_masked_fill(equalTo, fillWith, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    _ left: T, _ right: U, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Dropout: Model {
  public init(probability: Float, entirety: Bool = false, name: String = "") {
    super.init(ccv_cnnp_dropout(probability, entirety ? 1 : 0, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Scalmul: Model {
  public init(_ a: Float, name: String = "") {
    super.init(ccv_cnnp_scalar_mul(a, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class BatchNorm: Model {
  public init(momentum: Float, epsilon: Float, name: String = "") {
    super.init(ccv_cnnp_batch_norm(momentum, epsilon, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class LayerNorm: Model {
  public init(epsilon: Float, axis: [Int], name: String = "") {
    let axis32: [Int32] = axis.map { Int32($0) }
    super.init(ccv_cnnp_layer_norm(epsilon, axis32, Int32(axis.count), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Flatten: Model {
  public init(name: String = "") {
    super.init(ccv_cnnp_flatten(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class Convolution: Model {
  public init(
    groups: Int, filters: Int, filterSize: [Int], noBias: Bool = false, hint: Hint = Hint(),
    name: String = ""
  ) {
    let kdim = toCDimensionsArray(filterSize)
    super.init(
      ccv_cnnp_convolution(
        Int32(groups), Int32(filters), kdim, noBias ? 1 : 0, hint.toCHint(), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class MaxPool: Model {
  public init(filterSize: [Int] = [], hint: Hint = Hint(), name: String = "") {
    let kdim = toCDimensionsArray(filterSize)
    super.init(ccv_cnnp_max_pool(kdim, hint.toCHint(), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

public final class AveragePool: Model {
  public init(filterSize: [Int] = [], hint: Hint = Hint(), name: String = "") {
    let kdim = toCDimensionsArray(filterSize)
    super.init(ccv_cnnp_average_pool(kdim, hint.toCHint(), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}
