import C_nnc

/// Sum inputs.
public final class Sum: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// Add two inputs together. It will do broadcast if needed.
public final class Add: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// Multiply two inputs together. It will do broadcast if needed.
public final class Mul: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// Matrix-multiplication over two inputs.
public final class Matmul: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// A linear layer model.
public final class Dense: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// A reshape model.
public final class Reshape: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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
  /**
   * Reshape an IO to a new dimension. You cannot reshape data types.
   *
   * - Parameters:
   *   - dimensions: The new dimensions for the input.
   *   - offset: Whether apply certain offset for each dimension.
   *   - increments: What's the step size for each dimension.
   */
  public func reshaped(_ dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil)
    -> Model.IO
  {
    return Reshape(dimensions: dimensions, offset: offset, increments: increments)(self)
  }
}

/// A ReLU activation model.
public final class ReLU: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

private typealias _ReLU = ReLU

extension Model.IO {
  /**
   * Apply ReLU activation to the said IO.
   */
  public func ReLU() -> Model.IO {
    return _ReLU()(self)
  }
}

/// A softmax activation model.
public final class Softmax: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

extension Model.IO {
  /**
   * Apply softmax activation to the said IO.
   */
  public func softmax() -> Model.IO {
    return Softmax()(self)
  }
}

/// A sigmoid activation model.
public final class Sigmoid: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

extension Model.IO {
  /**
   * Apply sigmoid activation to the said IO.
   */
  public func sigmoid() -> Model.IO {
    return Sigmoid()(self)
  }
}

/// A tanh activation model.
public final class Tanh: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(name: String = "") {
    super.init(ccv_cnnp_tanh(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension Model.IO {
  /**
   * Apply tanh activation to the said IO.
   */
  public func tanh() -> Model.IO {
    return Tanh()(self)
  }
}

/// A swish activation model.
public final class Swish: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

extension Model.IO {
  /**
   * Apply swish activation to the said IO.
   */
  public func swish() -> Model.IO {
    return Swish()(self)
  }
}

public final class Transpose: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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
  public func transposed(_ axisA: Int, _ axisB: Int) -> Model.IO {
    return Transpose(axisA, axisB)(self)
  }
}

/// The masked fill model. If the value equal to a given constant, fill with another constant.
public final class MaskedFill: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// The dropout model.
public final class Dropout: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// Multiply all values with a constant.
public final class Scalmul: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// Batch normalization model.
public final class BatchNorm: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// Layer normalization model.
public final class LayerNorm: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// Make the input tensor to be 1-D tensor (respecting N).
public final class Flatten: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// Convolution model.
public final class Convolution: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// max pooling model.
public final class MaxPool: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// average pooling model.
public final class AveragePool: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

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

/// reduce sum model.
public final class ReduceSum: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(axis: [Int], name: String = "") {
    precondition(axis.count > 0)
    super.init(ccv_cnnp_reduce_sum(axis.map { Int32($0) }, Int32(axis.count), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

/// reduce max model.
public final class ReduceMax: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(axis: [Int], name: String = "") {
    precondition(axis.count > 0)
    super.init(ccv_cnnp_reduce_max(axis.map { Int32($0) }, Int32(axis.count), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension Model.IO {
  public func reduced(_ op: ReduceOp, axis: [Int]) -> Model.IO {
    switch op {
    case .sum:
      return ReduceSum(axis: axis)(self)
    case .max:
      return ReduceMax(axis: axis)(self)
    }
  }
}

/// min model.
public final class Min: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(name: String = "") {
    super.init(ccv_cnnp_min(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension Functional {
  public static func min(_ left: Model.IO, _ right: Model.IO) -> Model.IO {
    return Min()(left, right)
  }
}

/// max model.
public final class Max: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(name: String = "") {
    super.init(ccv_cnnp_max(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension Functional {
  public static func max(_ left: Model.IO, _ right: Model.IO) -> Model.IO {
    return Max()(left, right)
  }
}

/// Concatenate model.
public final class Concat: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(axis: Int, name: String = "") {
    super.init(ccv_cnnp_concat(Int32(axis), name))
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

extension Functional {
  public static func concat(axis: Int, _ inputs: Model.IO...) -> Model.IO {
    return Concat(axis: axis).apply(inputs)
  }
}

/// LSTM model.
public final class LSTM: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(
    masked: Bool, hiddenSize: Int, numberOfLayers: Int, projectSize: Int? = nil, bias: Bool = true,
    batchFirst: Bool = true, bidirectional: Bool = false, dropout: Float? = nil, name: String = ""
  ) {
    super.init(
      ccv_cnnp_lstm(
        masked ? 1 : 0, Int32(hiddenSize), Int32(projectSize ?? 0), Int32(numberOfLayers),
        bias ? 1 : 0, batchFirst ? 1 : 0, bidirectional ? 1 : 0, dropout ?? 0, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    _ x: T, mask: U, streamContext: StreamContext? = nil
  ) -> T where U.ElementNumeric == Int32, T.AnyTensor == U.AnyTensor {
    let outputs = self(inputs: x, mask, streamContext: streamContext)
    return T(outputs[0])
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ x: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: x, streamContext: streamContext)
    return T(outputs[0])
  }
}

/// Embedding model.
public final class Embedding: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init<T: TensorNumeric>(
    _ dataType: T.Type, vocabularySize: Int, embeddingSize: Int, name: String = ""
  ) {
    super.init(
      ccv_cnnp_embedding(T.dataType.toC, Int32(vocabularySize), Int32(embeddingSize), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    _ x: U, streamContext: StreamContext? = nil
  ) -> T where U.ElementNumeric == Int32, T.AnyTensor == U.AnyTensor {
    let outputs = self(inputs: x, streamContext: streamContext)
    return T(outputs[0])
  }
}

/// IndexSelect model.
public final class IndexSelect: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(name: String = "") {
    super.init(ccv_cnnp_index_select(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    _ x: T, index: U, streamContext: StreamContext? = nil
  ) -> T where U.ElementNumeric == Int32, T.AnyTensor == U.AnyTensor {
    let outputs = self(inputs: x, index, streamContext: streamContext)
    return T(outputs[0])
  }
}
