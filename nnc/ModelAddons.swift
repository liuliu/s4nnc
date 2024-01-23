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

extension Functional {
  public static func mul(left: ModelIOConvertible, right: ModelIOConvertible, scalar: Float)
    -> Model.IO
  {
    return Mul(scalar: scalar)(left, right)
  }
}

/// Div two inputs together. It will not do broadcast.
public final class Div: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(reciprocal: Bool = false, name: String = "") {
    super.init(ccv_cnnp_div(reciprocal ? 1 : 0, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension ModelIOConvertible {
  /**
   * Compute the reciprocal for a model IO.
   */
  public func reciprocal() -> Model.IO {
    return Div(reciprocal: true)(self)
  }
}

/// Square root of a input. It will not do broadcast.
public final class SquareRoot: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(name: String = "") {
    super.init(ccv_cnnp_sqrt(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension ModelIOConvertible {
  /**
   * Compute the reciprocal for a model IO.
   */
  public func squareRoot() -> Model.IO {
    return SquareRoot()(self)
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

extension Functional {
  public static func matmul(
    left: ModelIOConvertible, right: ModelIOConvertible, leftTranspose: (Int, Int),
    rightTranspose: (Int, Int)
  ) -> Model.IO {
    return Matmul(transposeA: leftTranspose, transposeB: rightTranspose)(left, right)
  }
}

/// Comlex number multiplication over two inputs.
public final class Cmul: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(name: String = "") {
    super.init(ccv_cnnp_cmul(name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension Functional {
  public static func cmul(left: ModelIOConvertible, right: ModelIOConvertible) -> Model.IO {
    return Cmul()(left, right)
  }
}

/// A linear layer model.
public final class Dense: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(count: Int, noBias: Bool = false, trainable: Bool? = nil, name: String = "") {
    super.init(
      ccv_cnnp_dense(
        Int32(count), noBias ? 1 : 0, trainable == true ? 1 : (trainable == false ? 0 : -1), name))
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

  public init(
    dimensions: TensorShape, offset: TensorShape? = nil, strides: TensorShape? = nil,
    format: TensorFormat? = nil, name: String = ""
  ) {
    var dimensions = dimensions.dims
    var offset = offset?.dims ?? (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    var strides = strides?.dims ?? (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let _model = withUnsafePointer(to: &dimensions) { dimensions -> OpaquePointer in
      let dimensions = UnsafeRawPointer(dimensions).assumingMemoryBound(to: Int32.self)
      return withUnsafePointer(to: &offset) { offset -> OpaquePointer in
        let offset = UnsafeRawPointer(offset).assumingMemoryBound(to: Int32.self)
        return withUnsafePointer(to: &strides) { strides -> OpaquePointer in
          let strides = UnsafeRawPointer(strides).assumingMemoryBound(to: Int32.self)
          return ccv_cnnp_reshape(format?.toC ?? 0, dimensions, offset, strides, name)!
        }
      }
    }
    super.init(_model)
  }
}

extension ModelIOConvertible {
  /**
   * Reshape an IO to a new dimension. You cannot reshape data types.
   *
   * - Parameters:
   *   - dimensions: The new dimensions for the input.
   *   - offset: Whether apply certain offset for each dimension.
   *   - strides: What's the stride for each dimension.
   *   - format: What's the new format of the tensor.
   */
  public func reshaped(
    _ dimensions: TensorShape, offset: TensorShape? = nil, strides: TensorShape? = nil,
    format: TensorFormat? = nil
  )
    -> Model.IO
  {
    return Reshape(dimensions: dimensions, offset: offset, strides: strides, format: format)(self)
  }
  /**
   * Reshape an IO to a new dimension. You cannot reshape data types.
   *
   * - Parameters:
   *   - shape: The new dimensions and format for the input.
   *   - offset: Whether apply certain offset for each dimension.
   *   - strides: What's the stride for each dimension.
   *   - format: What's the new format of the tensor.
   */
  public func reshaped(
    _ shape: TensorShapeFormat, offset: TensorShape? = nil, strides: TensorShape? = nil
  )
    -> Model.IO
  {
    return Reshape(dimensions: shape.shape, offset: offset, strides: strides, format: shape.format)(
      self)
  }
}

/// A identity model.
public final class Identity: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(name: String = "") {
    super.init(ccv_cnnp_identity(name))
  }
}

extension ModelIOConvertible {
  /**
   * Identity op for a model IO. This doesn't do anything but to change the order of execution.
   */
  public func identity() -> Model.IO {
    return Identity()(self)
  }
}

/// A permute model.
public final class Permute: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(indices: [Int], name: String = "") {
    var indices = toCDimensions(indices)
    let _model = withUnsafePointer(to: &indices) { indices -> OpaquePointer in
      let indices = UnsafeRawPointer(indices).assumingMemoryBound(to: Int32.self)
      return ccv_cnnp_permute(indices, name)!
    }
    super.init(_model)
  }
}

extension ModelIOConvertible {
  /**
   * Permute an IO according to the indices.
   *
   * - Parameters:
   *   - indices: The dimensions to pick from the input.
   */
  public func permuted(_ indices: Int...)
    -> Model.IO
  {
    return Permute(indices: indices)(self)
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

extension ModelIOConvertible {
  /**
   * Apply ReLU activation to the said IO.
   */
  public func ReLU() -> Model.IO {
    return _ReLU()(self)
  }
}

/// A leaky ReLU activation model.
public final class LeakyReLU: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(negativeSlope: Float, name: String = "") {
    super.init(ccv_cnnp_leaky_relu(negativeSlope, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension ModelIOConvertible {
  /**
   * Apply leaky ReLU activation to the said IO.
   */
  public func leakyReLU(negativeSlope: Float) -> Model.IO {
    return LeakyReLU(negativeSlope: negativeSlope)(self)
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

extension ModelIOConvertible {
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

extension ModelIOConvertible {
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

extension ModelIOConvertible {
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

extension ModelIOConvertible {
  /**
   * Apply swish activation to the said IO.
   */
  public func swish() -> Model.IO {
    return Swish()(self)
  }
}

/// A GELU activation model.
public final class GELU: Model {
  public enum Approximate {
    case none
    case tanh
  }

  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(approximate: Approximate = .none, name: String = "") {
    super.init(ccv_cnnp_gelu(approximate == .tanh ? 1 : 0, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

private typealias _GELU = GELU

extension ModelIOConvertible {
  /**
   * Apply GELU activation to the said IO.
   */
  public func GELU(approximate: GELU.Approximate = .none) -> Model.IO {
    return _GELU(approximate: approximate)(self)
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

extension ModelIOConvertible {
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

  public init(momentum: Float, epsilon: Float, trainable: Bool? = nil, name: String = "") {
    super.init(
      ccv_cnnp_batch_norm(
        momentum, epsilon, trainable == true ? 1 : (trainable == false ? 0 : -1), name))
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

  public init(epsilon: Float, axis: [Int], trainable: Bool? = nil, name: String = "") {
    let axis32: [Int32] = axis.map { Int32($0) }
    super.init(
      ccv_cnnp_layer_norm(
        epsilon, axis32, Int32(axis.count), trainable == true ? 1 : (trainable == false ? 0 : -1),
        name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

/// Group normalization model.
public final class GroupNorm: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(
    axis: Int, groups: Int, epsilon: Float, reduce: [Int], trainable: Bool? = nil, name: String = ""
  ) {
    let axis32: [Int32] = reduce.map { Int32($0) }
    super.init(
      ccv_cnnp_group_norm(
        Int32(axis), Int32(groups), epsilon, axis32, Int32(axis32.count),
        trainable == true ? 1 : (trainable == false ? 0 : -1), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

/// RMSNorm model.
public final class RMSNorm: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(epsilon: Float, axis: [Int], trainable: Bool? = nil, name: String = "") {
    let axis32: [Int32] = axis.map { Int32($0) }
    super.init(
      ccv_cnnp_rmsnorm(
        epsilon, axis32, Int32(axis.count), trainable == true ? 1 : (trainable == false ? 0 : -1),
        name))
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

  public enum Format {  // These are more understandable names than reuse TensorFormat. We will translate to TensorFormat.
    case OIHW
    case OHWI
  }

  public init(
    groups: Int, filters: Int, filterSize: [Int], dilation: [Int] = [], noBias: Bool = false,
    hint: Hint = Hint(), format: Format? = nil, trainable: Bool? = nil, name: String = ""
  ) {
    let kdim = toCDimensionsArray(filterSize)
    let dilation = toCDimensionsArray(dilation)
    let format: TensorFormat? = format.map {
      switch $0 {
      case .OIHW:
        return TensorFormat.NCHW
      case .OHWI:
        return TensorFormat.NHWC
      }
    }
    super.init(
      ccv_cnnp_convolution(
        Int32(groups), Int32(filters), kdim, dilation, noBias ? 1 : 0, hint.toCHint(),
        format?.toC ?? 0, trainable == true ? 1 : (trainable == false ? 0 : -1), name)
    )
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

/// Convolution Transpose model.
public final class ConvolutionTranspose: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(
    groups: Int, filters: Int, filterSize: [Int], dilation: [Int] = [], outputPadding: Int = 0,
    noBias: Bool = false, hint: Hint = Hint(), format: Convolution.Format? = nil,
    trainable: Bool? = nil, name: String = ""
  ) {
    let kdim = toCDimensionsArray(filterSize)
    let dilation = toCDimensionsArray(dilation)
    let format: TensorFormat? = format.map {
      switch $0 {
      case .OIHW:
        return TensorFormat.NCHW
      case .OHWI:
        return TensorFormat.NHWC
      }
    }
    super.init(
      ccv_cnnp_convolution_transpose(
        Int32(groups), Int32(filters), kdim, dilation, Int32(outputPadding), noBias ? 1 : 0,
        hint.toCHint(), format?.toC ?? 0, trainable == true ? 1 : (trainable == false ? 0 : -1),
        name)
    )
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

/// upsample model.
public final class Upsample: Model {
  public enum Mode {
    case nearest
    case bilinear
    var rawValue: Int32 {
      switch self {
      case .nearest:
        return Int32(CCV_NNC_UPSAMPLE_NEAREST)
      case .bilinear:
        return Int32(CCV_NNC_UPSAMPLE_BILINEAR)
      }
    }
  }

  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(_ mode: Mode, widthScale: Float, heightScale: Float, name: String = "") {
    super.init(ccv_cnnp_upsample(mode.rawValue, widthScale, heightScale, name))
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

/// reduce mean model.
public final class ReduceMean: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(axis: [Int], name: String = "") {
    precondition(axis.count > 0)
    super.init(ccv_cnnp_reduce_mean(axis.map { Int32($0) }, Int32(axis.count), name))
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

/// reduce min model.
public final class ReduceMin: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(axis: [Int], name: String = "") {
    precondition(axis.count > 0)
    super.init(ccv_cnnp_reduce_min(axis.map { Int32($0) }, Int32(axis.count), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

/// reduce norm2 model.
public final class ReduceNorm2: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(axis: [Int], name: String = "") {
    precondition(axis.count > 0)
    super.init(ccv_cnnp_reduce_norm2(axis.map { Int32($0) }, Int32(axis.count), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ input: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: input, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension ModelIOConvertible {
  public func reduced(_ op: ReduceOp, axis: [Int]) -> Model.IO {
    switch op {
    case .sum:
      return ReduceSum(axis: axis)(self)
    case .mean:
      return ReduceMean(axis: axis)(self)
    case .max:
      return ReduceMax(axis: axis)(self)
    case .min:
      return ReduceMin(axis: axis)(self)
    case .norm2:
      return ReduceNorm2(axis: axis)(self)
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
  public static func min(_ left: ModelIOConvertible, _ right: ModelIOConvertible) -> Model.IO {
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
  public static func max(_ left: ModelIOConvertible, _ right: ModelIOConvertible) -> Model.IO {
    return Max()(left, right)
  }
}

/// Extract model.
public final class Extract: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(_ index: Int, name: String = "") {
    super.init(ccv_cnnp_extract(Int32(index), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension ModelIOConvertible {
  public subscript(index: Int) -> Model.IO {
    return Extract(index)(self)
  }
}

/// Argmax model.
public final class Argmax: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(axis: Int, name: String = "") {
    super.init(ccv_cnnp_argmax(Int32(axis), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension ModelIOConvertible {
  public func argmax(axis: Int) -> Model.IO {
    return Argmax(axis: axis)(self)
  }
}

/// Argmin model.
public final class Argmin: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(axis: Int, name: String = "") {
    super.init(ccv_cnnp_argmin(Int32(axis), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    let outputs = self(inputs: left, right, streamContext: streamContext)
    return T(outputs[0])
  }
}

extension ModelIOConvertible {
  public func argmin(axis: Int) -> Model.IO {
    return Argmin(axis: axis)(self)
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
  public static func concat(axis: Int, _ inputs: ModelIOConvertible...) -> Model.IO {
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
    batchFirst: Bool = true, bidirectional: Bool = false, dropout: Float? = nil,
    trainable: Bool? = nil, name: String = ""
  ) {
    super.init(
      ccv_cnnp_lstm(
        masked ? 1 : 0, Int32(hiddenSize), Int32(projectSize ?? 0), Int32(numberOfLayers),
        bias ? 1 : 0, batchFirst ? 1 : 0, bidirectional ? 1 : 0, dropout ?? 0,
        trainable == true ? 1 : (trainable == false ? 0 : -1), name))
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
    _ dataType: T.Type, vocabularySize: Int, embeddingSize: Int, trainable: Bool? = nil,
    name: String = ""
  ) {
    super.init(
      ccv_cnnp_embedding(
        T.dataType.toC, Int32(vocabularySize), Int32(embeddingSize),
        trainable == true ? 1 : (trainable == false ? 0 : -1), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    _ x: U, streamContext: StreamContext? = nil
  ) -> T where U.ElementNumeric == Int32, T.AnyTensor == U.AnyTensor {
    let outputs = self(inputs: x, streamContext: streamContext)
    return T(outputs[0])
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    _ x: U, streamContext: StreamContext? = nil
  ) -> T where U.ElementNumeric == Float32, T.AnyTensor == U.AnyTensor {
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

  public func callAsFunction<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    _ x: T, index: U, streamContext: StreamContext? = nil
  ) -> T where U.ElementNumeric == Float32, T.AnyTensor == U.AnyTensor {
    let outputs = self(inputs: x, index, streamContext: streamContext)
    return T(outputs[0])
  }
}

/// DatatypeConversion model.
public final class DatatypeConversion: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(_ dataType: DataType?, sameAsLast: Bool = false, name: String = "") {
    super.init(ccv_cnnp_datatype_conversion(dataType?.toC ?? 0, sameAsLast ? 1 : 0, name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    _ x: T, sameAs: U, streamContext: StreamContext? = nil
  ) -> U where T.AnyTensor == U.AnyTensor {
    let outputs = self(inputs: x, sameAs, streamContext: streamContext)
    return U(outputs[0])
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    _ x: T, of type: U.ElementNumeric.Type, sameAs: U.Type = U.self,
    streamContext: StreamContext? = nil
  ) -> U where T.AnyTensor == U.AnyTensor {
    let outputs = self(inputs: x, streamContext: streamContext)
    return U(outputs[0])
  }
}

extension ModelIOConvertible {
  /**
   * Convert an IO to a new datatype.
   *
   * - Parameters:
   *   - datatype: The new datatype for the input.
   */
  public func to(_ dataType: DataType) -> Model.IO {
    return DatatypeConversion(dataType)(self)
  }
  /**
   * Convert an IO to a new datatype.
   *
   * - Parameters:
   *   - of: The other ModelIO which will share the same input.
   */
  public func to(of other: ModelIOConvertible) -> Model.IO {
    return DatatypeConversion(nil, sameAsLast: true)(self, other)
  }
}

extension ModelIOConvertible {
  func clamped(
    min: Float?, max: Float?
  ) -> Model.IO {
    precondition(min != nil || max != nil)
    return Model(ccv_cnnp_clamp(min ?? Float.nan, max ?? Float.nan, nil))(self)
  }

  /// Clamp the given model IO between two values.
  public func clamped(_ range: ClosedRange<Float>)
    -> Model.IO
  {
    return clamped(min: range.lowerBound, max: range.upperBound)
  }

  /// Clamp the given model IO with a lower bound.
  public func clamped(_ range: PartialRangeFrom<Float>)
    -> Model.IO
  {
    return clamped(min: range.lowerBound, max: nil)
  }

  /// Clamp the given model IO with an upper bound.
  public func clamped(_ range: PartialRangeThrough<Float>)
    -> Model.IO
  {
    return clamped(min: nil, max: range.upperBound)
  }
}

/// Parameter model.
public final class Parameter<Element: TensorNumeric>: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(
    _ kind: DeviceKind, format: TensorFormat, shape: TensorShape, initBound: Float = 0,
    trainable: Bool? = nil, name: String = ""
  ) {
    super.init(
      ccv_cnnp_parameter(
        toCTensorParams(kind, dataType: Element.dataType, format: format, shape: shape), initBound,
        trainable == true ? 1 : (trainable == false ? 0 : -1), name))
  }

  public init(
    _ kind: DeviceKind, _ dimensionFormat: TensorShapeFormat, initBound: Float = 0,
    trainable: Bool? = nil, name: String = ""
  ) {
    super.init(
      ccv_cnnp_parameter(
        toCTensorParams(
          kind, dataType: Element.dataType, format: dimensionFormat.format,
          shape: dimensionFormat.shape), initBound,
        trainable == true ? 1 : (trainable == false ? 0 : -1), name))
  }
}

extension Parameter: ModelIOConvertible {
  public var io: Model.IO {
    return apply([])
  }
}

/// Scalar model.
public final class Scalar: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init<Element: TensorNumeric>(
    _ kind: DeviceKind, format: TensorFormat, value: Float, of: Element.Type = Element.self,
    name: String = ""
  ) {
    super.init(ccv_cnnp_scalar(kind.toC, format.toC, Element.dataType.toC, value, name))
  }

  init(value: Float, name: String = "") {
    super.init(ccv_cnnp_scalar(0, 0, 0, value, name))
  }
}

/// Scaled-dot-product-attention model.
public final class ScaledDotProductAttention: Model {
  required init(_ model: OpaquePointer) {
    super.init(model)
  }

  public init(
    scale: Float, isCausal: Bool = false, hasAttentionMask: Bool = false, upcast: Bool = false,
    multiHeadOutputProjectionFused: Bool = false, noBias: Bool = false, trainable: Bool? = nil,
    name: String = ""
  ) {
    super.init(
      ccv_cnnp_scaled_dot_product_attention(
        scale, isCausal ? 1 : 0, hasAttentionMask ? 1 : 0, upcast ? 1 : 0,
        multiHeadOutputProjectionFused ? 1 : 0, noBias ? 1 : 0,
        trainable == true ? 1 : (trainable == false ? 0 : -1), name))
  }

  public func callAsFunction<T: DynamicGraph.TensorGroup>(
    queries q: T, keys k: T, values v: T, attentionMask: T? = nil,
    streamContext: StreamContext? = nil
  ) -> T {
    if let attentionMask = attentionMask {
      let outputs = self(inputs: q, k, v, attentionMask, streamContext: streamContext)
      return T(outputs[0])
    } else {
      let outputs = self(inputs: q, k, v, streamContext: streamContext)
      return T(outputs[0])
    }
  }
}

/// Custom model.
public final class CustomModel: Model {
  required init(_ model: OpaquePointer) {
    isa = ccv_nnc_cmd_vtab_t()
    super.init(model)
  }

  public enum IOType {
    case inputOrOutput
    case noTensor
    case tensorNotOutput
    case sharedTensor(ccv_cnnp_cmd_exec_io_init_state_t)
    case sharedTensorAsTrainable(ccv_cnnp_cmd_exec_io_init_state_t)
  }

  private var isa: ccv_nnc_cmd_vtab_t

  public init(
    inputs: [IOType], outputs: [IOType], hint: Hint, trainable: Bool? = nil, name: String = "",
    shapeInference: @convention(c) (
      _: ccv_nnc_cmd_t, _: UnsafePointer<ccv_nnc_tensor_param_t>?, _: Int32, _: ccv_nnc_hint_t,
      _: UnsafeMutablePointer<ccv_nnc_tensor_param_t>?, _: Int32
    ) -> Void,
    execute: @convention(c) (
      _: ccv_nnc_cmd_t, _: ccv_nnc_hint_t, _: Int32,
      _: UnsafePointer<UnsafeMutablePointer<ccv_nnc_tensor_t>?>?, _: Int32,
      _: UnsafePointer<UnsafeMutablePointer<ccv_nnc_tensor_t>?>?, _: Int32, _: OpaquePointer?
    ) -> Int32
  ) {
    isa = ccv_nnc_cmd_vtab_s()
    isa.tensor_auto = shapeInference
    isa.exec = execute
    let cmd = ccv_nnc_cmd(UInt32(CCV_NNC_CUSTOM_FORWARD), &isa, ccv_nnc_cmd_param_t(), 0)
    let cInputs: [ccv_cnnp_cmd_exec_io_t] = inputs.map {
      var io = ccv_cnnp_cmd_exec_io_t()
      switch $0 {
      case .inputOrOutput:
        io.type = Int32(CCV_CNNP_IO)
      case .noTensor:
        io.type = Int32(CCV_CNNP_NO_TENSOR)
      case .tensorNotOutput:
        io.type = Int32(CCV_CNNP_TENSOR_NOT_OUTPUT)
      case .sharedTensor(let init_state):
        io.type = Int32(CCV_CNNP_INIT_SHARED_TENSOR)
        io.init_state = init_state
      case .sharedTensorAsTrainable(let init_state):
        io.type = Int32(CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE)
        io.init_state = init_state
      }
      return io
    }
    let cOutputs: [Int32] = outputs.map {
      switch $0 {
      case .inputOrOutput:
        return Int32(CCV_CNNP_IO)
      case .noTensor:
        return Int32(CCV_CNNP_NO_TENSOR)
      case .tensorNotOutput:
        return Int32(CCV_CNNP_TENSOR_NOT_OUTPUT)
      case .sharedTensor(_):
        return Int32(CCV_CNNP_INIT_SHARED_TENSOR)
      case .sharedTensorAsTrainable(_):
        return Int32(CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE)
      }
    }
    super.init(
      ccv_cnnp_cmd_exec(
        cmd, hint.toCHint(), 0, cInputs, Int32(inputs.count), cOutputs, Int32(outputs.count),
        trainable == true ? 1 : (trainable == false ? 0 : -1), name))
  }
}
