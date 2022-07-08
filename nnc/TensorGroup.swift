/// Represent a type-checked parameters that encapsulate both tensor and group of tensors.
public protocol DynamicGraph_TensorGroup: DynamicGraph_AnyTensorGroup {
  associatedtype ElementNumeric: TensorNumeric
  init(_: AnyTensor)
  subscript(ranges: Range<Int>..., streamContext streamContext: StreamContext?) -> Self { get set }
  subscript(ranges: [Range<Int>], streamContext streamContext: StreamContext?) -> Self { get set }
  subscript(range: Range<Int>, streamContext streamContext: StreamContext?) -> Self { get set }
  subscript(i0: Int, range: Range<Int>, streamContext streamContext: StreamContext?) -> Self {
    get set
  }
  subscript(i0: Int, i1: Int, range: Range<Int>, streamContext streamContext: StreamContext?)
    -> Self
  { get set }
  subscript(
    i0: Int, i1: Int, i2: Int, range: Range<Int>, streamContext streamContext: StreamContext?
  ) -> Self { get set }
  subscript(
    i0: Int, i1: Int, i2: Int, i3: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  ) -> Self { get set }
  subscript(
    i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  ) -> Self { get set }
  subscript(
    i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  ) -> Self { get set }
  subscript(
    i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  ) -> Self { get set }
  subscript(range: UnboundedRange, streamContext streamContext: StreamContext?) -> Self { get set }
  subscript(i0: Int, range: UnboundedRange, streamContext streamContext: StreamContext?) -> Self {
    get set
  }
  subscript(i0: Int, i1: Int, range: UnboundedRange, streamContext streamContext: StreamContext?)
    -> Self
  { get set }
  subscript(
    i0: Int, i1: Int, i2: Int, range: UnboundedRange, streamContext streamContext: StreamContext?
  ) -> Self { get set }
  subscript(
    i0: Int, i1: Int, i2: Int, i3: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  ) -> Self { get set }
  subscript(
    i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  ) -> Self { get set }
  subscript(
    i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  ) -> Self { get set }
  subscript(
    i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  ) -> Self { get set }
  /// Transpose from axisA to axisB.
  func transposed(_ axisA: Int, _ axisB: Int, streamContext: StreamContext?) -> Self
  /// Fill the given tensor with uniform random values.
  func rand(_ range: ClosedRange<Float>, streamContext: StreamContext?)
  /// Fill the given tensor with normal-distributed random values.
  func randn(std: Float, mean: Float, streamContext: StreamContext?)
  /// Copy the given tensor to GPU.
  func toGPU(_ ordinal: Int, streamContext: StreamContext?) -> Self
  /// Copy the given tensor to CPU.
  func toCPU(streamContext: StreamContext?) -> Self
  /// Fill the given tensor with a value.
  func full(_ value: Float, streamContext: StreamContext?)
  /// Interpolate from this tensor to the other tensor.
  func lerp(_ weight: Float, to: Self, streamContext: StreamContext?)
  /// Clamp the given tensor between two values.
  func clamp(_ range: ClosedRange<Float>, streamContext: StreamContext?)
  /// Clamp the given tensor with a lower bound.
  func clamp(_ range: PartialRangeFrom<Float>, streamContext: StreamContext?)
  /// Clamp the given tensor with an upper bound.
  func clamp(_ range: PartialRangeThrough<Float>, streamContext: StreamContext?)
  /// Detach current tensor from the graph. Afterwards, it is always "isConstant" and cannot requiresGrad.
  mutating func detach()
  /// Clamp the given tensor between two values.
  func clamped(_ range: ClosedRange<Float>, streamContext: StreamContext?) -> Self
  /// Clamp the given tensor with a lower bound.
  func clamped(_ range: PartialRangeFrom<Float>, streamContext: StreamContext?) -> Self
  /// Clamp the given tensor with an upper bound.
  func clamped(_ range: PartialRangeThrough<Float>, streamContext: StreamContext?) -> Self
  /// Reduce along a given dimension.
  func reduced(_ op: ReduceOp, axis: [Int], streamContext: StreamContext?) -> Self
  /// Scale the given tensor with a constant inplace.
  func scale(by a: Float, streamContext: StreamContext?)
  /// Apply softmax activation to the given tensor inplace.
  func softmax(streamContext: StreamContext?)
  /// Apply ReLU activation to the given tensor inplace.
  func ReLU(streamContext: StreamContext?)
  /// Apply sigmoid activation to the given tensor inplace.
  func sigmoid(streamContext: StreamContext?)
  /// Apply tanh activation to the given tensor inplace.
  func tanh(streamContext: StreamContext?)
  /// Apply swish activation to the given tensor inplace.
  func swish(streamContext: StreamContext?)
}

extension DynamicGraph_TensorGroup {
  @inlinable
  public subscript(ranges: Range<Int>..., streamContext streamContext: StreamContext? = nil) -> Self
  {
    get { self[ranges, streamContext: streamContext] }
    set { self[ranges, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(ranges: [Range<Int>], streamContext streamContext: StreamContext? = nil) -> Self
  {
    get { self[ranges, streamContext: streamContext] }
    set { self[ranges, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(range: Range<Int>, streamContext streamContext: StreamContext? = nil) -> Self {
    get { self[range, streamContext: streamContext] }
    set { self[range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, range: Range<Int>, streamContext streamContext: StreamContext? = nil)
    -> Self
  {
    get { self[i0, range, streamContext: streamContext] }
    set { self[i0, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, range: Range<Int>,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, range, streamContext: streamContext] }
    set { self[i0, i1, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, range: Range<Int>,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, i2, range, streamContext: streamContext] }
    set { self[i0, i1, i2, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: Range<Int>,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, i2, i3, range, streamContext: streamContext] }
    set { self[i0, i1, i2, i3, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: Range<Int>,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, range, streamContext: streamContext] }
    set { self[i0, i1, i2, i3, i4, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: Range<Int>,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, i5, range, streamContext: streamContext] }
    set { self[i0, i1, i2, i3, i4, i5, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, range: Range<Int>,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, i5, i6, range, streamContext: streamContext] }
    set { self[i0, i1, i2, i3, i4, i5, i6, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(range: UnboundedRange, streamContext streamContext: StreamContext? = nil) -> Self
  {
    get { self[range, streamContext: streamContext] }
    set { self[range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, range: UnboundedRange, streamContext streamContext: StreamContext? = nil)
    -> Self
  {
    get { self[i0, range, streamContext: streamContext] }
    set { self[i0, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, range, streamContext: streamContext] }
    set { self[i0, i1, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, i2, range, streamContext: streamContext] }
    set { self[i0, i1, i2, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, i2, i3, range, streamContext: streamContext] }
    set { self[i0, i1, i2, i3, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, range, streamContext: streamContext] }
    set { self[i0, i1, i2, i3, i4, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, i5, range, streamContext: streamContext] }
    set { self[i0, i1, i2, i3, i4, i5, range, streamContext: streamContext] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int,
    range: UnboundedRange, streamContext streamContext: StreamContext? = nil
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, i5, i6, range, streamContext: streamContext] }
    set { self[i0, i1, i2, i3, i4, i5, i6, range, streamContext: streamContext] = newValue }
  }
  /// Transpose from axisA to axisB.
  @inlinable
  public func transposed(_ axisA: Int, _ axisB: Int, streamContext: StreamContext? = nil) -> Self {
    transposed(axisA, axisB, streamContext: streamContext)
  }
  /// Fill the given tensor with uniform random values.
  @inlinable
  public func rand(_ range: ClosedRange<Float> = 0...1, streamContext: StreamContext? = nil) {
    rand(range, streamContext: streamContext)
  }
  /// Fill the given tensor with normal-distributed random values.
  @inlinable
  public func randn(std: Float = 1, mean: Float = 0, streamContext: StreamContext? = nil) {
    randn(std: std, mean: mean, streamContext: streamContext)
  }
  /// Copy the given tensor to GPU.
  @inlinable
  public func toGPU(_ ordinal: Int = 0, streamContext: StreamContext? = nil) -> Self {
    toGPU(ordinal, streamContext: streamContext)
  }
  /// Copy the given tensor to CPU.
  @inlinable
  public func toCPU(streamContext: StreamContext? = nil) -> Self {
    toCPU(streamContext: streamContext)
  }
  /// Fill the given tensor with a value.
  @inlinable
  public func full(_ value: Float = 0, streamContext: StreamContext? = nil) {
    full(value, streamContext: streamContext)
  }
  /// Interpolate from this tensor to the other tensor.
  @inlinable
  public func lerp(_ weight: Float, to: Self, streamContext: StreamContext? = nil) {
    lerp(weight, to: to, streamContext: streamContext)
  }
  /// Clamp the given tensor between two values.
  @inlinable
  public func clamp(_ range: ClosedRange<Float>, streamContext: StreamContext? = nil) {
    clamp(range, streamContext: streamContext)
  }
  /// Clamp the given tensor with a lower bound.
  @inlinable
  public func clamp(_ range: PartialRangeFrom<Float>, streamContext: StreamContext? = nil) {
    clamp(range, streamContext: streamContext)
  }
  /// Clamp the given tensor with an upper bound.
  @inlinable
  public func clamp(_ range: PartialRangeThrough<Float>, streamContext: StreamContext? = nil) {
    clamp(range, streamContext: streamContext)
  }
  /// Clamp the given tensor between two values.
  @inlinable
  public func clamped(_ range: ClosedRange<Float>, streamContext: StreamContext? = nil) -> Self {
    clamped(range, streamContext: streamContext)
  }
  /// Clamp the given tensor with a lower bound.
  @inlinable
  public func clamped(_ range: PartialRangeFrom<Float>, streamContext: StreamContext? = nil) -> Self
  {
    clamped(range, streamContext: streamContext)
  }
  /// Clamp the given tensor with an upper bound.
  @inlinable
  public func clamped(_ range: PartialRangeThrough<Float>, streamContext: StreamContext? = nil)
    -> Self
  {
    clamped(range, streamContext: streamContext)
  }
  /// Reduce along a given dimension.
  @inlinable
  public func reduced(_ op: ReduceOp, axis: [Int], streamContext: StreamContext? = nil) -> Self {
    reduced(op, axis: axis, streamContext: streamContext)
  }
  /// Scale the given tensor with a constant inplace.
  @inlinable
  public func scale(by a: Float, streamContext: StreamContext? = nil) {
    scale(by: a, streamContext: streamContext)
  }
  /// Apply softmax activation to the given tensor inplace.
  @inlinable
  public func softmax(streamContext: StreamContext? = nil) {
    softmax(streamContext: streamContext)
  }
  /// Apply ReLU activation to the given tensor inplace.
  @inlinable
  public func ReLU(streamContext: StreamContext? = nil) {
    ReLU(streamContext: streamContext)
  }
  /// Apply sigmoid activation to the given tensor inplace.
  @inlinable
  public func sigmoid(streamContext: StreamContext? = nil) {
    sigmoid(streamContext: streamContext)
  }
  /// Apply tanh activation to the given tensor inplace.
  @inlinable
  public func tanh(streamContext: StreamContext? = nil) {
    tanh(streamContext: streamContext)
  }
  /// Apply swish activation to the given tensor inplace.
  @inlinable
  public func swish(streamContext: StreamContext? = nil) {
    swish(streamContext: streamContext)
  }
}

extension DynamicGraph {
  public typealias TensorGroup = DynamicGraph_TensorGroup
}

extension DynamicGraph.Tensor: DynamicGraph.TensorGroup {
  public typealias ElementNumeric = Element
}

extension DynamicGraph.Group: DynamicGraph.TensorGroup
where Element: DynamicGraph.TensorGroup, Element: DynamicGraph.AnyTensor {
  public typealias ElementNumeric = Element.ElementNumeric
}
