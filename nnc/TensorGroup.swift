/// Represent a type-checked parameters that encapsulate both tensor and group of tensors.
public protocol DynamicGraph_TensorGroup: DynamicGraph_AnyTensorGroup {
  associatedtype ElementNumeric: TensorNumeric
  init(_: AnyTensor)
  var grad: AnyTensor? { get set }
  var typeErased: AnyTensor { get }
  var isNaN: Bool { get }
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
  /// Make a copy of the given tensor.
  func copied(streamContext: StreamContext?) -> Self
  /// Only ake a copy of the given tensor if it is not contiguous in memory.
  func contiguous(streamContext: StreamContext?) -> Self
  /// Reduce along a given dimension.
  func reduced(_ op: ReduceOp, axis: [Int], streamContext: StreamContext?) -> Self
  /// Scale the given tensor with a constant inplace.
  func scale(by a: Float, streamContext: StreamContext?)
  /// Scale the given tensor with a constant.
  func scaled(by a: Float, streamContext: StreamContext?) -> Self
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
  /// Chunk the current tensor into multiple ones.
  func chunked(_ numberOfChunks: Int, axis: Int, streamContext: StreamContext?) -> [Self]
}

extension DynamicGraph_TensorGroup {
  @inlinable
  public subscript(ranges: Range<Int>...) -> Self
  {
    get { self[ranges, streamContext: nil] }
    set { self[ranges, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(ranges: [Range<Int>]) -> Self
  {
    get { self[ranges, streamContext: nil] }
    set { self[ranges, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(range: Range<Int>) -> Self {
    get { self[range, streamContext: nil] }
    set { self[range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, range: Range<Int>)
    -> Self
  {
    get { self[i0, range, streamContext: nil] }
    set { self[i0, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, range: Range<Int>
  ) -> Self {
    get { self[i0, i1, range, streamContext: nil] }
    set { self[i0, i1, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, range: Range<Int>
  ) -> Self {
    get { self[i0, i1, i2, range, streamContext: nil] }
    set { self[i0, i1, i2, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: Range<Int>
  ) -> Self {
    get { self[i0, i1, i2, i3, range, streamContext: nil] }
    set { self[i0, i1, i2, i3, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: Range<Int>
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, range, streamContext: nil] }
    set { self[i0, i1, i2, i3, i4, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: Range<Int>
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, i5, range, streamContext: nil] }
    set { self[i0, i1, i2, i3, i4, i5, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, range: Range<Int>
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, i5, i6, range, streamContext: nil] }
    set { self[i0, i1, i2, i3, i4, i5, i6, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(range: UnboundedRange) -> Self
  {
    get { self[range, streamContext: nil] }
    set { self[range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, range: UnboundedRange)
    -> Self
  {
    get { self[i0, range, streamContext: nil] }
    set { self[i0, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, range: UnboundedRange
  ) -> Self {
    get { self[i0, i1, range, streamContext: nil] }
    set { self[i0, i1, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, range: UnboundedRange
  ) -> Self {
    get { self[i0, i1, i2, range, streamContext: nil] }
    set { self[i0, i1, i2, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: UnboundedRange
  ) -> Self {
    get { self[i0, i1, i2, i3, range, streamContext: nil] }
    set { self[i0, i1, i2, i3, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: UnboundedRange
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, range, streamContext: nil] }
    set { self[i0, i1, i2, i3, i4, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: UnboundedRange
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, i5, range, streamContext: nil] }
    set { self[i0, i1, i2, i3, i4, i5, range, streamContext: nil] = newValue }
  }
  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int,
    range: UnboundedRange
  ) -> Self {
    get { self[i0, i1, i2, i3, i4, i5, i6, range, streamContext: nil] }
    set { self[i0, i1, i2, i3, i4, i5, i6, range, streamContext: nil] = newValue }
  }
  /// Transpose from axisA to axisB.
  @inlinable
  public func transposed(_ axisA: Int, _ axisB: Int) -> Self {
    transposed(axisA, axisB, streamContext: nil)
  }
  /// Fill the given tensor with uniform random values.
  @inlinable
  public func rand(_ range: ClosedRange<Float> = 0...1) {
    rand(range, streamContext: nil)
  }
  /// Fill the given tensor with normal-distributed random values.
  @inlinable
  public func randn(std: Float = 1, mean: Float = 0) {
    randn(std: std, mean: mean, streamContext: nil)
  }
  /// Copy the given tensor to GPU.
  @inlinable
  public func toGPU(_ ordinal: Int = 0) -> Self {
    toGPU(ordinal, streamContext: nil)
  }
  /// Copy the given tensor to CPU.
  @inlinable
  public func toCPU() -> Self {
    toCPU(streamContext: nil)
  }
  /// Fill the given tensor with a value.
  @inlinable
  public func full(_ value: Float = 0) {
    full(value, streamContext: nil)
  }
  /// Interpolate from this tensor to the other tensor.
  @inlinable
  public func lerp(_ weight: Float, to: Self) {
    lerp(weight, to: to, streamContext: nil)
  }
  /// Clamp the given tensor between two values.
  @inlinable
  public func clamp(_ range: ClosedRange<Float>) {
    clamp(range, streamContext: nil)
  }
  /// Clamp the given tensor with a lower bound.
  @inlinable
  public func clamp(_ range: PartialRangeFrom<Float>) {
    clamp(range, streamContext: nil)
  }
  /// Clamp the given tensor with an upper bound.
  @inlinable
  public func clamp(_ range: PartialRangeThrough<Float>) {
    clamp(range, streamContext: nil)
  }
  /// Clamp the given tensor between two values.
  @inlinable
  public func clamped(_ range: ClosedRange<Float>) -> Self {
    clamped(range, streamContext: nil)
  }
  /// Clamp the given tensor with a lower bound.
  @inlinable
  public func clamped(_ range: PartialRangeFrom<Float>) -> Self {
    clamped(range, streamContext: nil)
  }
  /// Clamp the given tensor with an upper bound.
  @inlinable
  public func clamped(_ range: PartialRangeThrough<Float>)
    -> Self
  {
    clamped(range, streamContext: nil)
  }
  /// Make a copy of the given tensor.
  @inlinable
  public func copied() -> Self {
    copied(streamContext: nil)
  }
  /// Only make a copy of the given tensor if it is not contiguous in memory.
  @inlinable
  public func contiguous() -> Self {
    contiguous(streamContext: nil)
  }
  /// Reduce along a given dimension.
  @inlinable
  public func reduced(_ op: ReduceOp, axis: [Int]) -> Self {
    reduced(op, axis: axis, streamContext: nil)
  }
  /// Scale the given tensor with a constant inplace.
  @inlinable
  public func scale(by a: Float) {
    scale(by: a, streamContext: nil)
  }
  /// Scale the given tensor with a constant.
  @inlinable
  public func scaled(by a: Float) -> Self {
    scaled(by: a, streamContext: nil)
  }
  /// Apply softmax activation to the given tensor inplace.
  @inlinable
  public func softmax() {
    softmax(streamContext: nil)
  }
  /// Apply ReLU activation to the given tensor inplace.
  @inlinable
  public func ReLU() {
    ReLU(streamContext: nil)
  }
  /// Apply sigmoid activation to the given tensor inplace.
  @inlinable
  public func sigmoid() {
    sigmoid(streamContext: nil)
  }
  /// Apply tanh activation to the given tensor inplace.
  @inlinable
  public func tanh() {
    tanh(streamContext: nil)
  }
  /// Apply swish activation to the given tensor inplace.
  @inlinable
  public func swish() {
    swish(streamContext: nil)
  }
  /// Chunk the current tensor into multiple ones.
  @inlinable
  public func chunked(_ numberOfChunks: Int, axis: Int = 0)
    -> [Self]
  {
    chunked(numberOfChunks, axis: axis, streamContext: nil)
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
