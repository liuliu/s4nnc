import C_nnc
import MuJoCo
import NNC

extension Tensor {
  /**
   * Initialize a tensor from MjArray. This doesn't copy the data over, rather, we simply keep the
   * original MjArray alive. That's also why we don't support any data conversion. If you want to
   * keep the resulting Tensor for later usage (rather than do the computation right now), you need
   * to make your own copies because MjArray **CANNOT** be immutable and can tied to underlying
   * MjData updates (through `MjModel.forward` or `MjModel.step`). */
  @inlinable
  public init(mjArray: MjArray<Element>) {
    // MjArray is one dimension. Treat this as a C dimension.
    self.init(
      .CPU, format: .NCHW, dimensions: [mjArray.count], unsafeMutablePointer: mjArray + 0,
      bindLifetimeOf: mjArray)
  }
}

extension MjArray where Element: TensorNumeric {
  @inlinable
  public subscript(bounds: Range<Int>) -> Tensor<Element> {
    get { Tensor(mjArray: self[bounds]) }
    set {
      newValue.withUnsafeBytes {
        precondition(
          MemoryLayout<Element>.size * (bounds.upperBound - bounds.lowerBound) == $0.count)
        guard let source = $0.baseAddress else { return }
        UnsafeMutableRawPointer(self + bounds.lowerBound).copyMemory(
          from: source, byteCount: $0.count)
      }
    }
  }
  @inlinable
  public subscript(bounds: ClosedRange<Int>) -> Tensor<Element> {
    get {
      return self[bounds.lowerBound..<(bounds.upperBound + 1)]
    }
    set {
      self[bounds.lowerBound..<(bounds.upperBound + 1)] = newValue
    }
  }
  @inlinable
  public subscript(bounds: PartialRangeUpTo<Int>) -> Tensor<Element> {
    get {
      return self[0..<bounds.upperBound]
    }
    set {
      self[0..<bounds.upperBound] = newValue
    }
  }
  @inlinable
  public subscript(bounds: PartialRangeThrough<Int>) -> Tensor<Element> {
    get {
      return self[0..<(bounds.upperBound + 1)]
    }
    set {
      self[0..<(bounds.upperBound + 1)] = newValue
    }
  }
  @inlinable
  public subscript(bounds: PartialRangeFrom<Int>) -> Tensor<Element> {
    get {
      return self[bounds.lowerBound..<count]
    }
    set {
      self[bounds.lowerBound..<count] = newValue
    }
  }
  @inlinable
  public subscript(x: (UnboundedRange_) -> Void) -> Tensor<Element> {
    get {
      return Tensor(mjArray: self)
    }
    set {
      newValue.withUnsafeBytes {
        precondition(MemoryLayout<Element>.size * count == $0.count)
        guard let source = $0.baseAddress else { return }
        UnsafeMutableRawPointer(self + 0).copyMemory(from: source, byteCount: $0.count)
      }
    }
  }
}
