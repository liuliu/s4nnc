#if canImport(C_nnc)
import C_nnc
#elseif canImport(C_swiftpm_nnc)
import C_swiftpm_nnc
#endif

/// Hint are parameters to these operations that changes shape from input
/// to output. It given proper stride / padding parameters for these operations.
public struct Hint {
  public var stride: [Int]

  public struct Border {
    public var begin: [Int]
    public var end: [Int]

    public init() {
      begin = []
      end = []
    }

    public init(_ border: [Int]) {
      begin = border
      end = border
    }

    public init(begin: [Int], end: [Int]) {
      self.begin = begin
      self.end = end
    }
  }

  public var border: Border

  public init() {
    stride = []
    border = Border()
  }

  public init(stride: [Int], border: Border = Border()) {
    self.stride = stride
    self.border = border
  }
}

extension Hint {
  func toCHint() -> ccv_nnc_hint_t {
    var hint = ccv_nnc_hint_t()
    if stride.count > 0 {
      hint.stride.dim = toCDimensions(stride)
    }
    if border.begin.count > 0 {
      hint.border.begin = toCDimensions(border.begin)
    }
    if border.end.count > 0 {
      hint.border.end = toCDimensions(border.end)
    }
    return hint
  }
}
