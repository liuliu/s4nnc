/// Empty protocol for other places to recognize AnyTensor and AnyGroup with dynamic dispatch.
public protocol DynamicGraph_AnyParameters {
}

/// Protocol for other places to recognize AnyTensor and AnyGroup with static dispatch.
public protocol DynamicGraph_Any: DynamicGraph_AnyParameters {
  var graph: DynamicGraph { get }
  var untyped: [DynamicGraph.AnyTensor] { get }
  var dimensions: [Int] { get }
  var format: TensorFormat { get }
  var increments: [Int] { get }
  var isConstant: Bool { get }
  var requiresGrad: Bool { get set }
}

/// Protocol for group of tensors.
public protocol DynamicGraph_AnyGroup: DynamicGraph_Any {
}

extension Model.Parameters: DynamicGraph_AnyParameters {
}

extension DynamicGraph {

  public typealias AnyGroup = DynamicGraph_AnyGroup

  /// Type-aware group of tensors.
  public struct Group<Element: DynamicGraph.AnyTensor>: RandomAccessCollection {
    var underlyingArray: [Element]

    public typealias Element = Element
    public typealias Index = Int
    public typealias Indices = Range<Index>
    public typealias SubSequence = Array<Element>.SubSequence
    public var endIndex: Index { underlyingArray.endIndex }
    public var indices: Indices { underlyingArray.indices }
    public var startIndex: Index { underlyingArray.startIndex }
    public func formIndex(after i: inout Index) { underlyingArray.formIndex(after: &i) }
    public func formIndex(before i: inout Index) { underlyingArray.formIndex(before: &i) }
    public subscript(position: Index) -> Element { underlyingArray[position] }
    public subscript(x: Indices) -> SubSequence { underlyingArray[x] }

    public init(_ elements: Element...) {
      underlyingArray = elements
    }

    public init<OtherElement: DynamicGraph.AnyTensor>(_ otherGroup: Group<OtherElement>) {
      if let upcastUnderlyingArray = otherGroup.underlyingArray as? [Element] {
        underlyingArray = upcastUnderlyingArray
      } else {
        let otherUnderlyingArray = otherGroup.underlyingArray
        var underlyingArray = [Element]()
        underlyingArray.reserveCapacity(otherUnderlyingArray.count)
        for element in otherUnderlyingArray {
          underlyingArray.append(Element(element))
        }
        self.underlyingArray = underlyingArray
      }
    }

    public init(_ underlyingArray: [Element]) {
      self.underlyingArray = underlyingArray
    }
  }

}

extension DynamicGraph.Group: DynamicGraph.AnyGroup {
  public var untyped: [DynamicGraph.AnyTensor] { underlyingArray as [DynamicGraph.AnyTensor] }
  public var graph: DynamicGraph { underlyingArray[0].graph }
  public var dimensions: [Int] {
    let dimensions = underlyingArray[0].dimensions
    for tensor in underlyingArray {
      assert(dimensions == tensor.dimensions)
    }
    return dimensions
  }
  public var format: TensorFormat {
    let format = underlyingArray[0].format
    for tensor in underlyingArray {
      assert(format == tensor.format)
    }
    return format
  }
  public var increments: [Int] {
    let increments = underlyingArray[0].increments
    for tensor in underlyingArray {
      assert(increments == tensor.increments)
    }
    return increments
  }
  public var isConstant: Bool {
    let isConstant = underlyingArray[0].isConstant
    for tensor in underlyingArray {
      assert(isConstant == tensor.isConstant)
    }
    return isConstant
  }
  public var requiresGrad: Bool {
    get {
      let requiresGrad = underlyingArray[0].requiresGrad
      for tensor in underlyingArray {
        assert(requiresGrad == tensor.requiresGrad)
      }
      return requiresGrad
    }
    set(v) {
      for tensor in underlyingArray {
        tensor.requiresGrad = v
      }
    }
  }
}

public typealias Group = DynamicGraph.Group

extension DynamicGraph.Group where Element: DynamicGraph.AnyTensor {
  public func reshaped(
    format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil
  ) -> Self {
    return Group(
      underlyingArray.map {
        $0.reshaped(format: format, dimensions: dimensions, offset: offset, increments: increments)
      })
  }
  public func reshaped(
    _ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil
  ) -> Self {
    return reshaped(
      format: dimensionFormat.format, dimensions: dimensionFormat.dimensions, offset: offset,
      increments: increments)
  }
}
