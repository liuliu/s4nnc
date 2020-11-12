// Empty protocol for other places to recognize AnyTensor and AnyGroup with dynamic dispatch.
public protocol DynamicGraph_Any {
    var dimensions: [Int] { get }
}
public protocol DynamicGraph_AnyGroup: DynamicGraph_Any {
  var underlying: [DynamicGraph.AnyTensor] { get }
}

extension DynamicGraph_AnyGroup {
  public var dimensions: [Int] {
    let dimensions = underlying[0].dimensions
    for tensor in underlying {
      assert(dimensions == tensor.dimensions)
    }
    return dimensions
  }
}

extension DynamicGraph.AnyTensor: DynamicGraph_Any {
}

extension DynamicGraph {

  public typealias AnyGroup = DynamicGraph_AnyGroup

  public struct Group<Element: DynamicGraph.AnyTensor>: RandomAccessCollection, DynamicGraph.AnyGroup {
    public var underlying: [DynamicGraph.AnyTensor] { underlyingArray as [DynamicGraph.AnyTensor] }
    var underlyingArray: [Element]

    public typealias Element = Element
    public typealias Index = Int
    public typealias Indices = Range<Index>
    public typealias SubSequence = Array<Element>.SubSequence
    public var endIndex: Index{ underlyingArray.endIndex }
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

public typealias Group = DynamicGraph.Group

public extension DynamicGraph.Group where Element: DynamicGraph.Tensor<Float64> {
  func reshape(format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return Group(underlyingArray.map { $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments) })
  }
  func reshape(_ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return reshape(format: dimensionFormat.format, dimensions: dimensionFormat.dimensions, offset: offset, increments: increments)
  }
}

public extension DynamicGraph.Group where Element: DynamicGraph.Tensor<Int64> {
  func reshape(format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return Group(underlyingArray.map { $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments) })
  }
  func reshape(_ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return reshape(format: dimensionFormat.format, dimensions: dimensionFormat.dimensions, offset: offset, increments: increments)
  }
}

public extension DynamicGraph.Group where Element: DynamicGraph.Tensor<Float32> {
  func reshape(format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return Group(underlyingArray.map { $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments) })
  }
  func reshape(_ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return reshape(format: dimensionFormat.format, dimensions: dimensionFormat.dimensions, offset: offset, increments: increments)
  }
}

public extension DynamicGraph.Group where Element: DynamicGraph.Tensor<Int32> {
  func reshape(format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return Group(underlyingArray.map { $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments) })
  }
  func reshape(_ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return reshape(format: dimensionFormat.format, dimensions: dimensionFormat.dimensions, offset: offset, increments: increments)
  }
}

public extension DynamicGraph.Group where Element: DynamicGraph.Tensor<Float16> {
  func reshape(format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return Group(underlyingArray.map { $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments) })
  }
  func reshape(_ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return reshape(format: dimensionFormat.format, dimensions: dimensionFormat.dimensions, offset: offset, increments: increments)
  }
}

public extension DynamicGraph.Group where Element: DynamicGraph.Tensor<UInt8> {
  func reshape(format: TensorFormat, dimensions: [Int], offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return Group(underlyingArray.map { $0.reshape(format: format, dimensions: dimensions, offset: offset, increments: increments) })
  }
  func reshape(_ dimensionFormat: TensorDimensionFormat, offset: [Int]? = nil, increments: [Int]? = nil) -> Self {
    return reshape(format: dimensionFormat.format, dimensions: dimensionFormat.dimensions, offset: offset, increments: increments)
  }
}
