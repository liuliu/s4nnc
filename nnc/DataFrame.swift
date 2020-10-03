import C_nnc

public protocol DataSeries: Sequence {
  func prefetch(_ i: Int, streamContext: StreamContext?)
  func next(_ streamContext: StreamContext?) -> Element?
}

public extension DataSeries {
  func prefetch(_ i: Int) {
    self.prefetch(i, streamContext: nil)
  }
  func next() -> Element? {
    return self.next(nil)
  }
}

public struct DataSeriesIterator<S: DataSeries>: IteratorProtocol {

  public typealias Element = S.Element

  private let dataSeries: S

  public init(_ dataSeries: S) {
    self.dataSeries = dataSeries
  }

  public func next() -> Element? {
    return dataSeries.next()
  }

}

public final class DataFrame {

  final class Wrapped<T> {
    let value: T
    init(_ value: T) {
      self.value = value
    }
  }

  struct ColumnProperty {
    enum PropertyType {
      case object
      case tensor
    }
    var index: Int
    var type: PropertyType
  }

  private let underlying: Wrapped<[AnyObject]>
  private let _dataframe: OpaquePointer
  private var columnProperties: [String: ColumnProperty]

  public init<S: Sequence>(from sequence: S, name: String = "main") {
    underlying = Wrapped(Array(sequence) as [AnyObject])
    var column_data = ccv_cnnp_column_data_t()
    column_data.data_enum = { _, row_idxs, row_size, data, context, _ in
      guard let row_idxs = row_idxs else { return }
      guard let data = data else { return }
      let underlying = Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context!).takeUnretainedValue()
      for i in 0..<Int(row_size) {
        let idx = Int((row_idxs + i).pointee)
        let value = underlying.value[idx]
        (data + i).initialize(to: Unmanaged.passRetained(value).toOpaque())
      }
    }
    column_data.data_deinit = { data, _ in
      guard let data = data else { return }
      Unmanaged<AnyObject>.fromOpaque(data).release()
    }
    column_data.context_deinit = { context in
      guard let context = context else { return }
      Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context).release()
    }
    column_data.context = Unmanaged.passRetained(underlying).toOpaque()
    _dataframe = ccv_cnnp_dataframe_new(&column_data, 1, Int32(underlying.value.count))!
    columnProperties = [name: ColumnProperty(index: 0, type: .object)]
  }

  public func shuffle() {
    ccv_cnnp_dataframe_shuffle(_dataframe)
  }

  public subscript(firstIndex: String, indices: String...) -> ManyUntypedSeries {
    get {
      var properties = [ColumnProperty]()
      properties.append(columnProperties[firstIndex]!)
      for index in indices {
        properties.append(columnProperties[index]!)
      }
      let i: [Int32] = properties.map { Int32($0.index) }
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, i, Int32(i.count))!
      let rowCount = ccv_cnnp_dataframe_row_count(_dataframe)
      return ManyUntypedSeries(iter, count: Int(rowCount), properties: properties)
    }
  }

  public subscript(index: String) -> UntypedSeries {
    get {
      let columnProperty = columnProperties[index]!
      var i: Int32 = Int32(columnProperty.index)
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, &i, 1)!
      let rowCount = ccv_cnnp_dataframe_row_count(_dataframe)
      return UntypedSeries(iter, count: Int(rowCount), name: index, property: columnProperty)
    }
    set (v) {
      switch (v.role) {
        case .opaque(_):
          columnProperties[index] = columnProperties[v.name!]!
        case .scalar(let scalar):
          self.add(from: scalar, name: index)
        case .sequence(let sequence):
          self.add(from: sequence, name: index)
        case .image(let property):
          self.add(toLoadImage: property, name: index)
        case .map(let property, let mapper, let outputType):
          self.add(map: mapper, property: property, outputType: outputType, name: index)
        case .multimap(let properties, let mapper, let outputType):
          self.add(multimap: mapper, properties: properties, outputType: outputType, name: index)
      }
    }
  }

  public subscript<Element>(index: String, type: Element.Type) -> TypedSeries<Element> {
    get {
      let columnProperty = columnProperties[index]!
      var i: Int32 = Int32(columnProperty.index)
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, &i, 1)!
      let rowCount = ccv_cnnp_dataframe_row_count(_dataframe)
      return TypedSeries(iter, count: Int(rowCount), property: columnProperty)
    }
  }

  public var count: Int {
    return Int(ccv_cnnp_dataframe_row_count(_dataframe))
  }

  deinit {
    ccv_cnnp_dataframe_free(_dataframe)
  }
}

fileprivate enum UntypedSeriesRole {
  // getter
  case opaque(OpaquePointer)
  // setter
  case scalar(AnyObject)
  case sequence(DataFrame.Wrapped<[AnyObject]>)
  case image(DataFrame.ColumnProperty)
  case map(DataFrame.ColumnProperty, (AnyObject) -> AnyObject, DataFrame.ColumnProperty.PropertyType)
  case multimap([DataFrame.ColumnProperty], ([AnyObject]) -> AnyObject, DataFrame.ColumnProperty.PropertyType)
}

public extension DataFrame {

  final class UntypedSeries: DataSeries {

    public typealias Element = AnyObject

    public func makeIterator() -> DataSeriesIterator<UntypedSeries> {
      switch role {
        case .opaque(let iter):
          ccv_cnnp_dataframe_iter_set_cursor(iter, 0)
        default:
          fatalError()
      }
      return DataSeriesIterator(self)
    }

    public func prefetch(_ i: Int, streamContext: StreamContext?) {
      switch role {
        case .opaque(let iter):
          ccv_cnnp_dataframe_iter_prefetch(iter, Int32(i), streamContext?._stream)
        default:
          fatalError()
      }
    }

    public func next(_ streamContext: StreamContext?) -> AnyObject? {
      switch role {
        case .opaque(let iter):
          var data: UnsafeMutableRawPointer? = nil
          let retval = ccv_cnnp_dataframe_iter_next(iter, &data, 1, streamContext?._stream)
          guard retval == 0 else { return nil }
          if data == nil {
            return nil
          }
          switch property!.type {
          case .object:
            return Unmanaged<AnyObject>.fromOpaque(data!).takeUnretainedValue()
          case .tensor:
            return _AnyTensor(data!.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false).toAnyTensor() as AnyObject
          }
        default:
          fatalError()
      }
    }

    public var underestmiatedCount: Int {
      return count
    }

    public let count: Int

    fileprivate let role: UntypedSeriesRole
    fileprivate let name: String?
    fileprivate let property: ColumnProperty?

    fileprivate init(_ iter: OpaquePointer, count: Int, name: String, property: ColumnProperty) {
      role = .opaque(iter)
      self.count = count
      self.name = name
      self.property = property
    }

    fileprivate init( _ role: UntypedSeriesRole) {
      self.role = role
      count = 0
      name = nil
      property = nil
    }

    deinit {
      if case .opaque(let iter) = role {
        ccv_cnnp_dataframe_iter_free(iter)
      }
    }
  }
}

public extension DataFrame {
  final class ManyUntypedSeries: DataSeries {

    public typealias Element = [AnyObject]

    public func makeIterator() -> DataSeriesIterator<ManyUntypedSeries> {
      ccv_cnnp_dataframe_iter_set_cursor(_iter, 0)
      return DataSeriesIterator(self)
    }

    public func prefetch(_ i: Int, streamContext: StreamContext?) {
      ccv_cnnp_dataframe_iter_prefetch(_iter, Int32(i), streamContext?._stream)
    }

    public func next(_ streamContext: StreamContext?) -> [AnyObject]? {
      let count = properties.count
      let data = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: count)
      let retval = ccv_cnnp_dataframe_iter_next(_iter, data, Int32(count), streamContext?._stream)
      guard retval == 0 else { return nil }
      var columnData = [AnyObject]()
      for i in 0..<count {
        let data = data[i]!
        let object: AnyObject
        switch properties[i].type {
        case .object:
          object = Unmanaged<AnyObject>.fromOpaque(data).takeUnretainedValue()
        case .tensor:
          object = _AnyTensor(data.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false).toAnyTensor() as AnyObject
        }
        columnData.append(object)
      }
      data.deallocate()
      return columnData
    }

    public var underestmiatedCount: Int {
      return count
    }

    public let count: Int

    private let _iter: OpaquePointer
    private let properties: [ColumnProperty]

    fileprivate init(_ iter: OpaquePointer, count: Int, properties: [ColumnProperty]) {
      _iter = iter
      self.properties = properties
      self.count = count
    }

    deinit {
      ccv_cnnp_dataframe_iter_free(_iter)
    }
  }
}

public extension DataFrame {
  final class TypedSeries<Element>: DataSeries {

    public typealias Element = Element

    public func makeIterator() -> DataSeriesIterator<TypedSeries> {
      ccv_cnnp_dataframe_iter_set_cursor(_iter, 0)
      return DataSeriesIterator(self)
    }

    public func prefetch(_ i: Int, streamContext: StreamContext?) {
      ccv_cnnp_dataframe_iter_prefetch(_iter, Int32(i), streamContext?._stream)
    }

    public func next(_ streamContext: StreamContext?) -> Element? {
      var data: UnsafeMutableRawPointer? = nil
      let retval = ccv_cnnp_dataframe_iter_next(_iter, &data, 1, streamContext?._stream)
      guard retval == 0 else { return nil }
      if data == nil {
        return nil
      }
      switch property.type {
      case .object:
        return Unmanaged<AnyObject>.fromOpaque(data!).takeUnretainedValue() as? Element
      case .tensor:
        return _AnyTensor(data!.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false).toTensor(Element.self)
      }
    }

    public var underestmiatedCount: Int {
      return count
    }

    public let count: Int

    private let _iter: OpaquePointer
    private let property: ColumnProperty

    fileprivate init(_ iter: OpaquePointer, count: Int, property: ColumnProperty) {
      _iter = iter
      self.count = count
      self.property = property
    }

    deinit {
      ccv_cnnp_dataframe_iter_free(_iter)
    }
  }
}

// MARK - Scalar support

public extension DataFrame.UntypedSeries {
  static func from(_ scalar: Any) -> DataFrame.UntypedSeries {
    return DataFrame.UntypedSeries(.scalar(scalar as AnyObject)) // Wrap this.
  }
}

private extension DataFrame {
  private func add(from scalar: AnyObject, name: String) {
    let index = ccv_cnnp_dataframe_add(_dataframe, {  _, row_idxs, row_size, data, context, _ in
      guard let data = data else { return }
      for i in 0..<Int(row_size) {
        (data + i).initialize(to: context)
      }
    }, 0, nil, Unmanaged.passRetained(scalar).toOpaque(), { context in
      guard let context = context else { return }
      Unmanaged<AnyObject>.fromOpaque(context).release()
    })
    columnProperties[name] = ColumnProperty(index: Int(index), type: .object)
  }
}

// MARK - Sequence support

public extension DataFrame.UntypedSeries {
  static func from<S: Sequence>(_ sequence: S) -> DataFrame.UntypedSeries {
    return DataFrame.UntypedSeries(.sequence(DataFrame.Wrapped(Array(sequence) as [AnyObject])))
  }
}

private extension DataFrame {
  private func add(from sequence: Wrapped<[AnyObject]>, name: String) {
    assert(sequence.value.count == count)
    let index = ccv_cnnp_dataframe_add(_dataframe, {  _, row_idxs, row_size, data, context, _ in
      guard let row_idxs = row_idxs else { return }
      guard let data = data else { return }
      let underlying = Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context!).takeUnretainedValue()
      for i in 0..<Int(row_size) {
        let idx = Int((row_idxs + i).pointee)
        let value = underlying.value[idx]
        (data + i).initialize(to: Unmanaged.passRetained(value).toOpaque())
      }
    }, 0, { data, _ in
      guard let data = data else { return }
      Unmanaged<AnyObject>.fromOpaque(data).release()
    }, Unmanaged.passRetained(sequence).toOpaque(), { context in
      guard let context = context else { return }
      Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context).release()
    })
    columnProperties[name] = ColumnProperty(index: Int(index), type: .object)
  }
}

// MARK - Load image.

public extension DataFrame.UntypedSeries {
  func toLoadImage() -> DataFrame.UntypedSeries {
    guard let property = property else {
      fatalError("Can only load from series from DataFrame")
    }
    return DataFrame.UntypedSeries(.image(property))
  }
}

public extension DataFrame.TypedSeries where Element == String {
  func toLoadImage() -> DataFrame.UntypedSeries {
    return DataFrame.UntypedSeries(.image(property))
  }
}

private extension DataFrame {
  private func add(toLoadImage property: ColumnProperty, name: String) {
    var inputIndex = Int32(property.index)
    let pathIndex = ccv_cnnp_dataframe_map(_dataframe, { input, _, row_size, data, context, _ in
      guard let input = input else { return }
      guard let data = data else { return }
      let inputData = input[0]!
      for i in 0..<Int(row_size) {
        var path = Unmanaged<AnyObject>.fromOpaque(inputData[i]!).takeUnretainedValue() as! String
        let utf8: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>> = path.withUTF8 {
          let string = UnsafeMutablePointer<UInt8>.allocate(capacity: $0.count + 1)
          string.initialize(from: $0.baseAddress!, count: $0.count)
          // null-terminated
          (string + $0.count).initialize(to: 0)
          let container = UnsafeMutablePointer<UnsafeMutablePointer<UInt8>>.allocate(capacity: 1)
          container.initialize(to: string)
          return container
        }
        (data + i).initialize(to: utf8)
      }
    }, 0, { container, _ in
      guard let container = container else { return }
      let string = container.assumingMemoryBound(to: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>>.self)[0]
      string.deallocate()
      container.deallocate()
    }, &inputIndex, 1, nil, nil)
    let index = ccv_cnnp_dataframe_read_image(_dataframe, pathIndex, 0)
    columnProperties[name] = ColumnProperty(index: Int(index), type: .tensor)
  }
}

public extension DataFrame.UntypedSeries {
  func map<T, U>(_ mapper: @escaping (T) -> U) -> DataFrame.UntypedSeries {
    let wrappedMapper = { (obj: AnyObject) -> AnyObject in
      return mapper(obj as! T) as AnyObject
    }
    // Use only return type for whether this is a tensor or not.
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      // Special handling if this is a tensor, for C-interop, we will unwrap the underlying tensor.
      return DataFrame.UntypedSeries(.map(property!, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.map(property!, wrappedMapper, .object))
    }
  }
}

public extension DataFrame.TypedSeries {
  func map<U>(_ mapper: @escaping (Element) -> U) -> DataFrame.UntypedSeries {
    let wrappedMapper = { (obj: AnyObject) -> AnyObject in
      return mapper(obj as! Element) as AnyObject
    }
    // Use only return type for whether this is a tensor or not.
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      // Special handling if this is a tensor, for C-interop, we will unwrap the underlying tensor.
      return DataFrame.UntypedSeries(.map(property, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.map(property, wrappedMapper, .object))
    }
  }
}

private extension DataFrame {
  private final class WrappedMapper {
    let property: ColumnProperty
    let map: (AnyObject) -> AnyObject
    let outputType: ColumnProperty.PropertyType
    var tensors: [OpaquePointer: _AnyTensor]?
    init(property: ColumnProperty, map: @escaping (AnyObject) -> AnyObject, outputType: ColumnProperty.PropertyType) {
      self.property = property
      self.map = map
      self.outputType = outputType
      switch outputType {
      case .object:
        tensors = nil
      case .tensor:
        tensors = [OpaquePointer: _AnyTensor]()
      }
    }
  }
  private func add(map: @escaping (AnyObject) -> AnyObject, property: ColumnProperty, outputType: ColumnProperty.PropertyType, name: String) {
    var inputIndex = Int32(property.index)
    let index = ccv_cnnp_dataframe_map(_dataframe, { input, _, row_size, data, context, _ in
      guard let input = input else { return }
      guard let data = data else { return }
      let inputData = input[0]!
      let wrappedMapper = Unmanaged<WrappedMapper>.fromOpaque(context!).takeUnretainedValue()
      for i in 0..<Int(row_size) {
        let object: AnyObject
        switch wrappedMapper.property.type {
        case .object:
          object = Unmanaged<AnyObject>.fromOpaque(inputData[i]!).takeUnretainedValue()
        case .tensor:
          object = _AnyTensor(inputData[i]!.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false).toAnyTensor() as AnyObject
        }
        let output = wrappedMapper.map(object)
        switch wrappedMapper.outputType {
        case .object:
          (data + i).initialize(to: Unmanaged.passRetained(output).toOpaque())
        case .tensor:
          let tensor = output as! AnyTensor
          wrappedMapper.tensors![OpaquePointer(tensor.underlying._tensor)] = tensor.underlying
          (data + i).initialize(to: tensor.underlying._tensor)
        }
      }
    }, 0, { object, context in
      guard let object = object else { return }
      let wrappedMapper = Unmanaged<WrappedMapper>.fromOpaque(context!).takeUnretainedValue()
      switch wrappedMapper.outputType {
      case .object:
        Unmanaged<AnyObject>.fromOpaque(object).release()
      case .tensor:
        wrappedMapper.tensors![OpaquePointer(object)] = nil
      }
    }, &inputIndex, 1, Unmanaged.passRetained(WrappedMapper(property: property, map: map, outputType: outputType)).toOpaque(), { mapper in
      Unmanaged<WrappedMapper>.fromOpaque(mapper!).release()
    })
    columnProperties[name] = ColumnProperty(index: Int(index), type: outputType)
  }
}

public extension DataFrame.ManyUntypedSeries {
  func map<C0, C1, U>(_ mapper: @escaping (C0, C1) -> U) -> DataFrame.UntypedSeries {
    assert(properties.count == 2)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(objs[0] as! C0, objs[1] as! C1) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  func map<C0, C1, C2, U>(_ mapper: @escaping (C0, C1, C2) -> U) -> DataFrame.UntypedSeries {
    assert(properties.count == 3)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(objs[0] as! C0, objs[1] as! C1, objs[2] as! C2) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  func map<C0, C1, C2, C3, U>(_ mapper: @escaping (C0, C1, C2, C3) -> U) -> DataFrame.UntypedSeries {
    assert(properties.count == 4)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  func map<C0, C1, C2, C3, C4, U>(_ mapper: @escaping (C0, C1, C2, C3, C4) -> U) -> DataFrame.UntypedSeries {
    assert(properties.count == 5)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  func map<C0, C1, C2, C3, C4, C5, U>(_ mapper: @escaping (C0, C1, C2, C3, C4, C5) -> U) -> DataFrame.UntypedSeries {
    assert(properties.count == 6)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4, objs[5] as! C5) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  func map<C0, C1, C2, C3, C4, C5, C6, U>(_ mapper: @escaping (C0, C1, C2, C3, C4, C5, C6) -> U) -> DataFrame.UntypedSeries {
    assert(properties.count == 7)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4, objs[5] as! C5, objs[6] as! C6) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  func map<C0, C1, C2, C3, C4, C5, C6, C7, U>(_ mapper: @escaping (C0, C1, C2, C3, C4, C5, C6, C7) -> U) -> DataFrame.UntypedSeries {
    assert(properties.count == 8)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4, objs[5] as! C5, objs[6] as! C6, objs[7] as! C7) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  func map<C0, C1, C2, C3, C4, C5, C6, C7, C8, U>(_ mapper: @escaping (C0, C1, C2, C3, C4, C5, C6, C7, C8) -> U) -> DataFrame.UntypedSeries {
    assert(properties.count == 9)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4, objs[5] as! C5, objs[6] as! C6, objs[7] as! C7, objs[8] as! C8) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  func map<C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, U>(_ mapper: @escaping (C0, C1, C2, C3, C4, C5, C6, C7, C8, C9) -> U) -> DataFrame.UntypedSeries {
    assert(properties.count == 10)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4, objs[5] as! C5, objs[6] as! C6, objs[7] as! C7, objs[8] as! C8, objs[9] as! C9) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
}

private extension DataFrame {
  private final class WrappedManyMapper {
    let properties: [ColumnProperty]
    let map: ([AnyObject]) -> AnyObject
    let outputType: ColumnProperty.PropertyType
    var tensors: [OpaquePointer: _AnyTensor]?
    init(properties: [ColumnProperty], map: @escaping ([AnyObject]) -> AnyObject, outputType: ColumnProperty.PropertyType) {
      self.properties = properties
      self.map = map
      self.outputType = outputType
      switch outputType {
      case .object:
        tensors = nil
      case .tensor:
        tensors = [OpaquePointer: _AnyTensor]()
      }
    }
  }
  private func add(multimap: @escaping ([AnyObject]) -> AnyObject, properties: [ColumnProperty], outputType: ColumnProperty.PropertyType, name: String) {
    let inputIndex = properties.map { Int32($0.index) }
    let index = ccv_cnnp_dataframe_map(_dataframe, { input, _, row_size, data, context, _ in
      guard let input = input else { return }
      guard let data = data else { return }
      let wrappedManyMapper = Unmanaged<WrappedManyMapper>.fromOpaque(context!).takeUnretainedValue()
      for i in 0..<Int(row_size) {
        var objects = [AnyObject]()
        for (j, property) in wrappedManyMapper.properties.enumerated() {
          let object: AnyObject
          switch property.type {
          case .object:
            object = Unmanaged<AnyObject>.fromOpaque(input[j]![i]!).takeUnretainedValue()
          case .tensor:
            object = _AnyTensor(input[j]![i]!.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false).toAnyTensor() as AnyObject
          }
          objects.append(object)
        }
        let output = wrappedManyMapper.map(objects)
        switch wrappedManyMapper.outputType {
        case .object:
          (data + i).initialize(to: Unmanaged.passRetained(output).toOpaque())
        case .tensor:
          let tensor = output as! AnyTensor
          wrappedManyMapper.tensors![OpaquePointer(tensor.underlying._tensor)] = tensor.underlying
          (data + i).initialize(to: tensor.underlying._tensor)
        }
      }
    }, 0, { object, context in
      guard let object = object else { return }
      let wrappedManyMapper = Unmanaged<WrappedManyMapper>.fromOpaque(context!).takeUnretainedValue()
      switch wrappedManyMapper.outputType {
      case .object:
        Unmanaged<AnyObject>.fromOpaque(object).release()
      case .tensor:
        wrappedManyMapper.tensors![OpaquePointer(object)] = nil
      }
    }, inputIndex, Int32(inputIndex.count), Unmanaged.passRetained(WrappedManyMapper(properties: properties, map: multimap, outputType: outputType)).toOpaque(), { mapper in
      Unmanaged<WrappedManyMapper>.fromOpaque(mapper!).release()
    })
    columnProperties[name] = ColumnProperty(index: Int(index), type: outputType)
  }
}
