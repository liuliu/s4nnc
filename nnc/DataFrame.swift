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
          return Unmanaged<AnyObject>.fromOpaque(data!).takeUnretainedValue()
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
        let object = Unmanaged<AnyObject>.fromOpaque(data[i]!).takeUnretainedValue()
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
      return Unmanaged<AnyObject>.fromOpaque(data!).takeUnretainedValue() as? Element
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
  func toLoadImage(_ name: String) -> DataFrame.UntypedSeries {
    guard let property = property else {
      fatalError("Can only load from series from DataFrame")
    }
    return DataFrame.UntypedSeries(.image(property))
  }
}

public extension DataFrame.TypedSeries where Element == String {
  func toLoadImage(_ name: String) -> DataFrame.UntypedSeries {
    return DataFrame.UntypedSeries(.image(property))
  }
}

private extension DataFrame {
  private func add(toLoadImage property: ColumnProperty, name: String) {
    var inputIndex: Int32 = Int32(property.index)
    let index = ccv_cnnp_dataframe_map(_dataframe, { input, _, row_size, data, context, _ in
      guard let input = input else { return }
      guard let data = data else { return }
      for i in 0..<Int(row_size) {
        // (data + i).initialize(to: Unmanaged.passRetained(value).toOpaque())
      }
    }, 0, { data, _ in
      guard let data = data else { return }
      Unmanaged<AnyObject>.fromOpaque(data).release()
    }, &inputIndex, 1, nil, nil)
    // columnProperties[name] = ColumnProperty(index: Int(index), type: .object)
  }
}

public extension DataFrame.UntypedSeries {
  func map(_ name: String) {
  }
}

public extension DataFrame.TypedSeries {
  func map(_ name: String) {
  }
}

public extension DataFrame.ManyUntypedSeries {
  func map(_ name: String) {
  }
}
