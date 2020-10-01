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
    var index: Int
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
    columnProperties = [name: ColumnProperty(index: 0)]
  }

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
    columnProperties[name] = ColumnProperty(index: Int(index))
  }

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
    columnProperties[name] = ColumnProperty(index: Int(index))
  }

  public subscript(firstIndex: String, indices: String...) -> ManyUntypedSeries {
    get {
      var i = [Int32]()
      i.append(Int32(columnProperties[firstIndex]!.index))
      for index in indices {
        i.append(Int32(columnProperties[index]!.index))
      }
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, i, Int32(i.count))!
      let rowCount = ccv_cnnp_dataframe_row_count(_dataframe)
      return ManyUntypedSeries(iter, columnCount: i.count, count: Int(rowCount))
    }
  }

  public subscript(index: String) -> UntypedSeries {
    get {
      let columnProperty = columnProperties[index]!
      var i: Int32 = Int32(columnProperty.index)
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, &i, 1)!
      let rowCount = ccv_cnnp_dataframe_row_count(_dataframe)
      return UntypedSeries(iter, count: Int(rowCount), name: index)
    }
    set (v) {
      switch (v.underlying) {
        case .opaque(_):
          columnProperties[index] = columnProperties[v.name]!
        case .scalar(let scalar):
          self.add(from: scalar, name: index)
        case .sequence(let sequence):
          self.add(from: sequence, name: index)
      }
    }
  }

  public subscript<Element>(index: String, type: Element.Type) -> TypedSeries<Element> {
    get {
      let columnProperty = columnProperties[index]!
      var i: Int32 = Int32(columnProperty.index)
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, &i, 1)!
      let rowCount = ccv_cnnp_dataframe_row_count(_dataframe)
      return TypedSeries(iter, count: Int(rowCount))
    }
  }

  public var count: Int {
    return Int(ccv_cnnp_dataframe_row_count(_dataframe))
  }

  deinit {
    ccv_cnnp_dataframe_free(_dataframe)
  }
}

public extension DataFrame {
  final class UntypedSeries: DataSeries {

    public typealias Element = AnyObject

    public func makeIterator() -> DataSeriesIterator<UntypedSeries> {
      switch underlying {
        case .opaque(let iter):
          ccv_cnnp_dataframe_iter_set_cursor(iter, 0)
        default:
          fatalError()
      }
      return DataSeriesIterator(self)
    }

    enum Underlying {
      case opaque(OpaquePointer)
      case scalar(AnyObject)
      case sequence(Wrapped<[AnyObject]>)
    }

    public static func from(_ scalar: Any) -> UntypedSeries {
      return UntypedSeries(scalar as AnyObject) // Wrap this.
    }

    public static func from<S: Sequence>(_ sequence: S) -> UntypedSeries {
      return UntypedSeries(Wrapped(Array(sequence) as [AnyObject]))
    }

    public func prefetch(_ i: Int, streamContext: StreamContext?) {
      switch underlying {
        case .opaque(let iter):
          ccv_cnnp_dataframe_iter_prefetch(iter, Int32(i), streamContext?._stream)
        default:
          fatalError()
      }
    }

    public func next(_ streamContext: StreamContext?) -> AnyObject? {
      switch underlying {
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

    fileprivate let underlying: Underlying
    fileprivate let name: String

    fileprivate init(_ iter: OpaquePointer, count: Int, name: String) {
      underlying = .opaque(iter)
      self.count = count
      self.name = name
    }

    private init( _ scalar: AnyObject) {
      underlying = .scalar(scalar)
      count = 0
      self.name = ""
    }

    private init(_ sequence: Wrapped<[AnyObject]>) {
      underlying = .sequence(sequence)
      count = 0
      self.name = ""
    }

    deinit {
      if case .opaque(let iter) = underlying {
        ccv_cnnp_dataframe_iter_free(iter)
      }
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

    fileprivate init(_ iter: OpaquePointer, count: Int) {
      _iter = iter
      self.count = count
    }

    deinit {
      ccv_cnnp_dataframe_iter_free(_iter)
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
      let data = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: columnCount)
      let retval = ccv_cnnp_dataframe_iter_next(_iter, data, Int32(columnCount), streamContext?._stream)
      guard retval == 0 else { return nil }
      var columnData = [AnyObject]()
      for i in 0..<columnCount {
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
    private let columnCount: Int

    fileprivate init(_ iter: OpaquePointer, columnCount: Int, count: Int) {
      _iter = iter
      self.columnCount = columnCount
      self.count = count
    }

    deinit {
      ccv_cnnp_dataframe_iter_free(_iter)
    }
  }
}
