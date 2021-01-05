import C_nnc

public protocol DataSeries: Sequence {
  func prefetch(_ i: Int, streamContext: StreamContext?)
  func next(_ streamContext: StreamContext?) -> Element?
}

extension DataSeries {
  public func prefetch(_ i: Int) {
    self.prefetch(i, streamContext: nil)
  }
  public func next() -> Element? {
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

final class _DataFrame {
  private let underlying: AnyObject?
  private let parent: _DataFrame?
  let dataframe: OpaquePointer
  init(dataframe: OpaquePointer, parent: _DataFrame? = nil, underlying: AnyObject? = nil) {
    self.dataframe = dataframe
    self.parent = parent
    self.underlying = underlying
  }

  func shuffle() {
    if let parent = parent {
      // Only shuffle the source.
      parent.shuffle()
    } else {
      ccv_cnnp_dataframe_shuffle(dataframe)
    }
  }

  var count: Int {
    return Int(ccv_cnnp_dataframe_row_count(dataframe))
  }

  deinit {
    ccv_cnnp_dataframe_free(dataframe)
  }
}

/// A pandas-inspired dataframe. Dataframe is a tabular data representation. This particular
/// Dataframe implementation is most useful to implement data feeder pipeline. You need some
/// transformations so some text or images can be transformed into tensors for a model to consume.
/// Dataframe can be used to implement that pipeline.
public struct DataFrame {

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

  var columnProperties: [String: ColumnProperty]

  /**
   * List of columns within this dataframe.
   */
  public var columns: [String] {
    return Array(columnProperties.keys)
  }

  let _dataframe: _DataFrame

  init(dataframe: _DataFrame, columnProperties: [String: ColumnProperty]) {
    CmdParamsFactory.factory.sink()
    _dataframe = dataframe
    self.columnProperties = columnProperties
  }

  /**
   * Initialize a dataframe from a sequence of objects.
   */
  public init<S: Sequence>(from sequence: S, name: String = "0") {
    CmdParamsFactory.factory.sink()
    let underlying = Wrapped(Array(sequence) as [AnyObject])
    var column_data = ccv_cnnp_column_data_t()
    let propertyType: ColumnProperty.PropertyType
    if underlying.value.count > 0 && underlying.value[0] is AnyTensor {
      column_data.data_enum = { _, row_idxs, row_size, data, context, _ in
        guard let row_idxs = row_idxs else { return }
        guard let data = data else { return }
        let underlying = Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context!).takeUnretainedValue()
        for i in 0..<Int(row_size) {
          let idx = Int((row_idxs + i).pointee)
          let tensor = underlying.value[idx] as! AnyTensor
          (data + i).initialize(to: tensor.cTensor)
        }
      }
      propertyType = .tensor
    } else {
      column_data.data_enum = { _, row_idxs, row_size, data, context, _ in
        guard let row_idxs = row_idxs else { return }
        guard let data = data else { return }
        let underlying = Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context!).takeUnretainedValue()
        for i in 0..<Int(row_size) {
          let idx = Int((row_idxs + i).pointee)
          let value = underlying.value[idx]
          if let opaque = data[i] {
            Unmanaged<AnyObject>.fromOpaque(opaque).release()
          }
          (data + i).initialize(to: Unmanaged.passRetained(value).toOpaque())
        }
      }
      propertyType = .object
    }
    column_data.data_deinit = { data, _ in
      guard let data = data else { return }
      Unmanaged<AnyObject>.fromOpaque(data).release()
    }
    column_data.context_deinit = { context in
      guard let context = context else { return }
      Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context).release()
    }
    let dataframe: OpaquePointer = name.withCString {
      column_data.name = UnsafeMutablePointer<Int8>(mutating: $0)
      column_data.context = Unmanaged.passRetained(underlying).toOpaque()
      return ccv_cnnp_dataframe_new(&column_data, 1, Int32(underlying.value.count))!
    }
    columnProperties = [name: ColumnProperty(index: 0, type: propertyType)]
    _dataframe = _DataFrame(dataframe: dataframe, underlying: underlying)
  }

  /**
   * Shuffle the dataframe.
   */
  public mutating func shuffle() {
    _dataframe.shuffle()
  }

  public subscript(firstIndex: String, secondIndex: String, indices: String...) -> ManyUntypedSeries
  {
    var properties = [ColumnProperty]()
    properties.append(columnProperties[firstIndex]!)
    properties.append(columnProperties[secondIndex]!)
    for index in indices {
      properties.append(columnProperties[index]!)
    }
    let rowCount = ccv_cnnp_dataframe_row_count(_dataframe.dataframe)
    return ManyUntypedSeries(count: Int(rowCount), properties: properties, dataframe: _dataframe)
  }

  public subscript<S: Sequence>(indices: S) -> ManyUntypedSeries where S.Element == String {
    let properties = indices.map { columnProperties[$0]! }
    assert(properties.count > 0)
    let rowCount = ccv_cnnp_dataframe_row_count(_dataframe.dataframe)
    return ManyUntypedSeries(count: Int(rowCount), properties: properties, dataframe: _dataframe)
  }

  public subscript(index: String) -> UntypedSeries? {
    get {
      guard let columnProperty = columnProperties[index] else {
        return nil
      }
      let rowCount = ccv_cnnp_dataframe_row_count(_dataframe.dataframe)
      return UntypedSeries(
        count: Int(rowCount), name: index, property: columnProperty, dataframe: _dataframe)
    }
    set(v) {
      guard let v = v else {
        columnProperties[index] = nil
        return
      }
      switch v.action {
      case .iterator:
        columnProperties[index] = columnProperties[v.name!]!
      case .scalar(let scalar):
        self.add(from: scalar, name: index)
      case .sequence(let sequence):
        self.add(from: sequence, name: index)
      case .map(let property, let mapper, let outputType):
        self.add(map: mapper, property: property, outputType: outputType, name: index)
      case .multimap(let properties, let mapper, let outputType):
        self.add(multimap: mapper, properties: properties, outputType: outputType, name: index)
      case .native(let property, let transformer, let context):
        let dataframe = _dataframe.dataframe
        columnProperties[index] = transformer(dataframe, property, index, context)
      }
    }
  }

  public subscript<Element>(index: String, type: Element.Type) -> TypedSeries<Element> {
    let columnProperty = columnProperties[index]!
    let rowCount = ccv_cnnp_dataframe_row_count(_dataframe.dataframe)
    return TypedSeries(count: Int(rowCount), property: columnProperty, dataframe: _dataframe)
  }

  /**
   * How many rows in the dataframe.
   */
  public var count: Int {
    return _dataframe.count
  }
}

extension DataFrame {
  final class _Iterator {
    let iterator: OpaquePointer
    init(iterator: OpaquePointer) {
      self.iterator = iterator
    }
    func setZero() {
      ccv_cnnp_dataframe_iter_set_cursor(iterator, 0)
    }
    deinit {
      ccv_cnnp_dataframe_iter_free(iterator)
    }
  }
}

enum UntypedSeriesAction {
  // getter
  case iterator
  // setter
  case scalar(AnyObject)
  case sequence(DataFrame.Wrapped<[AnyObject]>)
  case map(
    DataFrame.ColumnProperty, (AnyObject) -> AnyObject, DataFrame.ColumnProperty.PropertyType)
  case multimap(
    [DataFrame.ColumnProperty], ([AnyObject]) -> AnyObject, DataFrame.ColumnProperty.PropertyType)
  case native(
    DataFrame.ColumnProperty,
    (OpaquePointer, DataFrame.ColumnProperty, String, AnyObject?) -> DataFrame.ColumnProperty,
    AnyObject?)
}

extension DataFrame {

  public final class UntypedSeries: DataSeries {

    public typealias Element = AnyObject

    public func makeIterator() -> DataSeriesIterator<UntypedSeries> {
      switch action {
      case .iterator:
        iterator.setZero()
      default:
        fatalError()
      }
      return DataSeriesIterator(self)
    }

    public func prefetch(_ i: Int, streamContext: StreamContext?) {
      switch action {
      case .iterator:
        ccv_cnnp_dataframe_iter_prefetch(iterator.iterator, Int32(i), streamContext?._stream)
      default:
        fatalError()
      }
    }

    public func next(_ streamContext: StreamContext?) -> AnyObject? {
      switch action {
      case .iterator:
        var data: UnsafeMutableRawPointer? = nil
        let retval = ccv_cnnp_dataframe_iter_next(
          iterator.iterator, &data, 1, streamContext?._stream)
        guard retval == 0 else { return nil }
        if data == nil {
          return nil
        }
        switch property!.type {
        case .object:
          return Unmanaged<AnyObject>.fromOpaque(data!).takeUnretainedValue()
        case .tensor:
          return AnyTensorStorage(
            data!.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false
          )
          .toAnyTensor() as AnyObject
        }
      default:
        fatalError()
      }
    }

    public var underestmiatedCount: Int {
      return count
    }

    public let count: Int
    private lazy var iterator: _Iterator = {
      var i: Int32 = Int32(property!.index)
      let _dataframe = dataframe!.dataframe
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, &i, 1)!
      return _Iterator(iterator: iter)
    }()

    fileprivate let action: UntypedSeriesAction
    fileprivate let name: String?

    let property: ColumnProperty?
    let dataframe: _DataFrame?

    fileprivate init(count: Int, name: String, property: ColumnProperty, dataframe: _DataFrame) {
      action = .iterator
      self.count = count
      self.name = name
      self.property = property
      self.dataframe = dataframe
    }

    init(_ action: UntypedSeriesAction) {
      self.action = action
      count = 0
      name = nil
      property = nil
      dataframe = nil
    }
  }
}

extension DataFrame {
  public final class ManyUntypedSeries: DataSeries {

    public typealias Element = [AnyObject]

    public func makeIterator() -> DataSeriesIterator<ManyUntypedSeries> {
      iterator.setZero()
      return DataSeriesIterator(self)
    }

    public func prefetch(_ i: Int, streamContext: StreamContext?) {
      ccv_cnnp_dataframe_iter_prefetch(iterator.iterator, Int32(i), streamContext?._stream)
    }

    public func next(_ streamContext: StreamContext?) -> [AnyObject]? {
      let count = properties.count
      let data = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: count)
      let retval = ccv_cnnp_dataframe_iter_next(
        iterator.iterator, data, Int32(count), streamContext?._stream)
      guard retval == 0 else { return nil }
      var columnData = [AnyObject]()
      for i in 0..<count {
        let data = data[i]!
        let object: AnyObject
        switch properties[i].type {
        case .object:
          object = Unmanaged<AnyObject>.fromOpaque(data).takeUnretainedValue()
        case .tensor:
          object =
            AnyTensorStorage(data.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false)
            .toAnyTensor() as AnyObject
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

    private lazy var iterator: _Iterator = {
      let i: [Int32] = properties.map { Int32($0.index) }
      let _dataframe = dataframe.dataframe
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, i, Int32(i.count))!
      return _Iterator(iterator: iter)
    }()

    let properties: [ColumnProperty]
    let dataframe: _DataFrame

    fileprivate init(count: Int, properties: [ColumnProperty], dataframe: _DataFrame) {
      self.count = count
      self.properties = properties
      self.dataframe = dataframe
    }
  }
}

extension DataFrame {
  public final class TypedSeries<Element>: DataSeries {

    public typealias Element = Element

    public func makeIterator() -> DataSeriesIterator<TypedSeries> {
      iterator.setZero()
      return DataSeriesIterator(self)
    }

    public func prefetch(_ i: Int, streamContext: StreamContext?) {
      ccv_cnnp_dataframe_iter_prefetch(iterator.iterator, Int32(i), streamContext?._stream)
    }

    public func next(_ streamContext: StreamContext?) -> Element? {
      var data: UnsafeMutableRawPointer? = nil
      let retval = ccv_cnnp_dataframe_iter_next(iterator.iterator, &data, 1, streamContext?._stream)
      guard retval == 0 else { return nil }
      if data == nil {
        return nil
      }
      switch property.type {
      case .object:
        return Unmanaged<AnyObject>.fromOpaque(data!).takeUnretainedValue() as? Element
      case .tensor:
        return AnyTensorStorage(
          data!.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false
        )
        .toTensor(Element.self)
      }
    }

    public var underestmiatedCount: Int {
      return count
    }

    public let count: Int

    private lazy var iterator: _Iterator = {
      var i: Int32 = Int32(property.index)
      let _dataframe = dataframe.dataframe
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, &i, 1)!
      return _Iterator(iterator: iter)
    }()
    let property: ColumnProperty
    let dataframe: _DataFrame

    fileprivate init(count: Int, property: ColumnProperty, dataframe: _DataFrame) {
      self.count = count
      self.property = property
      self.dataframe = dataframe
    }
  }
}

// MARK - Scalar support

extension DataFrame.UntypedSeries {
  public static func from(_ scalar: Any) -> DataFrame.UntypedSeries {
    return DataFrame.UntypedSeries(.scalar(scalar as AnyObject))  // Wrap this.
  }
}

extension DataFrame {
  private mutating func add(from scalar: AnyObject, name: String) {
    let dataframe = _dataframe.dataframe
    if scalar is AnyTensor {
      let index = ccv_cnnp_dataframe_add(
        dataframe,
        { _, row_idxs, row_size, data, context, _ in
          guard let data = data else { return }
          let tensor = Unmanaged<AnyObject>.fromOpaque(context!).takeUnretainedValue() as! AnyTensor
          for i in 0..<Int(row_size) {
            (data + i).initialize(to: tensor.cTensor)
          }
        }, 0, nil, Unmanaged.passRetained(scalar).toOpaque(),
        { context in
          guard let context = context else { return }
          Unmanaged<AnyObject>.fromOpaque(context).release()
        }, name)
      columnProperties[name] = ColumnProperty(index: Int(index), type: .tensor)
    } else {
      let index = ccv_cnnp_dataframe_add(
        dataframe,
        { _, row_idxs, row_size, data, context, _ in
          guard let data = data else { return }
          for i in 0..<Int(row_size) {
            (data + i).initialize(to: context)
          }
        }, 0, nil, Unmanaged.passRetained(scalar).toOpaque(),
        { context in
          guard let context = context else { return }
          Unmanaged<AnyObject>.fromOpaque(context).release()
        }, name)
      columnProperties[name] = ColumnProperty(index: Int(index), type: .object)
    }
  }
}

// MARK - Sequence support

extension DataFrame.UntypedSeries {
  /// Create a new column from a sequence of objects.
  public static func from<S: Sequence>(_ sequence: S) -> DataFrame.UntypedSeries {
    return DataFrame.UntypedSeries(.sequence(DataFrame.Wrapped(Array(sequence) as [AnyObject])))
  }
}

extension DataFrame {
  private mutating func add(from sequence: Wrapped<[AnyObject]>, name: String) {
    assert(sequence.value.count == count)
    let dataframe = _dataframe.dataframe
    if sequence.value.count > 0 && sequence.value[0] is AnyTensor {
      let index = ccv_cnnp_dataframe_add(
        dataframe,
        { _, row_idxs, row_size, data, context, _ in
          guard let row_idxs = row_idxs else { return }
          guard let data = data else { return }
          let underlying = Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context!)
            .takeUnretainedValue()
          for i in 0..<Int(row_size) {
            let idx = Int((row_idxs + i).pointee)
            let tensor = underlying.value[idx] as! AnyTensor
            (data + i).initialize(to: tensor.cTensor)
          }
        }, 0,
        { data, _ in
          guard let data = data else { return }
          Unmanaged<AnyObject>.fromOpaque(data).release()
        }, Unmanaged.passRetained(sequence).toOpaque(),
        { context in
          guard let context = context else { return }
          Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context).release()
        }, name)
      columnProperties[name] = ColumnProperty(index: Int(index), type: .tensor)
    } else {
      let index = ccv_cnnp_dataframe_add(
        dataframe,
        { _, row_idxs, row_size, data, context, _ in
          guard let row_idxs = row_idxs else { return }
          guard let data = data else { return }
          let underlying = Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context!)
            .takeUnretainedValue()
          for i in 0..<Int(row_size) {
            let idx = Int((row_idxs + i).pointee)
            let value = underlying.value[idx]
            if let opaque = data[i] {
              Unmanaged<AnyObject>.fromOpaque(opaque).release()
            }
            (data + i).initialize(to: Unmanaged.passRetained(value).toOpaque())
          }
        }, 0,
        { data, _ in
          guard let data = data else { return }
          Unmanaged<AnyObject>.fromOpaque(data).release()
        }, Unmanaged.passRetained(sequence).toOpaque(),
        { context in
          guard let context = context else { return }
          Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context).release()
        }, name)
      columnProperties[name] = ColumnProperty(index: Int(index), type: .object)
    }
  }
}

extension DataFrame.UntypedSeries {
  /// Create a new column by applying some transformations on an existing column.
  public func map<T, U>(_ mapper: @escaping (T) -> U) -> DataFrame.UntypedSeries {
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

extension DataFrame.TypedSeries {
  /// Create a new column by applying some transformations on an existing column.
  public func map<U>(_ mapper: @escaping (Element) -> U) -> DataFrame.UntypedSeries {
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

extension DataFrame {
  private final class WrappedMapper {
    let property: ColumnProperty
    let map: (AnyObject) -> AnyObject
    let outputType: ColumnProperty.PropertyType
    var tensors: [OpaquePointer: AnyTensorStorage]?
    init(
      property: ColumnProperty, map: @escaping (AnyObject) -> AnyObject,
      outputType: ColumnProperty.PropertyType
    ) {
      self.property = property
      self.map = map
      self.outputType = outputType
      switch outputType {
      case .object:
        tensors = nil
      case .tensor:
        tensors = [OpaquePointer: AnyTensorStorage]()
      }
    }
  }
  static func add(
    to dataframe: OpaquePointer, map: @escaping (AnyObject) -> AnyObject, property: ColumnProperty,
    outputType: ColumnProperty.PropertyType, name: String
  ) -> ColumnProperty {
    var inputIndex = Int32(property.index)
    let index = ccv_cnnp_dataframe_map(
      dataframe,
      { input, _, row_size, data, context, _ in
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
            object =
              AnyTensorStorage(
                inputData[i]!.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false
              ).toAnyTensor() as AnyObject
          }
          let output = wrappedMapper.map(object)
          switch wrappedMapper.outputType {
          case .object:
            if let opaque = data[i] {
              Unmanaged<AnyObject>.fromOpaque(opaque).release()
            }
            (data + i).initialize(to: Unmanaged.passRetained(output).toOpaque())
          case .tensor:
            if let opaque = data[i] {
              wrappedMapper.tensors![OpaquePointer(opaque)] = nil
            }
            let tensor = output as! AnyTensor
            wrappedMapper.tensors![OpaquePointer(tensor.cTensor)] = tensor.storage
            (data + i).initialize(to: tensor.cTensor)
          }
        }
      }, 0,
      { object, context in
        guard let object = object else { return }
        let wrappedMapper = Unmanaged<WrappedMapper>.fromOpaque(context!).takeUnretainedValue()
        switch wrappedMapper.outputType {
        case .object:
          Unmanaged<AnyObject>.fromOpaque(object).release()
        case .tensor:
          wrappedMapper.tensors![OpaquePointer(object)] = nil
        }
      }, &inputIndex, 1,
      Unmanaged.passRetained(WrappedMapper(property: property, map: map, outputType: outputType))
        .toOpaque(),
      { mapper in
        Unmanaged<WrappedMapper>.fromOpaque(mapper!).release()
      }, name)
    return ColumnProperty(index: Int(index), type: outputType)
  }
  private mutating func add(
    map: @escaping (AnyObject) -> AnyObject, property: ColumnProperty,
    outputType: ColumnProperty.PropertyType, name: String
  ) {
    columnProperties[name] = Self.add(
      to: _dataframe.dataframe, map: map, property: property, outputType: outputType, name: name)
  }
}

extension DataFrame.ManyUntypedSeries {
  /// Create a new column by applying some transformations on some existing columns.
  public func map<C0, C1, U>(_ mapper: @escaping (C0, C1) -> U) -> DataFrame.UntypedSeries {
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
  /// Create a new column by applying some transformations on some existing columns.
  public func map<C0, C1, C2, U>(_ mapper: @escaping (C0, C1, C2) -> U) -> DataFrame.UntypedSeries {
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
  /// Create a new column by applying some transformations on some existing columns.
  public func map<C0, C1, C2, C3, U>(_ mapper: @escaping (C0, C1, C2, C3) -> U)
    -> DataFrame.UntypedSeries
  {
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
  /// Create a new column by applying some transformations on some existing columns.
  public func map<C0, C1, C2, C3, C4, U>(_ mapper: @escaping (C0, C1, C2, C3, C4) -> U)
    -> DataFrame.UntypedSeries
  {
    assert(properties.count == 5)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4)
        as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  /// Create a new column by applying some transformations on some existing columns.
  public func map<C0, C1, C2, C3, C4, C5, U>(_ mapper: @escaping (C0, C1, C2, C3, C4, C5) -> U)
    -> DataFrame.UntypedSeries
  {
    assert(properties.count == 6)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(
        objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4,
        objs[5] as! C5) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  /// Create a new column by applying some transformations on some existing columns.
  public func map<C0, C1, C2, C3, C4, C5, C6, U>(
    _ mapper: @escaping (C0, C1, C2, C3, C4, C5, C6) -> U
  ) -> DataFrame.UntypedSeries {
    assert(properties.count == 7)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(
        objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4,
        objs[5] as! C5, objs[6] as! C6) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  /// Create a new column by applying some transformations on some existing columns.
  public func map<C0, C1, C2, C3, C4, C5, C6, C7, U>(
    _ mapper: @escaping (C0, C1, C2, C3, C4, C5, C6, C7) -> U
  ) -> DataFrame.UntypedSeries {
    assert(properties.count == 8)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(
        objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4,
        objs[5] as! C5, objs[6] as! C6, objs[7] as! C7) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  /// Create a new column by applying some transformations on some existing columns.
  public func map<C0, C1, C2, C3, C4, C5, C6, C7, C8, U>(
    _ mapper: @escaping (C0, C1, C2, C3, C4, C5, C6, C7, C8) -> U
  ) -> DataFrame.UntypedSeries {
    assert(properties.count == 9)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(
        objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4,
        objs[5] as! C5, objs[6] as! C6, objs[7] as! C7, objs[8] as! C8) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
  /// Create a new column by applying some transformations on some existing columns.
  public func map<C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, U>(
    _ mapper: @escaping (C0, C1, C2, C3, C4, C5, C6, C7, C8, C9) -> U
  ) -> DataFrame.UntypedSeries {
    assert(properties.count == 10)
    let wrappedMapper = { (objs: [AnyObject]) -> AnyObject in
      return mapper(
        objs[0] as! C0, objs[1] as! C1, objs[2] as! C2, objs[3] as! C3, objs[4] as! C4,
        objs[5] as! C5, objs[6] as! C6, objs[7] as! C7, objs[8] as! C8, objs[9] as! C9) as AnyObject
    }
    if U.self is AnyTensor.Type || U.self == AnyTensor.self {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .tensor))
    } else {
      return DataFrame.UntypedSeries(.multimap(properties, wrappedMapper, .object))
    }
  }
}

extension DataFrame {
  private final class WrappedManyMapper {
    let properties: [ColumnProperty]
    let map: ([AnyObject]) -> AnyObject
    let outputType: ColumnProperty.PropertyType
    var tensors: [OpaquePointer: AnyTensorStorage]?
    init(
      properties: [ColumnProperty], map: @escaping ([AnyObject]) -> AnyObject,
      outputType: ColumnProperty.PropertyType
    ) {
      self.properties = properties
      self.map = map
      self.outputType = outputType
      switch outputType {
      case .object:
        tensors = nil
      case .tensor:
        tensors = [OpaquePointer: AnyTensorStorage]()
      }
    }
  }
  static func add(
    to dataframe: OpaquePointer, multimap: @escaping ([AnyObject]) -> AnyObject,
    properties: [ColumnProperty], outputType: ColumnProperty.PropertyType, name: String
  ) -> ColumnProperty {
    let inputIndex = properties.map { Int32($0.index) }
    let index = ccv_cnnp_dataframe_map(
      dataframe,
      { input, _, row_size, data, context, _ in
        guard let input = input else { return }
        guard let data = data else { return }
        let wrappedManyMapper = Unmanaged<WrappedManyMapper>.fromOpaque(context!)
          .takeUnretainedValue()
        for i in 0..<Int(row_size) {
          var objects = [AnyObject]()
          for (j, property) in wrappedManyMapper.properties.enumerated() {
            let object: AnyObject
            switch property.type {
            case .object:
              object = Unmanaged<AnyObject>.fromOpaque(input[j]![i]!).takeUnretainedValue()
            case .tensor:
              object =
                AnyTensorStorage(
                  input[j]![i]!.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false
                ).toAnyTensor() as AnyObject
            }
            objects.append(object)
          }
          let output = wrappedManyMapper.map(objects)
          switch wrappedManyMapper.outputType {
          case .object:
            if let opaque = data[i] {
              Unmanaged<AnyObject>.fromOpaque(opaque).release()
            }
            (data + i).initialize(to: Unmanaged.passRetained(output).toOpaque())
          case .tensor:
            if let opaque = data[i] {
              wrappedManyMapper.tensors![OpaquePointer(opaque)] = nil
            }
            let tensor = output as! AnyTensor
            wrappedManyMapper.tensors![OpaquePointer(tensor.cTensor)] = tensor.storage
            (data + i).initialize(to: tensor.cTensor)
          }
        }
      }, 0,
      { object, context in
        guard let object = object else { return }
        let wrappedManyMapper = Unmanaged<WrappedManyMapper>.fromOpaque(context!)
          .takeUnretainedValue()
        switch wrappedManyMapper.outputType {
        case .object:
          Unmanaged<AnyObject>.fromOpaque(object).release()
        case .tensor:
          wrappedManyMapper.tensors![OpaquePointer(object)] = nil
        }
      }, inputIndex, Int32(inputIndex.count),
      Unmanaged.passRetained(
        WrappedManyMapper(properties: properties, map: multimap, outputType: outputType)
      ).toOpaque(),
      { mapper in
        Unmanaged<WrappedManyMapper>.fromOpaque(mapper!).release()
      }, name)
    return ColumnProperty(index: Int(index), type: outputType)
  }
  private mutating func add(
    multimap: @escaping ([AnyObject]) -> AnyObject, properties: [ColumnProperty],
    outputType: ColumnProperty.PropertyType, name: String
  ) {
    columnProperties[name] = Self.add(
      to: _dataframe.dataframe, multimap: multimap, properties: properties, outputType: outputType,
      name: name)
  }
}
