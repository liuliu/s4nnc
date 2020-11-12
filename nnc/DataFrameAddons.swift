import C_nnc

// MARK - Load CSV

extension DataFrame {
  public convenience init?(fromCSV filePath: String, automaticUseHeader: Bool = true, delimiter: String = ",", quotation: String = "\"") {
    var columnSize: Int32 = 0
    let fileHandle = fopen(filePath, "r")
    guard fileHandle != nil else { return nil }
    assert(delimiter.count == 1)
    assert(quotation.count == 1)
    var _delimiter = delimiter
    let delim = _delimiter.withUTF8 { $0.withMemoryRebound(to: CChar.self) { $0[0] } }
    var _quotation = quotation
    let quote = _quotation.withUTF8 { $0.withMemoryRebound(to: CChar.self) { $0[0] } }
    let dataframe_ = ccv_cnnp_dataframe_from_csv_new(fileHandle, Int32(CCV_CNNP_DATAFRAME_CSV_FILE), 0, delim, quote, (automaticUseHeader ? 1 : 0), &columnSize)
    fclose(fileHandle)
    guard let dataframe = dataframe_ else {
      return nil
    }
    guard columnSize > 0 else {
      ccv_cnnp_dataframe_free(dataframe)
      return nil
    }
    var columnProperties = [String: ColumnProperty]()
    for i in 0..<columnSize {
      var inputIndex: Int32 = Int32(i)
      let stringIndex = ccv_cnnp_dataframe_map(dataframe, { input, _, row_size, data, context, _ in
        guard let input = input else { return }
        guard let data = data else { return }
        let inputData = input[0]!
        for i in 0..<Int(row_size) {
          guard let str = inputData[i].map({ String(cString: $0.assumingMemoryBound(to: Int8.self)) }) else {
            continue
          }
          if let opaque = data[i] {
            Unmanaged<AnyObject>.fromOpaque(opaque).release()
          }
          let obj = str as AnyObject
          let utf8 = Unmanaged<AnyObject>.passRetained(obj).toOpaque()
          (data + i).initialize(to: utf8)
        }
      }, 0, { obj, _ in
        guard let obj = obj else { return }
        Unmanaged<AnyObject>.fromOpaque(obj).release()
      }, &inputIndex, 1, nil, nil, nil)
      let columnName: String
      if automaticUseHeader {
        let cString = ccv_cnnp_dataframe_column_name(dataframe, Int32(i))
        columnName = cString.map { String(cString: $0) } ?? "\(i)"
      } else {
        columnName = "\(i)"
      }
      columnProperties[columnName] = ColumnProperty(index: Int(stringIndex), type: .object)
    }
    self.init(dataframe: dataframe, underlying: nil, columnProperties: columnProperties)
  }
}

// MARK - Batching

extension DataFrame {
  convenience init(dataframe: DataFrame, properties: [ColumnProperty], size: Int, repeating: Int) {
    let _dataframe = dataframe._dataframe
    let columnSize: Int32 = Int32(properties.count)
    let indices: [Int32] = properties.map { Int32($0.index) }
    let batching = ccv_cnnp_dataframe_batching_new(_dataframe, indices, columnSize, Int32(size), Int32(repeating), Int32(CCV_TENSOR_FORMAT_NCHW))!
    var columnProperties = [String: ColumnProperty]()
    if repeating == 1 {
      for (i, property) in properties.enumerated() {
        // These must have names.
        let name = ccv_cnnp_dataframe_column_name(_dataframe, Int32(property.index))!
        let index = ccv_cnnp_dataframe_extract_tuple(batching, 0, Int32(i), name)
        columnProperties[String(cString: name)] = ColumnProperty(index: Int(index), type: .tensor)
      }
    } else {
      for i in 0..<repeating {
        for (j, property) in properties.enumerated() {
          // These must have names.
          let name = ccv_cnnp_dataframe_column_name(_dataframe, Int32(property.index))!
          let index = ccv_cnnp_dataframe_extract_tuple(batching, 0, Int32(i * properties.count + j), name)
          columnProperties["\(String(cString: name))_\(i)"] = ColumnProperty(index: Int(index), type: .tensor)
        }
      }
    }
    self.init(dataframe: batching, underlying: dataframe, columnProperties: columnProperties)
  }
  public convenience init(batchOf: DataFrame.UntypedSeries, size: Int, repeating: Int = 1) {
    precondition(repeating > 0)
    guard let property = batchOf.property,
          let dataframe = batchOf.dataframe else {
      fatalError("An UntypedSeries has to be referenced from existing dataframe. Cannot be a temporary one.")
    }
    precondition(property.type == .tensor)
    self.init(dataframe: dataframe, properties: [property], size: size, repeating: repeating)
  }
  public convenience init<T: AnyTensor>(batchOf: DataFrame.TypedSeries<T>, size: Int, repeating: Int = 1) {
    let property = batchOf.property
    precondition(property.type == .tensor)
    self.init(dataframe: batchOf.dataframe, properties: [property], size: size, repeating: repeating)
  }
  public convenience init(batchOf: DataFrame.ManyUntypedSeries, size: Int, repeating: Int = 1) {
    let properties = batchOf.properties
    for property in properties {
      precondition(property.type == .tensor)
    }
    self.init(dataframe: batchOf.dataframe, properties: properties, size: size, repeating: repeating)
  }
}

// MARK - Load image.

extension DataFrame {
  static func addToLoadImage(_ _dataframe: OpaquePointer, _ property: ColumnProperty, _ name: String, _: AnyObject?) -> ColumnProperty {
    var inputIndex = Int32(property.index)
    let pathIndex = ccv_cnnp_dataframe_map(_dataframe, { input, _, row_size, data, context, _ in
      guard let input = input else { return }
      guard let data = data else { return }
      let inputData = input[0]!
      for i in 0..<Int(row_size) {
        var path = Unmanaged<AnyObject>.fromOpaque(inputData[i]!).takeUnretainedValue() as! String
        if let container = data[i] {
          let string = container.assumingMemoryBound(to: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>>.self)[0]
          string.deallocate()
          container.deallocate()
        }
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
    }, &inputIndex, 1, nil, nil, nil)
    let index = ccv_cnnp_dataframe_read_image(_dataframe, pathIndex, 0, name)
    return ColumnProperty(index: Int(index), type: .tensor)
  }
}

public extension DataFrame.UntypedSeries {
  func toLoadImage() -> DataFrame.UntypedSeries {
    guard let property = property else {
      fatalError("Can only load from series from DataFrame")
    }
    precondition(property.type == .object)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToLoadImage, nil))
  }
}

public extension DataFrame.TypedSeries where Element == String {
  func toLoadImage() -> DataFrame.UntypedSeries {
    precondition(property.type == .object)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToLoadImage, nil))
  }
}

// MARK - One-hot.

final class OneHotParams {
  let dataType: DataType
  let count: Int
  let onval: Float
  let offval: Float
  init(dataType: DataType, count: Int, onval: Float, offval:Float) {
    self.dataType = dataType
    self.count = count
    self.onval = onval
    self.offval = offval
  }
}

extension DataFrame {
  static func addToOneHot(_ _dataframe: OpaquePointer, _ property: ColumnProperty, _ name: String, _ params: AnyObject?) -> ColumnProperty {
    var inputIndex = Int32(property.index)
    let oneHotParams = params! as! OneHotParams
    let intIndex = ccv_cnnp_dataframe_map(_dataframe, { input, _, row_size, data, context, _ in
      guard let input = input else { return }
      guard let data = data else { return }
      let inputData = input[0]!
      for i in 0..<Int(row_size) {
        let int = Unmanaged<AnyObject>.fromOpaque(inputData[i]!).takeUnretainedValue() as! Int
        if let container = data[i] {
          container.assumingMemoryBound(to: Int32.self).initialize(to: Int32(int))
        } else {
          let container = UnsafeMutablePointer<Int32>.allocate(capacity: 1)
          container.initialize(to: Int32(int))
          (data + i).initialize(to: container)
        }
      }
    }, 0, { container, _ in
      guard let container = container else { return }
      container.deallocate()
    }, &inputIndex, 1, nil, nil, nil)
    let index = ccv_cnnp_dataframe_one_hot(_dataframe, intIndex, 0, Int32(oneHotParams.count), oneHotParams.onval, oneHotParams.offval, oneHotParams.dataType.toC, Int32(CCV_TENSOR_FORMAT_NCHW), name)
    return ColumnProperty(index: Int(index), type: .tensor)
  }
}

public extension DataFrame.UntypedSeries {
  func toOneHot<Element: TensorNumeric>(_ dataType: Element.Type, count: Int, onval: Float = 1, offval: Float = 0) -> DataFrame.UntypedSeries {
    guard let property = property else {
      fatalError("Can only load from series from DataFrame")
    }
    precondition(property.type == .object)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToOneHot, OneHotParams(dataType: Element.dataType, count: count, onval: onval, offval: offval)))
  }
}

public extension DataFrame.TypedSeries where Element == Int {
  func toOneHot<Element: TensorNumeric>(_ dataType: Element.Type, count: Int, onval: Float = 1, offval: Float = 0) -> DataFrame.UntypedSeries {
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToOneHot, OneHotParams(dataType: Element.dataType, count: count, onval: onval, offval: offval)))
  }
}

// MARK - Copy to GPU.

extension DataFrame {
  static func addToOneGPU(_ _dataframe: OpaquePointer, _ property: ColumnProperty, _ name: String, _ params: AnyObject?) -> ColumnProperty {
    var inputIndex = Int32(property.index)
    let tupleIndex = ccv_cnnp_dataframe_make_tuple(_dataframe, &inputIndex, 1, nil)
    let ordinal = params! as! Int
    let copyIndex = ccv_cnnp_dataframe_copy_to_gpu(_dataframe, tupleIndex, 0, 1, Int32(ordinal), nil)
    let index = ccv_cnnp_dataframe_extract_tuple(_dataframe, copyIndex, 0, name)
    return ColumnProperty(index: Int(index), type: .tensor)
  }
}

public extension DataFrame.UntypedSeries {
  func toGPU(_ ordinal: Int = 0) -> DataFrame.UntypedSeries {
    guard let property = property else {
      fatalError("Can only load from series from DataFrame")
    }
    precondition(property.type == .tensor)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToOneGPU, ordinal as AnyObject))
  }
}

public extension DataFrame.TypedSeries where Element: AnyTensor {
  func toGPU(_ ordinal: Int = 0) -> DataFrame.UntypedSeries {
    precondition(property.type == .tensor)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToOneGPU, ordinal as AnyObject))
  }
}

public extension DataFrame {
  struct ManyUntypedSeriesToGPU {
    var index: Int
    var namedIndex: [String: Int]
  }
}

extension DataFrame {
  static func extractTensorTuple(_ _dataframe: OpaquePointer, _ property: ColumnProperty, _ name: String, _ params: AnyObject?) -> ColumnProperty {
    let inputIndex = Int32(property.index)
    let tupleIndex = params! as! Int
    let index = ccv_cnnp_dataframe_extract_tuple(_dataframe, inputIndex, Int32(tupleIndex), name)
    return ColumnProperty(index: Int(index), type: .tensor)
  }
}

public extension DataFrame.ManyUntypedSeriesToGPU {
  subscript(name: String) -> DataFrame.UntypedSeries {
    let tupleIndex = namedIndex[name]!
    let property = DataFrame.ColumnProperty(index: index, type: .tensor)
    return DataFrame.UntypedSeries(.native(property, DataFrame.extractTensorTuple, tupleIndex as AnyObject))
  }
  subscript(tupleIndex: Int) -> DataFrame.UntypedSeries {
    precondition(tupleIndex < namedIndex.count)
    let property = DataFrame.ColumnProperty(index: index, type: .tensor)
    return DataFrame.UntypedSeries(.native(property, DataFrame.extractTensorTuple, tupleIndex as AnyObject))
  }
}

public extension DataFrame.ManyUntypedSeries {
  func toGPU(_ ordinal: Int = 0) -> DataFrame.ManyUntypedSeriesToGPU {
    for property in properties {
      precondition(property.type == .tensor)
    }
    let inputIndex = properties.map { Int32($0.index) }
    let _dataframe = dataframe._dataframe
    let tupleIndex = ccv_cnnp_dataframe_make_tuple(_dataframe, inputIndex, Int32(inputIndex.count), nil)
    let copyIndex = ccv_cnnp_dataframe_copy_to_gpu(_dataframe, tupleIndex, 0, Int32(inputIndex.count), Int32(ordinal), nil)
    var namedIndex = [String: Int]()
    for (i, property) in properties.enumerated() {
      let name = ccv_cnnp_dataframe_column_name(_dataframe, Int32(property.index))!
      namedIndex[String(cString: name)] = i
    }
    return DataFrame.ManyUntypedSeriesToGPU(index: Int(copyIndex), namedIndex: namedIndex)
  }
}

// MARK - One Squared

final class OneSquaredParams {
  let variableLength: Bool
  let maxLength: Int
  init(variableLength: Bool, maxLength: Int) {
    self.variableLength = variableLength
    self.maxLength = maxLength
  }
}

extension DataFrame {
  static func addToOneSquared(_ _dataframe: OpaquePointer, _ property: ColumnProperty, _ name: String, _ params: AnyObject?) -> ColumnProperty {
    var inputIndex = Int32(property.index)
    let oneSquareParams = params! as! OneSquaredParams
    let tupleIndex = ccv_cnnp_dataframe_one_squared(_dataframe, &inputIndex, 1, oneSquareParams.variableLength ? 1 : 0, Int32(oneSquareParams.maxLength), nil)
    let index = ccv_cnnp_dataframe_extract_tuple(_dataframe, tupleIndex, 0, name)
    return ColumnProperty(index: Int(index), type: .tensor)
  }
}

public extension DataFrame.UntypedSeries {
  func toOneSquared(maxLength: Int, variableLength: Bool = true) -> DataFrame.UntypedSeries {
    guard let property = property else {
      fatalError("Can only load from series from DataFrame")
    }
    precondition(property.type == .tensor)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToOneSquared, OneSquaredParams(variableLength: variableLength, maxLength: maxLength)))
  }
}

public extension DataFrame.TypedSeries where Element: AnyTensor {
  func toOneSquared(maxLength: Int, variableLength: Bool = true) -> DataFrame.UntypedSeries {
    precondition(property.type == .tensor)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToOneSquared, OneSquaredParams(variableLength: variableLength, maxLength: maxLength)))
  }
}

// MARK - Truncate

extension DataFrame {
  static func addToTruncate(_ _dataframe: OpaquePointer, _ property: ColumnProperty, _ name: String, _ params: AnyObject?) -> ColumnProperty {
    let otherProperty = (params! as! Wrapped<ColumnProperty>).value
    var inputIndex = Int32(property.index)
    var truncateIndex = Int32(otherProperty.index)
    let tupleIndex = ccv_cnnp_dataframe_truncate(_dataframe, &inputIndex, 1, &truncateIndex, 1, nil)
    let index = ccv_cnnp_dataframe_extract_tuple(_dataframe, tupleIndex, 0, name)
    return ColumnProperty(index: Int(index), type: .tensor)
  }
}

public extension DataFrame.UntypedSeries {
  func toTruncate(_ other: DataFrame.UntypedSeries) -> DataFrame.UntypedSeries {
    guard let property = property,
      let otherProperty = other.property else {
      fatalError("Can only load from series from DataFrame")
    }
    precondition(property.type == .tensor)
    precondition(otherProperty.type == .tensor)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToTruncate, DataFrame.Wrapped(otherProperty)))
  }

  func toTruncate<Element: AnyTensor>(_ other: DataFrame.TypedSeries<Element>) -> DataFrame.UntypedSeries {
    guard let property = property else {
      fatalError("Can only load from series from DataFrame")
    }
    precondition(property.type == .tensor)
    precondition(other.property.type == .tensor)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToTruncate, DataFrame.Wrapped(other.property)))
  }
}

public extension DataFrame.TypedSeries where Element: AnyTensor {
  func toTruncate(_ other: DataFrame.UntypedSeries) -> DataFrame.UntypedSeries {
    guard let otherProperty = other.property else {
      fatalError("Can only load from series from DataFrame")
    }
    precondition(property.type == .tensor)
    precondition(otherProperty.type == .tensor)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToTruncate, DataFrame.Wrapped(otherProperty)))
  }
  func toTruncate<OtherElement: AnyTensor>(_ other: DataFrame.TypedSeries<OtherElement>) -> DataFrame.UntypedSeries {
    precondition(property.type == .tensor)
    precondition(other.property.type == .tensor)
    return DataFrame.UntypedSeries(.native(property, DataFrame.addToTruncate, DataFrame.Wrapped(other.property)))
  }
}
