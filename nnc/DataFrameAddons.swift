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
  convenience init(dataframe: DataFrame, properties: [ColumnProperty], size: Int) {
    let _dataframe = dataframe._dataframe
    let columnSize: Int32 = Int32(properties.count)
    let indices: [Int32] = properties.map { Int32($0.index) }
    let batching = ccv_cnnp_dataframe_batching_new(_dataframe, indices, columnSize, Int32(size), 1, Int32(CCV_TENSOR_FORMAT_NCHW))!
    var columnProperties = [String: ColumnProperty]()
    for (i, property) in properties.enumerated() {
      // These must have names.
      let name = ccv_cnnp_dataframe_column_name(_dataframe, Int32(property.index))!
      let index = ccv_cnnp_dataframe_extract_tuple(batching, 0, Int32(i), name)
      columnProperties[String(cString: name)] = ColumnProperty(index: Int(index), type: .tensor)
    }
    self.init(dataframe: batching, underlying: dataframe, columnProperties: columnProperties)
  }
  public convenience init(batchOf: DataFrame.UntypedSeries, size: Int) {
    guard let property = batchOf.property,
          let dataframe = batchOf.dataframe else {
      fatalError("An UntypedSeries has to be referenced from existing dataframe. Cannot be a temporary one.")
    }
    precondition(property.type == .tensor)
    self.init(dataframe: dataframe, properties: [property], size: size)
  }
  public convenience init(batchOf: DataFrame.TypedSeries<AnyTensor>, size: Int) {
    let property = batchOf.property
    precondition(property.type == .tensor)
    self.init(dataframe: batchOf.dataframe, properties: [property], size: size)
  }
  public convenience init(batchOf: DataFrame.ManyUntypedSeries, size: Int) {
    let properties = batchOf.properties
    for property in properties {
      precondition(property.type == .tensor)
    }
    self.init(dataframe: batchOf.dataframe, properties: properties, size: size)
  }
  public convenience init?(batchOf: DataFrame.UntypedSeries?, size: Int) {
    guard let batchOf = batchOf else { return nil }
    self.init(batchOf: batchOf, size: size)
  }
  public convenience init?(batchOf: DataFrame.TypedSeries<AnyTensor>?, size: Int) {
    guard let batchOf = batchOf else { return nil }
    self.init(batchOf: batchOf, size: size)
  }
  public convenience init?(batchOf: DataFrame.ManyUntypedSeries?, size: Int) {
    guard let batchOf = batchOf else { return nil }
    self.init(batchOf: batchOf, size: size)
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

extension DataFrame {
  func add(toLoadImage property: ColumnProperty, name: String) {
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
    }, &inputIndex, 1, nil, nil, nil)
    let index = ccv_cnnp_dataframe_read_image(_dataframe, pathIndex, 0, name)
    columnProperties[name] = ColumnProperty(index: Int(index), type: .tensor)
  }
}
