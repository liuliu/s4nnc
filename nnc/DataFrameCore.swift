import C_nnc

// MARK - Sampling

extension DataFrame {
  final class WrappedSampler {
    let property: ColumnProperty
    let size: Int
    let repeating: Int?
    let sample: ([AnyObject]) -> AnyObject
    let outputType: ColumnProperty.PropertyType
    var tensors: [OpaquePointer: _AnyTensor]?
    init(property: ColumnProperty, size: Int, repeating: Int?, sample: @escaping ([AnyObject]) -> AnyObject, outputType: ColumnProperty.PropertyType) {
      self.property = property
      self.size = size
      self.repeating = repeating
      self.sample = sample
      self.outputType = outputType
      switch outputType {
      case .object:
        tensors = nil
      case .tensor:
        tensors = [OpaquePointer: _AnyTensor]()
      }
    }
  }
  // No repeating, or repeating is 1, this is simple case.
  static func sample(dataframe: _DataFrame, property: ColumnProperty, size: Int, sample: @escaping ([AnyObject]) -> AnyObject, outputType: ColumnProperty.PropertyType) -> OpaquePointer {
    let _dataframe = dataframe.dataframe
    let index = Int32(property.index)
    let sampler = WrappedSampler(property: property, size: size, repeating: nil, sample: sample, outputType: outputType)
    return ccv_cnnp_dataframe_sample_new(_dataframe, { input, inputSize, data, context, _ in
      guard let input = input else { return }
      guard let data = data else { return }
      let wrappedSampler = Unmanaged<WrappedSampler>.fromOpaque(context!).takeUnretainedValue()
      var inputObjects = [AnyObject]()
      inputObjects.reserveCapacity(Int(inputSize))
      switch wrappedSampler.property.type {
      case .object:
        for i in 0..<Int(inputSize) {
          inputObjects.append(Unmanaged<AnyObject>.fromOpaque(input[i]!).takeUnretainedValue())
        }
      case .tensor:
        for i in 0..<Int(inputSize) {
          inputObjects.append(_AnyTensor(input[i]!.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false).toAnyTensor() as AnyObject)
        }
      }
      let output = wrappedSampler.sample(inputObjects)
      switch wrappedSampler.outputType {
      case .object:
        if let opaque = data[0] {
          Unmanaged<AnyObject>.fromOpaque(opaque).release()
        }
        data.initialize(to: Unmanaged.passRetained(output).toOpaque())
      case .tensor:
        if let opaque = data[0] {
          wrappedSampler.tensors![OpaquePointer(opaque)] = nil
        }
        let tensor = output as! AnyTensor
        wrappedSampler.tensors![OpaquePointer(tensor.underlying._tensor)] = tensor.underlying
        data.initialize(to: tensor.underlying._tensor)
      }
    }, { object, context in
      guard let object = object else { return }
      let wrappedSampler = Unmanaged<WrappedSampler>.fromOpaque(context!).takeUnretainedValue()
      switch wrappedSampler.outputType {
      case .object:
        Unmanaged<AnyObject>.fromOpaque(object).release()
      case .tensor:
        wrappedSampler.tensors![OpaquePointer(object)] = nil
      }
    }, index, Int32(size), Unmanaged.passRetained(sampler).toOpaque(), { sampler in
      Unmanaged<WrappedSampler>.fromOpaque(sampler!).release()
    })!
  }
  init(dataframe: _DataFrame, property: ColumnProperty, size: Int, sample: @escaping ([AnyObject]) -> AnyObject, outputType: ColumnProperty.PropertyType) {
    let _dataframe = dataframe.dataframe
    let sampled = Self.sample(dataframe: dataframe, property: property, size: size, sample: sample, outputType: outputType)
    // These must have names.
    let name = ccv_cnnp_dataframe_column_name(_dataframe, Int32(property.index))!
    let columnProperties = [String(cString: name): ColumnProperty(index: 0, type: outputType)]
    self.init(dataframe: _DataFrame(dataframe: sampled, parent: dataframe), columnProperties: columnProperties)
  }
  static func sample(dataframe: _DataFrame, property: ColumnProperty, size: Int, repeating: Int, sample: @escaping ([AnyObject]) -> AnyObject, outputType: ColumnProperty.PropertyType) -> OpaquePointer {
    let _dataframe = dataframe.dataframe
    let index = Int32(property.index)
    let sampler = WrappedSampler(property: property, size: size, repeating: repeating, sample: sample, outputType: .object)
    return ccv_cnnp_dataframe_sample_new(_dataframe, { input, inputSize, data, context, _ in
      guard let input = input else { return }
      guard let data = data else { return }
      let wrappedSampler = Unmanaged<WrappedSampler>.fromOpaque(context!).takeUnretainedValue()
      var inputObjects = [AnyObject]()
      let size = wrappedSampler.size
      inputObjects.reserveCapacity(size)
      var outputObjects = [AnyObject]()
      let repeating = wrappedSampler.repeating!
      outputObjects.reserveCapacity(repeating)
      for i in 0..<repeating {
        switch wrappedSampler.property.type {
        case .object:
          for j in 0..<size {
            inputObjects.append(Unmanaged<AnyObject>.fromOpaque(input[(i * size + j) % Int(inputSize)]!).takeUnretainedValue())
          }
        case .tensor:
          for j in 0..<size {
            inputObjects.append(_AnyTensor(input[(i * size + j) % Int(inputSize)]!.assumingMemoryBound(to: ccv_nnc_tensor_t.self), selfOwned: false).toAnyTensor() as AnyObject)
          }
        }
        let output = wrappedSampler.sample(inputObjects)
        outputObjects.append(output)
        inputObjects.removeAll()
      }
      if let opaque = data[0] {
        Unmanaged<AnyObject>.fromOpaque(opaque).release()
      }
      data.initialize(to: Unmanaged.passRetained(outputObjects as AnyObject).toOpaque())
    }, { object, _ in
      guard let object = object else { return }
      Unmanaged<AnyObject>.fromOpaque(object).release()
    }, index, Int32(size * repeating), Unmanaged.passRetained(sampler).toOpaque(), { sampler in
      Unmanaged<WrappedSampler>.fromOpaque(sampler!).release()
    })!
  }
  // Has repeating, need to group the objects out.
  init(dataframe: _DataFrame, property: ColumnProperty, size: Int, repeating: Int, sample: @escaping ([AnyObject]) -> AnyObject, outputType: ColumnProperty.PropertyType) {
    precondition(repeating > 1)
    let sampled = Self.sample(dataframe: dataframe, property: property, size: size, repeating: repeating, sample: sample, outputType: outputType)
    let _dataframe = dataframe.dataframe
    var columnProperties = [String: ColumnProperty]()
    // These must have names.
    let name = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(property.index))!)
    let mainProperty = ColumnProperty(index: 0, type: .object)
    for i in 0..<repeating {
      let indexName = "\(name)_\(i)"
      columnProperties[indexName] = Self.add(to: sampled, map: { ($0 as! [AnyObject])[i] }, property: mainProperty, outputType: outputType, name: indexName)
    }
    self.init(dataframe: _DataFrame(dataframe: sampled, parent: dataframe), columnProperties: columnProperties)
  }
}

public extension DataFrame.UntypedSeries {
  func sample<T, U>(size: Int, repeating: Int? = nil, sampler: @escaping ([T]) -> U) -> DataFrame {
    guard let property = property,
          let dataframe = dataframe else {
      fatalError("An UntypedSeries has to be referenced from existing dataframe. Cannot be a temporary one.")
    }
    let wrappedSampler = { (obj: [AnyObject]) -> AnyObject in
      return sampler(obj as! [T]) as AnyObject
    }
    if let repeating = repeating, repeating > 1 {
      // Use only return type for whether this is a tensor or not.
      if U.self is AnyTensor.Type || U.self == AnyTensor.self {
        // Special handling if this is a tensor, for C-interop, we will unwrap the underlying tensor.
        return DataFrame(dataframe: dataframe, property: property, size: size, repeating: repeating, sample: wrappedSampler, outputType: .tensor)
      } else {
        return DataFrame(dataframe: dataframe, property: property, size: size, repeating: repeating, sample: wrappedSampler, outputType: .object)
      }
    } else {
      // Use only return type for whether this is a tensor or not.
      if U.self is AnyTensor.Type || U.self == AnyTensor.self {
        // Special handling if this is a tensor, for C-interop, we will unwrap the underlying tensor.
        return DataFrame(dataframe: dataframe, property: property, size: size, sample: wrappedSampler, outputType: .tensor)
      } else {
        return DataFrame(dataframe: dataframe, property: property, size: size, sample: wrappedSampler, outputType: .object)
      }
    }
  }
}

public extension DataFrame.TypedSeries {
  func sample<U>(size: Int, repeating: Int? = nil, sampler: @escaping ([Element]) -> U) -> DataFrame {
    let wrappedSampler = { (obj: [AnyObject]) -> AnyObject in
      return sampler(obj as! [Element]) as AnyObject
    }
    if let repeating = repeating, repeating > 1 {
      // Use only return type for whether this is a tensor or not.
      if U.self is AnyTensor.Type || U.self == AnyTensor.self {
        // Special handling if this is a tensor, for C-interop, we will unwrap the underlying tensor.
        return DataFrame(dataframe: dataframe, property: property, size: size, repeating: repeating, sample: wrappedSampler, outputType: .tensor)
      } else {
        return DataFrame(dataframe: dataframe, property: property, size: size, repeating: repeating, sample: wrappedSampler, outputType: .object)
      }
    } else {
      // Use only return type for whether this is a tensor or not.
      if U.self is AnyTensor.Type || U.self == AnyTensor.self {
        // Special handling if this is a tensor, for C-interop, we will unwrap the underlying tensor.
        return DataFrame(dataframe: dataframe, property: property, size: size, sample: wrappedSampler, outputType: .tensor)
      } else {
        return DataFrame(dataframe: dataframe, property: property, size: size, sample: wrappedSampler, outputType: .object)
      }
    }
  }
}

public extension DataFrame.ManyUntypedSeries {
  func sample<C0, C1, U0, U1>(size: Int, repeating: Int? = nil, sampler: @escaping ([(C0, C1)]) -> (U0, U1)) -> DataFrame {
    precondition(properties.count >= 2)
    let wrappedSampler = { (obj: [AnyObject]) -> AnyObject in
      return sampler(obj as! [(C0, C1)]) as AnyObject
    }
    let property = DataFrame.add(to: dataframe.dataframe, multimap: { input in
      return (input[0] as! C0, input[1] as! C1) as AnyObject
    }, properties: properties, outputType: .object, name: "makeTuple")
    let sampled: OpaquePointer
    if let repeating = repeating, repeating > 1 {
      sampled = DataFrame.sample(dataframe: dataframe, property: property, size: size, repeating: repeating, sample: wrappedSampler, outputType: .object)
    } else {
      sampled = DataFrame.sample(dataframe: dataframe, property: property, size: size, sample: wrappedSampler, outputType: .object)
    }
    let _dataframe = dataframe.dataframe
    var columnProperties = [String: DataFrame.ColumnProperty]()
    // These must have names.
    let mainProperty = DataFrame.ColumnProperty(index: 0, type: .object)
    if let repeating = repeating, repeating > 1 {
      for i in 0..<repeating {
        let indexName = "makeTuple_\(i)"
        let extractedProperty = DataFrame.add(to: sampled, map: { ($0 as! [AnyObject])[i] }, property: mainProperty, outputType: .object, name: indexName)
        let name0 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[0].index))!)
        let indexName0 = "\(name0)_\(i)"
        let outputType0: DataFrame.ColumnProperty.PropertyType = (U0.self is AnyTensor.Type || U0.self == AnyTensor.self) ? .tensor: .object
        columnProperties[indexName0] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1)).0 as AnyObject }, property: extractedProperty, outputType: outputType0, name: indexName0)
        let name1 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[1].index))!)
        let indexName1 = "\(name1)_\(i)"
        let outputType1: DataFrame.ColumnProperty.PropertyType = (U1.self is AnyTensor.Type || U1.self == AnyTensor.self) ? .tensor: .object
        columnProperties[indexName1] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1)).1 as AnyObject }, property: extractedProperty, outputType: outputType1, name: indexName1)
      }
    } else {
      let name0 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[0].index))!)
      let outputType0: DataFrame.ColumnProperty.PropertyType = (U0.self is AnyTensor.Type || U0.self == AnyTensor.self) ? .tensor: .object
      columnProperties[name0] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1)).0 as AnyObject }, property: mainProperty, outputType: outputType0, name: name0)
      let name1 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[1].index))!)
      let outputType1: DataFrame.ColumnProperty.PropertyType = (U1.self is AnyTensor.Type || U1.self == AnyTensor.self) ? .tensor: .object
      columnProperties[name1] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1)).1 as AnyObject }, property: mainProperty, outputType: outputType1, name: name1)
    }
    return DataFrame(dataframe: _DataFrame(dataframe: sampled, parent: dataframe), columnProperties: columnProperties)
  }

  func sample<C0, C1, C2, U0, U1, U2>(size: Int, repeating: Int? = nil, sampler: @escaping ([(C0, C1, C2)]) -> (U0, U1, U2)) -> DataFrame {
    precondition(properties.count >= 3)
    let wrappedSampler = { (obj: [AnyObject]) -> AnyObject in
      return sampler(obj as! [(C0, C1, C2)]) as AnyObject
    }
    let property = DataFrame.add(to: dataframe.dataframe, multimap: { input in
      return (input[0] as! C0, input[1] as! C1, input[2] as! C2) as AnyObject
    }, properties: properties, outputType: .object, name: "makeTuple")
    let sampled: OpaquePointer
    if let repeating = repeating, repeating > 1 {
      sampled = DataFrame.sample(dataframe: dataframe, property: property, size: size, repeating: repeating, sample: wrappedSampler, outputType: .object)
    } else {
      sampled = DataFrame.sample(dataframe: dataframe, property: property, size: size, sample: wrappedSampler, outputType: .object)
    }
    let _dataframe = dataframe.dataframe
    var columnProperties = [String: DataFrame.ColumnProperty]()
    // These must have names.
    let mainProperty = DataFrame.ColumnProperty(index: 0, type: .object)
    if let repeating = repeating, repeating > 1 {
      for i in 0..<repeating {
        let indexName = "makeTuple_\(i)"
        let extractedProperty = DataFrame.add(to: sampled, map: { ($0 as! [AnyObject])[i] }, property: mainProperty, outputType: .object, name: indexName)
        let name0 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[0].index))!)
        let indexName0 = "\(name0)_\(i)"
        let outputType0: DataFrame.ColumnProperty.PropertyType = (U0.self is AnyTensor.Type || U0.self == AnyTensor.self) ? .tensor: .object
        columnProperties[indexName0] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2)).0 as AnyObject }, property: extractedProperty, outputType: outputType0, name: indexName0)
        let name1 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[1].index))!)
        let indexName1 = "\(name1)_\(i)"
        let outputType1: DataFrame.ColumnProperty.PropertyType = (U1.self is AnyTensor.Type || U1.self == AnyTensor.self) ? .tensor: .object
        columnProperties[indexName1] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2)).1 as AnyObject }, property: extractedProperty, outputType: outputType1, name: indexName1)
        let name2 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[2].index))!)
        let indexName2 = "\(name2)_\(i)"
        let outputType2: DataFrame.ColumnProperty.PropertyType = (U2.self is AnyTensor.Type || U2.self == AnyTensor.self) ? .tensor: .object
        columnProperties[indexName2] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2)).2 as AnyObject }, property: extractedProperty, outputType: outputType2, name: indexName2)
      }
    } else {
      let name0 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[0].index))!)
      let outputType0: DataFrame.ColumnProperty.PropertyType = (U0.self is AnyTensor.Type || U0.self == AnyTensor.self) ? .tensor: .object
      columnProperties[name0] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2)).0 as AnyObject }, property: mainProperty, outputType: outputType0, name: name0)
      let name1 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[1].index))!)
      let outputType1: DataFrame.ColumnProperty.PropertyType = (U1.self is AnyTensor.Type || U1.self == AnyTensor.self) ? .tensor: .object
      columnProperties[name1] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2)).1 as AnyObject }, property: mainProperty, outputType: outputType1, name: name1)
      let name2 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[2].index))!)
      let outputType2: DataFrame.ColumnProperty.PropertyType = (U2.self is AnyTensor.Type || U2.self == AnyTensor.self) ? .tensor: .object
      columnProperties[name2] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2)).2 as AnyObject }, property: mainProperty, outputType: outputType2, name: name2)
    }
    return DataFrame(dataframe: _DataFrame(dataframe: sampled, parent: dataframe), columnProperties: columnProperties)
  }

  func sample<C0, C1, C2, C3, U0, U1, U2, U3>(size: Int, repeating: Int? = nil, sampler: @escaping ([(C0, C1, C2, C3)]) -> (U0, U1, U2, U3)) -> DataFrame {
    precondition(properties.count >= 4)
    let wrappedSampler = { (obj: [AnyObject]) -> AnyObject in
      return sampler(obj as! [(C0, C1, C2, C3)]) as AnyObject
    }
    let property = DataFrame.add(to: dataframe.dataframe, multimap: { input in
      return (input[0] as! C0, input[1] as! C1, input[2] as! C2, input[3] as! C3) as AnyObject
    }, properties: properties, outputType: .object, name: "makeTuple")
    let sampled: OpaquePointer
    if let repeating = repeating, repeating > 1 {
      sampled = DataFrame.sample(dataframe: dataframe, property: property, size: size, repeating: repeating, sample: wrappedSampler, outputType: .object)
    } else {
      sampled = DataFrame.sample(dataframe: dataframe, property: property, size: size, sample: wrappedSampler, outputType: .object)
    }
    let _dataframe = dataframe.dataframe
    var columnProperties = [String: DataFrame.ColumnProperty]()
    // These must have names.
    let mainProperty = DataFrame.ColumnProperty(index: 0, type: .object)
    if let repeating = repeating, repeating > 1 {
      for i in 0..<repeating {
        let indexName = "makeTuple_\(i)"
        let extractedProperty = DataFrame.add(to: sampled, map: { ($0 as! [AnyObject])[i] }, property: mainProperty, outputType: .object, name: indexName)
        let name0 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[0].index))!)
        let indexName0 = "\(name0)_\(i)"
        let outputType0: DataFrame.ColumnProperty.PropertyType = (U0.self is AnyTensor.Type || U0.self == AnyTensor.self) ? .tensor: .object
        columnProperties[indexName0] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2, U3)).0 as AnyObject }, property: extractedProperty, outputType: outputType0, name: indexName0)
        let name1 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[1].index))!)
        let indexName1 = "\(name1)_\(i)"
        let outputType1: DataFrame.ColumnProperty.PropertyType = (U1.self is AnyTensor.Type || U1.self == AnyTensor.self) ? .tensor: .object
        columnProperties[indexName1] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2, U3)).1 as AnyObject }, property: extractedProperty, outputType: outputType1, name: indexName1)
        let name2 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[2].index))!)
        let indexName2 = "\(name2)_\(i)"
        let outputType2: DataFrame.ColumnProperty.PropertyType = (U2.self is AnyTensor.Type || U2.self == AnyTensor.self) ? .tensor: .object
        columnProperties[indexName2] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2, U3)).2 as AnyObject }, property: extractedProperty, outputType: outputType2, name: indexName2)
        let name3 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[3].index))!)
        let indexName3 = "\(name3)_\(i)"
        let outputType3: DataFrame.ColumnProperty.PropertyType = (U3.self is AnyTensor.Type || U3.self == AnyTensor.self) ? .tensor: .object
        columnProperties[indexName3] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2, U3)).3 as AnyObject }, property: extractedProperty, outputType: outputType3, name: indexName3)
      }
    } else {
      let name0 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[0].index))!)
      let outputType0: DataFrame.ColumnProperty.PropertyType = (U0.self is AnyTensor.Type || U0.self == AnyTensor.self) ? .tensor: .object
      columnProperties[name0] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2, U3)).0 as AnyObject }, property: mainProperty, outputType: outputType0, name: name0)
      let name1 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[1].index))!)
      let outputType1: DataFrame.ColumnProperty.PropertyType = (U1.self is AnyTensor.Type || U1.self == AnyTensor.self) ? .tensor: .object
      columnProperties[name1] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2, U3)).1 as AnyObject }, property: mainProperty, outputType: outputType1, name: name1)
      let name2 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[2].index))!)
      let outputType2: DataFrame.ColumnProperty.PropertyType = (U2.self is AnyTensor.Type || U2.self == AnyTensor.self) ? .tensor: .object
      columnProperties[name2] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2, U3)).2 as AnyObject }, property: mainProperty, outputType: outputType2, name: name2)
      let name3 = String(cString: ccv_cnnp_dataframe_column_name(_dataframe, Int32(properties[3].index))!)
      let outputType3: DataFrame.ColumnProperty.PropertyType = (U3.self is AnyTensor.Type || U3.self == AnyTensor.self) ? .tensor: .object
      columnProperties[name3] = DataFrame.add(to: sampled, map: { ($0 as! (U0, U1, U2, U3)).3 as AnyObject }, property: mainProperty, outputType: outputType3, name: name3)
    }
    return DataFrame(dataframe: _DataFrame(dataframe: sampled, parent: dataframe), columnProperties: columnProperties)
  }
}
