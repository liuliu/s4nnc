import C_nnc

/// A type-erased model builder.
public class AnyModelBuilder {

  public var isTest: Bool = false

  var model: Model? = nil
  var t: Any? = nil
  var inputs: [DynamicGraph_Any]? = nil
  private let builder: (_: Any, _: [DynamicGraph_Any]) -> Model

  fileprivate init(builder: @escaping (_: Any, _: [DynamicGraph_Any]) -> Model, name: String = "") {
    self.builder = builder
    let _model = ccv_cnnp_dynamic_new(
      { _, _, ctx in
        let modelBuilder = Unmanaged<AnyModelBuilder>.fromOpaque(ctx!).takeUnretainedValue()
        let t = modelBuilder.t!
        let inputs = modelBuilder.inputs!
        let builder = modelBuilder.builder
        let model = builder(t, inputs)
        return model.obtainUnderlyingModel(modelBuilder.model!)
      }, Unmanaged.passUnretained(self).toOpaque(), name)!
    model = Model(_model)
  }

  private var _outputSize: Int? = nil
  var outputSize: Int {
    if let outputSize = _outputSize {
      return outputSize
    }
    // Compile explicitly.
    compileModel()
    // After the model compiled, we can access the outputSize now.
    let outputSize = Int(ccv_cnnp_model_output_size(model!.cModel))
    _outputSize = outputSize
    return outputSize
  }

  /**
   * Abstract representation of the stateful components from the model builder.
   */
  public var parameters: Model.Parameters {
    model!.parameters
  }

  /**
   * Broadly speaking, you can have two types of parameters, weight and bias.
   * You can get them in abstract fashion with this method.
   *
   * - Parameter type: Whether it is weight or bias.
   * - Returns: An abstract representation of parameters.
   */
  public func parameters(for type: Model.ParametersType) -> Model.Parameters {
    return model!.parameters(for: type)
  }

  private var _store: DynamicGraph._Store? = nil
  private var _key: String? = nil
  private var _reader:
    ((String, DataType, TensorFormat, TensorShape) -> DynamicGraph.Store.ModelReaderResult)? = nil

  func read(
    _ key: String, from store: DynamicGraph._Store,
    reader: ((String, DataType, TensorFormat, TensorShape) -> DynamicGraph.Store.ModelReaderResult)?
  ) {
    // If the model is compiled (signifies by _outputSize is set)
    if _outputSize != nil {
      guard let reader = reader else {
        ccv_cnnp_model_read(store.sqlite, key, model!.cModel)
        return
      }
      let readerHelper = DynamicGraph.Store.ModelReaderHelper(reader: reader, sqlite: store.sqlite)
      ccv_cnnp_model_set_io(
        model!.cModel,
        { (handle, name, dir, tensorOut) -> Int32 in
          let readerHelper = Unmanaged<DynamicGraph.Store.ModelReaderHelper>.fromOpaque(handle!)
            .takeUnretainedValue()
          let cTensorOut = tensorOut!.pointee
          let params = cTensorOut!.pointee.info
          let result = readerHelper.reader(
            name.map { String(cString: $0) } ?? "", DataType.from(cTensorParams: params),
            TensorFormat.from(cTensorParams: params), TensorShape(dims: params.dim))
          switch result {
          case .final(let tensor):
            let cTensor = tensor.cTensor
            let dataSize = ccv_nnc_tensor_data_size(cTensor.pointee.info)
            ccv_nnc_tensor_swap(cTensorOut, name, dir, cTensor.pointee.data.ptr, dataSize)
            return Int32(CCV_IO_FINAL)
          case .continue(let name):
            return ccv_nnc_tensor_read(readerHelper.sqlite, name, dir, tensorOut)
          case .fail:
            return Int32(CCV_IO_ERROR)
          }
        }, nil)
      let unmanaged = Unmanaged.passRetained(readerHelper)
      ccv_cnnp_model_read(unmanaged.toOpaque(), key, model!.cModel)
      ccv_cnnp_model_set_io(model!.cModel, nil, nil)
      unmanaged.release()
    }
    _store = store
    _key = key
  }

  fileprivate func compileModel() {
    let inputs = self.inputs!
    model!.compile(inputs: inputs)
    // If we have store / key, try to load parameters now after it is compiled.
    if let store = _store, let key = _key {
      if let reader = _reader {
        let readerHelper = DynamicGraph.Store.ModelReaderHelper(
          reader: reader, sqlite: store.sqlite)
        ccv_cnnp_model_set_io(
          model!.cModel,
          { (handle, name, dir, tensorOut) -> Int32 in
            let readerHelper = Unmanaged<DynamicGraph.Store.ModelReaderHelper>.fromOpaque(handle!)
              .takeUnretainedValue()
            let cTensorOut = tensorOut!.pointee
            let params = cTensorOut!.pointee.info
            let result = readerHelper.reader(
              name.map { String(cString: $0) } ?? "", DataType.from(cTensorParams: params),
              TensorFormat.from(cTensorParams: params), TensorShape(dims: params.dim))
            switch result {
            case .final(let tensor):
              let cTensor = tensor.cTensor
              let dataSize = ccv_nnc_tensor_data_size(cTensor.pointee.info)
              ccv_nnc_tensor_swap(cTensorOut, name, dir, cTensor.pointee.data.ptr, dataSize)
              return Int32(CCV_IO_FINAL)
            case .continue(let name):
              return ccv_nnc_tensor_read(readerHelper.sqlite, name, dir, tensorOut)
            case .fail:
              return Int32(CCV_IO_ERROR)
            }
          }, nil)
        let unmanaged = Unmanaged.passRetained(readerHelper)
        ccv_cnnp_model_read(unmanaged.toOpaque(), key, model!.cModel)
        ccv_cnnp_model_set_io(model!.cModel, nil, nil)
        unmanaged.release()
      } else {
        ccv_cnnp_model_read(store.sqlite, key, model!.cModel)
      }
      _reader = nil
      _store = nil
      _key = nil
    }
  }

}

/// A model builder is a more generic type of model. A model can be quite static,
/// thus, you have to be quite careful to have a model work with dynamic inputs.
/// You cannot use reshape, or anything that can generate fixed tensor outputs from
/// a fixed inputs.
///
/// A model builder on the other hand doesn't have that restriction. When input changes,
/// it simply calls the given builder closure to construct a new model. In such way,
/// you can continue to use reshape etc to assume fixed inputs and outputs, it will just
/// work for dynamic inputs. The newly built model will carry over stateful components
/// (parameters) from the old models, thus, it doesn't reset your training. This also means
/// you need to make sure parameter shape won't change when input changes, otherwise we
/// will fatal.
public final class ModelBuilder<T>: AnyModelBuilder {
  public init(_ builder: @escaping (_: T, _: [DynamicGraph_Any]) -> Model, name: String = "") {
    super.init(
      builder: { t, inputs in
        return builder(t as! T, inputs)
      }, name: name)
  }

  /**
   * Compile a model with the given inputs without executing it. After this, you can load
   * parameters from the store.
   */
  public func compile(_ t: T, inputs: [DynamicGraph_Any]) {
    self.t = t
    self.inputs = inputs
    compileModel()
    self.inputs = nil
    self.t = nil
  }

  /**
   * Compile a model with the given inputs without executing it. After this, you can load
   * parameters from the store.
   */
  public func compile(_ t: T, inputs: DynamicGraph_Any...) {
    compile(t, inputs: inputs)
  }
}

extension ModelBuilder where T == Void {
  public convenience init(_ builder: @escaping (_: [DynamicGraph_Any]) -> Model, name: String = "")
  {
    self.init(
      { t, inputs in
        return builder(inputs)
      }, name: name)
  }

  /**
   * Compile a model with the given inputs without executing it. After this, you can load
   * parameters from the store.
   */
  public func compile(inputs: [DynamicGraph_Any]) {
    self.t = Void()
    self.inputs = inputs
    compileModel()
    self.inputs = nil
    self.t = nil
  }

  /**
   * Compile a model with the given inputs without executing it. After this, you can load
   * parameters from the store.
   */
  public func compile(inputs: DynamicGraph_Any...) {
    compile(inputs: inputs)
  }
}
