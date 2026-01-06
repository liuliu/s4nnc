#if canImport(C_nnc)
import C_nnc
#elseif canImport(C_swiftpm_nnc)
import C_swiftpm_nnc
#endif

/// A type-erased model builder.
public class AnyModelBuilder: AnyModel {

  public var testing: Bool {
    get { model!.testing }
    set { model!.testing = newValue }
  }

  @usableFromInline
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

  fileprivate var _outputSize: Int? = nil
  var outputSize: Int {
    if let outputSize = _outputSize {
      return outputSize
    }
    // Compile explicitly.
    compileModel(isEager: false)
    // After the model compiled, we can access the outputSize now.
    let outputSize = Int(ccv_cnnp_model_output_size(model!.cModel))
    _outputSize = outputSize
    return outputSize
  }

  /**
   * Whether to enable memory reduction for this model. The current supported memory reduction
   * technique is to redo datatype conversion during backward pass if needed.
   */
  public var memoryReduction: Bool = false {
    didSet {
      ccv_cnnp_model_set_memory_reduction(model!.cModel, memoryReduction ? 1 : 0)
    }
  }

  /**
   * Specify the maximum number of streams we need to allocate to run this model.
   */
  public var maxConcurrency: StreamContext.Concurrency = .noLimit {
    didSet {
      ccv_cnnp_model_set_max_concurrency(model!.cModel, Int32(maxConcurrency.rawValue))
    }
  }

  /**
   * Abstract representation of the stateful components from the model builder.
   */
  public var parameters: Model.Parameters {
    model!.parameters
  }

  /**
   * Shortcut for weight parameter.
   */
  public var weight: Model.Parameters {
    parameters(for: .weight)
  }

  /**
   * Shortcut for bias parameter.
   */
  public var bias: Model.Parameters {
    parameters(for: .bias)
  }

  /**
   * The size of scratch memory allocated for this model.
   */
  public var runtimeMemorySize: UInt64 {
    return model!.runtimeMemorySize
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

  /**
   * Cancel current evaluation of this model. It only cancels the model that you know is currently
   * in evaluation, if you didn't get the execution order right, it won't have effect (you need
   * to make sure this method, if it is called, is strictly after call to callAsFunction and before
   * it returns).
   */
  public func cancel() {
    model?.cancel()
  }

  private var _store: DynamicGraph._Store? = nil
  private var _key: String? = nil
  private var _reader:
    ((String, DataType, TensorFormat, TensorShape) -> DynamicGraph.Store.ModelReaderResult)? = nil

  @usableFromInline
  func read(
    _ key: String, from store: DynamicGraph._Store, strict: Bool, codec: DynamicGraph.Store.Codec,
    reader: ((String, DataType, TensorFormat, TensorShape) -> DynamicGraph.Store.ModelReaderResult)?
  ) throws {
    // If the model is compiled (signifies by _outputSize is set)
    if _outputSize != nil {
      guard let reader = reader else {
        if codec.isEmpty {
          ccv_cnnp_model_read(store.sqlite, key, nil, model!.cModel)
        } else {
          var option = ccv_nnc_tensor_io_option_t()
          option.decode = codec.decode
          option.context = Unmanaged<DynamicGraph._Store>.passUnretained(store).toOpaque()
          ccv_cnnp_model_read(store.sqlite, key, &option, model!.cModel)
        }
        if strict, let _io = ccv_cnnp_model_parameter_first_uninit(model!.cModel) {
          throw DynamicGraph.Store.ModelReadError.missing(
            String(
              cString: ccv_cnnp_model_parameter_name(model!.cModel, _io)))
        }
        return
      }
      let readerHelper = DynamicGraph.Store.ModelReaderHelper(reader: reader, sqlite: store.sqlite)
      ccv_cnnp_model_set_io(
        model!.cModel,
        { (handle, name, options, params, tensorOut) -> Int32 in
          let readerHelper = Unmanaged<DynamicGraph.Store.ModelReaderHelper>.fromOpaque(handle!)
            .takeUnretainedValue()
          let params = tensorOut!.pointee?.pointee.info ?? params
          let result = readerHelper.reader(
            name.map { String(cString: $0) } ?? "", DataType.from(cTensorParams: params),
            TensorFormat.from(cTensorParams: params), TensorShape(dims: params.dim))
          switch result {
          case .final(let tensor):
            precondition(tensor.kind == .CPU)
            if tensorOut!.pointee == nil {
              tensorOut!.pointee = ccv_nnc_tensor_new(nil, params, 0)
            }
            var input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = tensor.cTensor
            ccv_nnc_cmd_exec(
              ccv_nnc_cmd(
                CCV_NNC_DATA_TRANSFER_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0),
              ccv_nnc_no_hint, 0, &input, 1, tensorOut, 1, nil)
            return Int32(CCV_IO_FINAL)
          case let .continue(name, codec, store):
            var params = params
            let sqlite = store?.store.sqlite ?? readerHelper.sqlite
            guard var options = options?.pointee else {
              return ccv_nnc_tensor_read(
                sqlite, name, nil, 0, &params, tensorOut)
            }
            if let store = store?.store {
              options.context = Unmanaged<DynamicGraph._Store>.passUnretained(store).toOpaque()
            }
            guard let codec = codec else {
              return ccv_nnc_tensor_read(
                sqlite, name, &options, 0, &params, tensorOut)
            }
            options.decode = codec.decode
            return ccv_nnc_tensor_read(
              sqlite, name, &options, 0, &params, tensorOut)
          case .fail:
            return Int32(CCV_IO_ERROR)
          }
        }, nil)
      let unmanaged = Unmanaged.passRetained(readerHelper)
      if codec.isEmpty {
        ccv_cnnp_model_read(unmanaged.toOpaque(), key, nil, model!.cModel)
      } else {
        var option = ccv_nnc_tensor_io_option_t()
        option.decode = codec.decode
        option.context = Unmanaged<DynamicGraph._Store>.passUnretained(store).toOpaque()
        ccv_cnnp_model_read(unmanaged.toOpaque(), key, &option, model!.cModel)
      }
      ccv_cnnp_model_set_io(model!.cModel, nil, nil)
      unmanaged.release()
      if strict, let _io = ccv_cnnp_model_parameter_first_uninit(model!.cModel) {
        throw DynamicGraph.Store.ModelReadError.missing(
          String(
            cString: ccv_cnnp_model_parameter_name(model!.cModel, _io)))
      }
    }
    _reader = reader
    _store = store
    _key = key
  }

  fileprivate func compileModel(isEager: Bool) {
    let inputs = self.inputs!
    model!.compile(inputs: inputs, isEager: isEager)
    // If we have store / key, try to load parameters now after it is compiled.
    if let store = _store, let key = _key {
      if let reader = _reader {
        let readerHelper = DynamicGraph.Store.ModelReaderHelper(
          reader: reader, sqlite: store.sqlite)
        ccv_cnnp_model_set_io(
          model!.cModel,
          { (handle, name, options, params, tensorOut) -> Int32 in
            let readerHelper = Unmanaged<DynamicGraph.Store.ModelReaderHelper>.fromOpaque(handle!)
              .takeUnretainedValue()
            let params = tensorOut!.pointee?.pointee.info ?? params
            let result = readerHelper.reader(
              name.map { String(cString: $0) } ?? "", DataType.from(cTensorParams: params),
              TensorFormat.from(cTensorParams: params), TensorShape(dims: params.dim))
            switch result {
            case .final(let tensor):
              precondition(tensor.kind == .CPU)
              if tensorOut!.pointee == nil {
                tensorOut!.pointee = ccv_nnc_tensor_new(nil, params, 0)
              }
              var input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = tensor.cTensor
              ccv_nnc_cmd_exec(
                ccv_nnc_cmd(
                  CCV_NNC_DATA_TRANSFER_FORWARD, nil, CmdParamsFactory.factory.newParams(), 0),
                ccv_nnc_no_hint, 0, &input, 1, tensorOut, 1, nil)
              return Int32(CCV_IO_FINAL)
            case let .continue(name, codec, store):
              var params = params
              let sqlite = store?.store.sqlite ?? readerHelper.sqlite
              guard var options = options?.pointee else {
                return ccv_nnc_tensor_read(
                  sqlite, name, nil, 0, &params, tensorOut)
              }
              if let store = store?.store {
                options.context = Unmanaged<DynamicGraph._Store>.passUnretained(store).toOpaque()
              }
              guard let codec = codec else {
                return ccv_nnc_tensor_read(
                  sqlite, name, &options, 0, &params, tensorOut)
              }
              options.decode = codec.decode
              return ccv_nnc_tensor_read(
                sqlite, name, &options, 0, &params, tensorOut)
            case .fail:
              return Int32(CCV_IO_ERROR)
            }
          }, nil)
        let unmanaged = Unmanaged.passRetained(readerHelper)
        ccv_cnnp_model_read(unmanaged.toOpaque(), key, nil, model!.cModel)
        ccv_cnnp_model_set_io(model!.cModel, nil, nil)
        unmanaged.release()
      } else {
        ccv_cnnp_model_read(store.sqlite, key, nil, model!.cModel)
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
  public func compile(_ t: T, inputs: [DynamicGraph_Any], isEager: Bool = false) {
    self.t = t
    self.inputs = inputs
    compileModel(isEager: isEager)
    _outputSize = Int(ccv_cnnp_model_output_size(model!.cModel))
    self.inputs = nil
    self.t = nil
  }

  /**
   * Compile a model with the given inputs without executing it. After this, you can load
   * parameters from the store.
   */
  public func compile(_ t: T, inputs: DynamicGraph_Any..., isEager: Bool = false) {
    compile(t, inputs: inputs, isEager: isEager)
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
  public func compile(inputs: [DynamicGraph_Any], isEager: Bool = false) {
    self.t = Void()
    self.inputs = inputs
    compileModel(isEager: isEager)
    _outputSize = Int(ccv_cnnp_model_output_size(model!.cModel))
    self.inputs = nil
    self.t = nil
  }

  /**
   * Compile a model with the given inputs without executing it. After this, you can load
   * parameters from the store.
   */
  public func compile(inputs: DynamicGraph_Any..., isEager: Bool = false) {
    compile(inputs: inputs, isEager: isEager)
  }
}
