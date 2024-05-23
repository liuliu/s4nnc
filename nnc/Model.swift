import C_nnc

public protocol ModelIOConvertible {
  var io: Model.IO { get }
}

/// A model is a base class for stateful operations on a dynamic graph. It can be
/// use to construct computations statically, thus, more efficient.
public class Model {

  /**
   * A IO class represent the abstract input / output for a model. It can correspond
   * to one or more tensors when the model is materialized.
   */
  public class IO: ModelIOConvertible {
    public var io: IO { self }
    @usableFromInline
    let _io: ccv_cnnp_model_io_t
    let model: Model?
    private let inputs: [IO]?

    @usableFromInline
    init(_ io: ccv_cnnp_model_io_t, model: Model? = nil, inputs: [IO]? = nil) {
      _io = io
      self.model = model
      self.inputs = inputs
    }
  }

  /**
   * Whether the existing model is for testing or training.
   */
  public var testing: Bool = false

  var dataParallel: Int? = nil  // Keep track of whether we applied data parallel to the model or not.
  public let cModel: OpaquePointer
  var owner: Model? = nil
  weak var graph: DynamicGraph? = nil
  var originals: [Any] = []  // Keep some other models alive if it enabled parameter sharing.

  private func ownerHook() {
    ccv_cnnp_model_notify_hook(
      cModel,
      { _, _, payload, ctx in
        guard payload != nil && payload != ctx else { return }
        let model = Unmanaged<Model>.fromOpaque(ctx!).takeUnretainedValue()
        model.owner = Unmanaged<Model>.fromOpaque(payload!).takeUnretainedValue()
      }, Unmanaged.passUnretained(self).toOpaque())
  }

  required init(_ model: OpaquePointer) {
    cModel = model
    ccv_cnnp_model_notify(cModel, 0, Unmanaged.passUnretained(self).toOpaque())
    ownerHook()
  }

  func obtainUnderlyingModel(_ owner: Model) -> OpaquePointer {
    ccv_cnnp_model_notify(cModel, 0, Unmanaged.passUnretained(owner).toOpaque())
    // self.owner = owner is not necessary because we will update in the callback.
    assert(self.owner === owner)
    return cModel
  }

  @inlinable
  func apply(_ inputs: [ModelIOConvertible]) -> IO {
    let inputIOs = inputs.map { $0.io }
    let _inputs: [ccv_cnnp_model_io_t?] = inputIOs.map { $0._io }
    let _io = ccv_cnnp_model_apply(cModel, _inputs, Int32(inputs.count))!
    return IO(_io, model: self, inputs: inputIOs)
  }

  @inlinable
  public func callAsFunction(_ inputs: ModelIOConvertible...) -> IO {
    return apply(inputs)
  }

  @inlinable
  public func callAsFunction(_ inputs: [ModelIOConvertible]) -> IO {
    return apply(inputs)
  }

  deinit {
    if owner == nil {
      ccv_cnnp_model_free(cModel)
      return
    }
    // Unhook because I am no longer active (but the model can still be available).
    ccv_cnnp_model_notify_hook(cModel, nil, nil)
  }

  public final class Parameters: IO {
    /**
     * The internal name for this parameter.
     */
    public var name: String {
      return String(
        cString: ccv_cnnp_model_parameter_name(model?.owner?.cModel ?? model?.cModel, _io))
    }

    /**
     * Get index into the parameters.
     */
    public subscript(position: Int) -> Parameters {
      assert(_io == model!._parameters)
      return model!.parameters(for: .index(position))
    }

    /**
     * Get the total number of parameters.
     */
    public var count: Int {
      guard _io == model!._parameters else {
        return 1
      }
      return Int(ccv_cnnp_model_parameter_count(model!.cModel))
    }

    /**
     * Get a parameter by loop over internally to find the matching one.
     */
    public func first(where block: @escaping (String) -> Bool) -> Parameters? {
      let wrapped = Wrapped(block)
      let unmanaged = Unmanaged.passRetained(wrapped)
      guard let io = ccv_cnnp_model_parameter_first(model!.cModel, { _, name, context in
        let block = Unmanaged<Wrapped<(String) -> Bool>>.fromOpaque(context!).takeUnretainedValue()
        return block.value(name.flatMap({ String(cString: $0) }) ?? "") ? 1 : 0
      }, unmanaged.toOpaque()) else {
        unmanaged.release()
        return nil
      }
      unmanaged.release()
      return Parameters(io, model: model)
    }

    /**
     * Get a list of parameters matching the condition.
     */
    public func filter(where block: @escaping (String) -> Bool) -> [Parameters] {
      let wrapped = Wrapped(block)
      let unmanaged = Unmanaged.passRetained(wrapped)
      let array = ccv_cnnp_model_parameters_filter(model!.cModel, { _, name, context in
        let block = Unmanaged<Wrapped<(String) -> Bool>>.fromOpaque(context!).takeUnretainedValue()
        return block.value(name.flatMap({ String(cString: $0) }) ?? "") ? 1 : 0
      }, unmanaged.toOpaque())!
      unmanaged.release()
      guard array.pointee.rnum > 0 else {
        ccv_array_free(array)
        return []
      }
      var parameters = [Parameters]()
      let io = array.pointee.data.assumingMemoryBound(to: ccv_cnnp_model_io_t.self)
      for i in 0..<Int(array.pointee.rnum) {
        parameters.append(Parameters(io[i], model: model))
      }
      ccv_array_free(array)
      return parameters
    }
  }

  var _parameters: ccv_cnnp_model_io_t? = nil

  /**
   * Abstract representation of the stateful components from the model.
   */
  public var parameters: Parameters {
    guard let _parameters = _parameters else {
      let parameters = ccv_cnnp_model_parameters(cModel, -1, -1)!
      self._parameters = parameters
      return Parameters(parameters, model: self)
    }
    return Parameters(_parameters, model: self)
  }

  /**
   * Shortcut for weight parameter.
   */
  public var weight: Parameters {
    parameters(for: .weight)
  }

  /**
   * Shortcut for bias parameter.
   */
  public var bias: Parameters {
    parameters(for: .bias)
  }

  /**
   * Whether this is initialized as trainable model or not.
   */
  public var trainable: Bool? {
    let trainable = ccv_cnnp_model_is_trainable(cModel)
    return trainable >= 0 ? trainable != 0 : nil
  }

  /**
   * Whether to enable gradient checkpointing for this model. Once it is enabled, we will re-run
   * the model forward pass again during backward pass. This is effective at reducing memory usage.
   */
  public var gradientCheckpointing: Bool {
    get {
      ccv_cnnp_model_gradient_checkpointing(cModel) != 0
    }
    set {
      ccv_cnnp_model_set_gradient_checkpointing(cModel, newValue ? 1 : 0)
    }
  }

  /**
   * Whether to enable memory reduction for this model. The current supported memory reduction
   * technique is to redo datatype conversion during backward pass if needed.
   */
  public var memoryReduction: Bool = false {
    didSet {
      ccv_cnnp_model_set_memory_reduction(cModel, memoryReduction ? 1 : 0)
    }
  }

  /**
   * Specify the maximum number of streams we need to allocate to run this model.
   */
  public var maxConcurrency: StreamContext.Concurrency = .noLimit {
    didSet {
      ccv_cnnp_model_set_max_concurrency(cModel, Int32(maxConcurrency.rawValue))
    }
  }

  public enum ParametersType {
    case weight
    case bias
    case index(Int)
  }

  private var _biasParameters: ccv_cnnp_model_io_t? = nil
  private var _weightParameters: ccv_cnnp_model_io_t? = nil

  /**
   * Broadly speaking, you can have two types of parameters, weight and bias.
   * You can get them in abstract fashion with this method.
   *
   * - Parameter type: Whether it is weight or bias.
   * - Returns: An abstract representation of parameters.
   */
  public func parameters(for type: ParametersType) -> Parameters {
    switch type {
    case .weight:
      guard let _weightParameters = _weightParameters else {
        let weightParameters = ccv_cnnp_model_parameters(
          cModel, Int32(CCV_CNNP_PARAMETER_SELECT_WEIGHT), -1)!
        self._weightParameters = weightParameters
        return Parameters(weightParameters, model: self)
      }
      return Parameters(_weightParameters, model: self)
    case .bias:
      guard let _biasParameters = _biasParameters else {
        let biasParameters = ccv_cnnp_model_parameters(
          cModel, Int32(CCV_CNNP_PARAMETER_SELECT_BIAS), -1)!
        self._biasParameters = biasParameters
        return Parameters(biasParameters, model: self)
      }
      return Parameters(_biasParameters, model: self)
    case .index(let idx):
      let indexParameters = ccv_cnnp_model_parameters(
        cModel, -1, Int32(idx))!
      return Parameters(indexParameters, model: self)
    }
  }
}

extension Model {
  /**
   * Make a copy of the model. This won't copy over the parameters. If you want, you need to copy
   * parameters over explicitly.
   */
  public func copied() -> Self {
    let newModel = Self(
      ccv_cnnp_model_copy(cModel, trainable == true ? 1 : (trainable == false ? 0 : -1)))
    return newModel
  }
}

extension Model {
  /**
   * Compile a model with the given inputs without executing it. After this, you can load
   * parameters from the store.
   * - isEager: Whether we want to push compilation as far as possible up until the actual execution.
   */
  public func compile(inputs: [DynamicGraph_Any], isEager: Bool = false) {
    assert(inputs.count > 0)
    let params = CmdParamsFactory.factory.newParams()
    let noop = ccv_nnc_cmd(CCV_NNC_NOOP, nil, params, 0)
    let parallel = inputs[0].untyped.count
    if let dataParallel = dataParallel {
      // You cannot run a model previously parallel and then not.
      assert(dataParallel == parallel)
    } else if parallel > 1 {
      ccv_cnnp_model_set_data_parallel(cModel, Int32(parallel))
    }
    let inputParams: [ccv_nnc_tensor_param_t] = inputs.map {
      let tensor = $0.untyped[0]
      return ccv_nnc_tensor_variable_params(tensor.graph.cGraph, tensor._tensor)
    }
    ccv_cnnp_model_compile(cModel, inputParams, Int32(inputParams.count), noop, noop)
    if isEager {
      let graph = inputs[0].graph
      let _inputs: [ccv_nnc_tensor_variable_t?] = inputs.map { $0.untyped[0]._tensor }
      let _streamContext = graph.streamContext?._stream
      ccv_nnc_dynamic_graph_dry_run(
        graph.cGraph, cModel, testing ? 1 : 0, _inputs, Int32(_inputs.count), _streamContext)
    }
  }
  /**
   * Compile a model with the given inputs without executing it. After this, you can load
   * parameters from the store.
   */
  public func compile(inputs: DynamicGraph_Any..., isEager: Bool = false) {
    compile(inputs: inputs, isEager: isEager)
  }
}

/// MARK - Functional and Sequential Models

extension Model {

  /**
   * You can compose a new model from old models when applying IO on them.
   *
   * - Parameters:
   *   - inputs: The input IOs for the new model, usually it is some set of Input objects.
   *   - outputs: The output IOs for the new model, usually it is outputs of some other models.
   *   - name: The name of the new model.
   */
  public convenience init(
    _ inputs: [IO], _ outputs: [IO], trainable: Bool? = nil, name: String = ""
  ) {
    let _inputs: [ccv_cnnp_model_io_t?] = inputs.map { $0._io }
    let _outputs: [ccv_cnnp_model_io_t?] = outputs.map { $0._io }
    let cModel = ccv_cnnp_model_new(
      _inputs, Int32(inputs.count), _outputs, Int32(outputs.count),
      trainable == true ? 1 : (trainable == false ? 0 : -1), name)!
    self.init(cModel)
    // Extending the lifetime of the inputs / outputs until self.init is done.
    withExtendedLifetime((inputs, outputs)) {}
  }

  /**
   * You can compose a new model of a list of models assuming one's output is another's input.
   *
   * - Parameters:
   *   - models: The array of models.
   *   - name: The name of the new model.
   */
  public convenience init(_ models: [Model], trainable: Bool? = nil, name: String = "") {
    let _models: [OpaquePointer?] = models.map { $0.cModel }
    let cModel = ccv_cnnp_sequential_new(
      _models, Int32(models.count), trainable == true ? 1 : (trainable == false ? 0 : -1), name)!
    self.init(cModel)
    // Extending the lifetime of the input until self.init is done.
    withExtendedLifetime(models) {}
  }

}

/// Model Inputs for Functional Model

public final class Input: Model.IO {
  public init() {
    let _io = ccv_cnnp_input()!
    super.init(_io)
  }
}

extension Model.IO {
  /**
   * Add non-functional dependencies between IOs. Normally, dependencies are inferred by usage.
   * However, in some cases you want to hack the dependency such that two unrelated ops can establish
   * a dependency. This is useful to enforce system to share memory for example.
   *
   * - Parameters:
   *   - dependencies: The IOs which will be the dependencies of the current IO.
   */
  public func add(dependencies: [Model.IO]) {
    let _dependencies: [ccv_cnnp_model_io_t?] = dependencies.map { $0._io }
    ccv_cnnp_model_add_dependencies(_io, _dependencies, Int32(dependencies.count))
  }
}
