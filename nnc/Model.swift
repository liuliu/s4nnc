import C_nnc

public class Model {

  public class IO {

    let _io: ccv_cnnp_model_io_t
    let model: Model?
    private let inputs: [IO]?

    init(_ io: ccv_cnnp_model_io_t, model: Model? = nil, inputs: [IO]? = nil) {
      _io = io
      self.model = model
      self.inputs = inputs
    }
  }

  public var isTest: Bool = false

  var dataParallel: Int? = nil // Keep track of whether we applied data parallel to the model or not.
  let _model: OpaquePointer
  var owner: Model? = nil

  private func ownerHook() {
    ccv_cnnp_model_notify_hook(_model, { _, _, payload, ctx in
      guard payload != nil && payload != ctx else { return }
      let model = Unmanaged<Model>.fromOpaque(ctx!).takeUnretainedValue()
      model.owner = Unmanaged<Model>.fromOpaque(payload!).takeUnretainedValue()
    }, Unmanaged.passUnretained(self).toOpaque())
  }

  init(_ model: OpaquePointer) {
    _model = model
    ccv_cnnp_model_notify(_model, 0, Unmanaged.passUnretained(self).toOpaque())
    ownerHook()
  }

  func obtainUnderlyingModel(_ owner: Model) -> OpaquePointer {
    ccv_cnnp_model_notify(_model, 0, Unmanaged.passUnretained(owner).toOpaque())
    // self.owner = owner is not necessary because we will update in the callback.
    assert(self.owner === owner)
    return _model
  }

  public func callAsFunction(_ inputs: IO...) -> IO {
    let _inputs: [ccv_cnnp_model_io_t?] = inputs.map { $0._io }
    let _io = ccv_cnnp_model_apply(_model, _inputs, Int32(inputs.count))!
    return IO(_io, model: self, inputs: inputs)
  }

  deinit {
    if owner == nil {
      ccv_cnnp_model_free(_model)
      return
    }
    // Unhook because I am no longer active (but the model can still be available).
    ccv_cnnp_model_notify_hook(_model, nil, nil)
  }

  public final class Parameters: IO {
  }

  var _parameters: ccv_cnnp_model_io_t? = nil

  public var parameters: Parameters {
    guard let _parameters = _parameters else {
      let parameters = ccv_cnnp_model_parameters(_model, -1, -1)!
      self._parameters = parameters
      return Parameters(parameters, model: self)
    }
    return Parameters(_parameters, model: self)
  }
  
  public enum ParametersType {
    case weight
    case bias
  }

  private var _biasParameters: ccv_cnnp_model_io_t? = nil
  private var _weightParameters: ccv_cnnp_model_io_t? = nil

  public func parameters(for type: ParametersType) -> Parameters {
    switch type {
    case .weight:
      guard let _weightParameters = _weightParameters else {
        let weightParameters = ccv_cnnp_model_parameters(_model, Int32(CCV_CNNP_PARAMETER_SELECT_WEIGHT), -1)!
        self._weightParameters = weightParameters
        return Parameters(weightParameters, model: self)
      }
      return Parameters(_weightParameters, model: self)
    case .bias:
      guard let _biasParameters = _biasParameters else {
        let biasParameters = ccv_cnnp_model_parameters(_model, Int32(CCV_CNNP_PARAMETER_SELECT_BIAS), -1)!
        self._biasParameters = biasParameters
        return Parameters(biasParameters, model: self)
      }
      return Parameters(_biasParameters, model: self)
    }
  }

}

/**
 * MARK - Functional and Sequential Models
 */

extension Model {

  public convenience init(_ inputs: [IO], _ outputs: [IO], name: String = "") {
    let _inputs: [ccv_cnnp_model_io_t?] = inputs.map { $0._io }
    let _outputs: [ccv_cnnp_model_io_t?] = outputs.map { $0._io }
    let _model = ccv_cnnp_model_new(_inputs, Int32(inputs.count), _outputs, Int32(outputs.count), name)!
    self.init(_model);
  }

  public convenience init(_ models: [Model], name: String = "") {
    let _models: [OpaquePointer?] = models.map { $0._model }
    let _model = ccv_cnnp_sequential_new(_models, Int32(models.count), name)!
    self.init(_model)
  }

}

/**
 * MARK - Model Inputs for Functional Model
 */

public final class Input: Model.IO {
  public init() {
    let _io = ccv_cnnp_input()!
    super.init(_io)
  }
}

/**
 * MARK - Hashable
 */

extension Model: Hashable {
  public static func ==(lhs: Model, rhs: Model) -> Bool {
    return lhs === rhs
  }

  public func hash(into hasher: inout Hasher) {
    hasher.combine(ObjectIdentifier(self))
  }
}
