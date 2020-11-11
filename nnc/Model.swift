import C_nnc

public class Model {

  public class IO {

    let _io: ccv_cnnp_model_io_t
    private let model: Model?
    private let inputs: [IO]?

    init(_ io: ccv_cnnp_model_io_t, model: Model? = nil, inputs: [IO]? = nil) {
      _io = io
      self.model = model
      self.inputs = inputs
    }
  }

  public var isTest: Bool = false

  let _model: OpaquePointer
  private var selfOwned: Bool = true

  private func ownerHook() {
    ccv_cnnp_model_owner_hook(_model, { _, owner, ctx in
      guard owner != nil else { return }
      let model = Unmanaged<Model>.fromOpaque(ctx!).takeUnretainedValue()
      model.selfOwned = false // No longer owned it, there is a new owner (!= nil).
    }, Unmanaged.passUnretained(self).toOpaque())
  }

  init(_ model: OpaquePointer) {
    _model = model
    ownerHook()
  }

  func obtainUnderlyingModel() -> OpaquePointer {
    selfOwned = false
    return _model
  }

  public func callAsFunction(_ inputs: IO...) -> IO {
    let _inputs: [ccv_cnnp_model_io_t?] = inputs.map { $0._io }
    let _io = ccv_cnnp_model_apply(_model, _inputs, Int32(inputs.count))!
    return IO(_io, model: self, inputs: inputs)
  }

  deinit {
    if selfOwned {
      ccv_cnnp_model_free(_model)
      return
    }
    // Unhook because I am no longer active (but the model can still be available).
    ccv_cnnp_model_owner_hook(_model, nil, nil)
  }

  public init(_ inputs: [IO], _ outputs: [IO], name: String = "") {
    let _inputs: [ccv_cnnp_model_io_t?] = inputs.map { $0._io }
    let _outputs: [ccv_cnnp_model_io_t?] = outputs.map { $0._io }
    _model = ccv_cnnp_model_new(_inputs, Int32(inputs.count), _outputs, Int32(outputs.count), name)!
    ownerHook()
  }

  public init(_ models: [Model], name: String = "") {
    let _models: [OpaquePointer?] = models.map { $0._model }
    _model = ccv_cnnp_sequential_new(_models, Int32(models.count), name)!
    ownerHook()
  }

}

public final class Input: Model.IO {
  public init() {
    let _io = ccv_cnnp_input()!
    super.init(_io)
  }
}
