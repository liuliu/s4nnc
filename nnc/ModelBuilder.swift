import C_nnc

public class AnyModelBuilder {

  public var isTest: Bool = false

  var model: Model? = nil
  var t: Any? = nil
  var inputs: [DynamicGraph_Any]? = nil
  private let builder: (_: Any, _: [DynamicGraph_Any]) -> Model

  fileprivate init(builder: @escaping (_: Any, _: [DynamicGraph_Any]) -> Model, name: String = "") {
    self.builder = builder
    let _model = ccv_cnnp_dynamic_new({ _, _, ctx in
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
    let model = builder(t!, inputs!)
    let outputSize = Int(ccv_cnnp_model_output_size(model._model))
    _outputSize = outputSize
    return outputSize
  }

}

public final class ModelBuilder<T>: AnyModelBuilder {
  public init(_ builder: @escaping(_: T, _: [DynamicGraph_Any]) -> Model, name: String = "") {
    super.init(builder: { t, inputs in
      return builder(t as! T, inputs)
    }, name: name)
  }
}

public extension ModelBuilder where T == Void {
  convenience init(_ builder: @escaping(_: [DynamicGraph_Any]) -> Model, name: String = "") {
    self.init({ t, inputs in
      return builder(inputs)
    }, name: name)
  }
}
