import C_nnc

public class AnyModelBuilder {

  public var isTest: Bool = false

  var model: Model? = nil
  private var t: Any? = nil
  private var inputs: [DynamicGraph.AnyTensor]? = nil
  private let builder: (_: Any, _: [DynamicGraph.AnyTensor]) -> Model

  fileprivate init(builder: @escaping (_: Any, _: [DynamicGraph.AnyTensor]) -> Model, name: String = "") {
    self.builder = builder
    let _model = ccv_cnnp_dynamic_new({ _, _, ctx in
      let modelBuilder = Unmanaged<AnyModelBuilder>.fromOpaque(ctx!).takeUnretainedValue()
      let t = modelBuilder.t!
      let inputs = modelBuilder.inputs!
      let builder = modelBuilder.builder
      let model = builder(t, inputs)
      return model.obtainUnderlyingModel()
    }, Unmanaged.passUnretained(self).toOpaque(), name)!
    model = Model(_model)
  }

  private var _outputSize: Int? = nil
  fileprivate var outputSize: Int {
    if let outputSize = _outputSize {
      return outputSize
    }
    let model = builder(t!, inputs!)
    let outputSize = Int(ccv_cnnp_model_output_size(model._model))
    _outputSize = outputSize
    return outputSize
  }

}

fileprivate extension AnyModelBuilder {
  func apply(_ t: Any, _ inputs: [DynamicGraph.AnyTensor], streamContext: StreamContext? = nil) -> [DynamicGraph.AnyTensor] {
    assert(inputs.count > 0)
    let graph = inputs[0].graph
    for input in inputs {
      assert(input.graph === graph)
    }
    let _inputs: [ccv_nnc_tensor_variable_t?] = inputs.map { $0._tensor }
    self.t = t
    self.inputs = inputs
    let outputSize = self.outputSize
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: outputSize)
    let outputs: [DynamicGraph.AnyTensor] = (0..<outputSize).map { _ in graph.variable() }
    for (i, variable) in outputs.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    let _graph = graph._graph
    let _streamContext = streamContext?._stream
    let _model = model!._model
    ccv_nnc_dynamic_graph_evaluate(_graph, _model, isTest ? 1 : 0, _inputs, Int32(_inputs.count), _outputs, Int32(outputSize), nil, _streamContext)
    _outputs.deallocate()
    self.inputs = nil
    return outputs
  }
}

public final class ModelBuilder<T>: AnyModelBuilder {
  public init(_ builder: @escaping(_: T, _: [DynamicGraph.AnyTensor]) -> Model, name: String = "") {
    super.init(builder: { t, inputs in
      return builder(t as! T, inputs)
    }, name: name)
  }
}

public extension ModelBuilder {
  func callAsFunction(_ t: T, _ inputs: [DynamicGraph.AnyTensor], streamContext: StreamContext? = nil) -> [DynamicGraph.AnyTensor] {
    return apply(t, inputs, streamContext: streamContext)
  }
}

public extension ModelBuilder where T == Void {
  convenience init(_ builder: @escaping(_: [DynamicGraph.AnyTensor]) -> Model, name: String = "") {
    self.init({ t, inputs in
      return builder(inputs)
    }, name: name)
  }
  func callAsFunction(_ inputs: [DynamicGraph.AnyTensor], streamContext: StreamContext? = nil) -> [DynamicGraph.AnyTensor] {
    return self(Void(), inputs, streamContext: streamContext)
  }
}
