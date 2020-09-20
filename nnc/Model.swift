import C_nnc

public class Model {

  public class IO {

    let _io: ccv_cnnp_model_io_t

    init(_ io: ccv_cnnp_model_io_t) {
      _io = io
    }
  }

  public var isTest: Bool = false

  let _model: OpaquePointer

  init(_ model: OpaquePointer) {
    _model = model
  }

  public func apply(_ inputs: IO...) -> IO {
    let _inputs: [ccv_cnnp_model_io_t?] = inputs.map { $0._io }
    let _io = ccv_cnnp_model_apply(_model, _inputs, Int32(inputs.count))!
    return IO(_io)
  }

}

public final class Input: Model.IO {
  public init() {
    let _io = ccv_cnnp_input()!
    super.init(_io)
  }
}

public final class Functional: Model {
  public init(_ inputs: [IO], _ outputs: [IO], name: String = "") {
    let _inputs: [ccv_cnnp_model_io_t?] = inputs.map { $0._io }
    let _outputs: [ccv_cnnp_model_io_t?] = outputs.map { $0._io }
    let _model = ccv_cnnp_model_new(_inputs, Int32(inputs.count), _outputs, Int32(outputs.count), name)!
    super.init(_model)
  }
}

public final class Sequential: Model {
  public init(_ models: [Model], name: String = "") {
    let _models: [OpaquePointer?] = models.map { $0._model }
    let _model = ccv_cnnp_sequential_new(_models, Int32(models.count), name)!
    super.init(_model)
  }
}

public extension Model {
  func callAsFunction(_ inputs: [DynamicGraph.AnyTensor], streamContext: StreamContext? = nil) -> [DynamicGraph.AnyTensor] {
    assert(inputs.count > 0)
    let graph = inputs[0].graph
    var _inputs = [ccv_nnc_tensor_variable_t?]()
    for input in inputs {
      assert(ObjectIdentifier(input.graph) == ObjectIdentifier(graph))
      _inputs.append(input._tensor)
    }
    let outputSize = ccv_cnnp_model_output_size(_model)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: Int(outputSize))
    var outputs = [DynamicGraph.AnyTensor]()
    for i in 0..<outputSize {
      let variable = graph.variable()
      outputs.append(variable)
      (_outputs + Int(i)).initialize(to: variable._tensor)
    }
    let _graph = graph._graph
    let _streamContext = streamContext?._stream
    ccv_nnc_dynamic_graph_evaluate(_graph, _model, isTest ? 1 : 0, _inputs, Int32(_inputs.count), _outputs, outputSize, nil, _streamContext)
    _outputs.deallocate()
    return outputs
  }
}
