import C_nnc

/// This protocol defines a generic constructs such that we can use DynamicGraph.AnyTensorGroup
/// to represent either a collection of tensors from multiple GPUs or one tensor.
/// It has a typed version DynamicGraph.TensorGroup to enforce type constraint.

public protocol DynamicGraph_AnyTensorGroup: DynamicGraph_Any {
  associatedtype AnyTensor: DynamicGraph_Any
  static func exec(
    cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_Any?], outputSize: Int32,
    streamContext: StreamContext?
  ) -> [AnyTensor]
  static func exec(
    cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_Any?],
    outputs: [DynamicGraph_Any],
    streamContext: StreamContext?)
  static func evaluate(
    model: OpaquePointer, isTest: Bool, dataParallel: inout Int?, inputs: [DynamicGraph_Any?],
    outputSize: Int32, streamContext: StreamContext?
  ) -> [AnyTensor]
}

extension DynamicGraph {
  public typealias AnyTensorGroup = DynamicGraph_AnyTensorGroup
}

extension DynamicGraph.AnyTensor: DynamicGraph_Any {
  public var untyped: [DynamicGraph.AnyTensor] { [self] }
}

extension DynamicGraph.AnyTensor: DynamicGraph.AnyTensorGroup {

  public typealias AnyTensor = DynamicGraph.AnyTensor

  public static func exec(
    cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_Any?], outputSize: Int32,
    streamContext: StreamContext?
  ) -> [AnyTensor] {
    precondition(inputs.count > 0)
    let firstTensor = inputs.first(where: { $0 != nil })!!
    let graph = firstTensor.graph
    for input in inputs {
      guard let input = input else { continue }
      precondition(input.graph === graph)
    }
    let _inputs: [ccv_nnc_tensor_variable_t?] = inputs.map { $0?.untyped[0]._tensor }
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    // Constants are very conservative, if all inputs are constants, then outputs can be constants.
    var outputsCanBeConstants = true
    for x in inputs {
      guard let x = x else { continue }
      if !x.isConstant {
        outputsCanBeConstants = false
        break
      }
    }
    let outputs: [DynamicGraph.AnyTensor]
    if outputsCanBeConstants {
      outputs = (0..<outputSize).map { _ in graph.constant() }
    } else {
      outputs = (0..<outputSize).map { _ in graph.variable() }
    }
    for (i, variable) in outputs.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, hint, 0, _inputs, Int32(_inputs.count), _outputs, outputSize, 0, _streamContext)
    _outputs.deallocate()
    return outputs
  }

  public static func exec(
    cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_Any?],
    outputs: [DynamicGraph_Any],
    streamContext: StreamContext?
  ) {
    precondition(inputs.count > 0)
    let firstTensor = inputs.first(where: { $0 != nil })!!
    let graph = firstTensor.graph
    for input in inputs {
      guard let input = input else { continue }
      precondition(input.graph === graph)
    }
    let _inputs: [ccv_nnc_tensor_variable_t?] = inputs.map { $0?.untyped[0]._tensor }
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: outputs.count)
    for (i, variable) in outputs.enumerated() {
      (_outputs + i).initialize(to: variable.untyped[0]._tensor)
    }
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, hint, 0, _inputs, Int32(_inputs.count), _outputs, Int32(outputs.count), 0,
      _streamContext)
    _outputs.deallocate()
  }

  public static func evaluate(
    model: OpaquePointer, isTest: Bool, dataParallel: inout Int?, inputs: [DynamicGraph_Any?],
    outputSize: Int32, streamContext: StreamContext?
  ) -> [AnyTensor] {
    precondition(inputs.count > 0)
    let firstTensor = inputs.first(where: { $0 != nil })!!
    let graph = firstTensor.graph
    for input in inputs {
      guard let input = input else { continue }
      precondition(input.graph === graph)
    }
    let _inputs: [ccv_nnc_tensor_variable_t?] = inputs.map { $0?.untyped[0]._tensor }
    // Constants are very conservative, if all inputs are constants, then outputs can be constants.
    var outputsCanBeConstants = true
    for x in inputs {
      guard let x = x else { continue }
      if !x.isConstant {
        outputsCanBeConstants = false
        break
      }
    }
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    let outputs: [DynamicGraph.AnyTensor]
    if outputsCanBeConstants {
      outputs = (0..<outputSize).map { _ in graph.constant() }
    } else {
      outputs = (0..<outputSize).map { _ in graph.variable() }
    }
    for (i, variable) in outputs.enumerated() {
      assert(variable.graph === graph)
      (_outputs + i).initialize(to: variable._tensor)
    }
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_evaluate(
      _graph, model, isTest ? 1 : 0, _inputs, Int32(_inputs.count), _outputs, outputSize, nil,
      _streamContext)
    // Set gradient update to noop. These will be reset when we call Optimizer.step.
    let params = CmdParamsFactory.factory.newParams()
    let noop = ccv_nnc_cmd(CCV_NNC_NOOP, nil, params, 0)
    ccv_cnnp_model_set_minimizer(model, noop, 1, nil, 0)
    _outputs.deallocate()
    return outputs
  }

}

extension DynamicGraph.Group: DynamicGraph_Any where Element: DynamicGraph.AnyTensor {
}

extension DynamicGraph.Group: DynamicGraph.AnyTensorGroup where Element: DynamicGraph.AnyTensor {

  public typealias AnyTensor = DynamicGraph.Group<DynamicGraph.AnyTensor>

  public static func exec(
    cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_Any?], outputSize: Int32,
    streamContext: StreamContext?
  ) -> [AnyTensor] {
    precondition(inputs.count > 0)
    let firstTensor = inputs.first(where: { $0 != nil })!!
    let graph = firstTensor.graph
    let parallel = firstTensor.untyped.count
    let inputSize = inputs.count
    var _inputs = [ccv_nnc_tensor_variable_t?](repeating: nil, count: parallel * inputSize)
    var outputsCanBeConstants = true
    for (i, input) in inputs.enumerated() {
      guard let input = input else { continue }
      assert(input.untyped.count == parallel)
      for (j, tensor) in input.untyped.enumerated() {
        assert(tensor.graph === graph)
        _inputs[j * inputSize + i] = tensor._tensor
        if !tensor.isConstant {
          outputsCanBeConstants = false
        }
      }
    }
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize) * parallel)
    let outputs: [DynamicGraph.Group<DynamicGraph.AnyTensor>]
    if outputsCanBeConstants {
      outputs = (0..<outputSize).map { _ in
        DynamicGraph.Group((0..<parallel).map { _ in graph.constant() })
      }
    } else {
      outputs = (0..<outputSize).map { _ in
        DynamicGraph.Group((0..<parallel).map { _ in graph.variable() })
      }
    }
    for (i, output) in outputs.enumerated() {
      for (j, tensor) in output.enumerated() {
        (_outputs + j * Int(outputSize) + i).initialize(to: tensor._tensor)
      }
    }
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, hint, 0, _inputs, Int32(inputSize * parallel), _outputs,
      outputSize * Int32(parallel), Int32(parallel), _streamContext)
    _outputs.deallocate()
    return outputs
  }

  public static func exec(
    cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_Any?],
    outputs: [DynamicGraph_Any],
    streamContext: StreamContext?
  ) {
    precondition(inputs.count > 0)
    let firstTensor = inputs.first(where: { $0 != nil })!!
    let graph = firstTensor.graph
    let parallel = firstTensor.untyped.count
    let inputSize = inputs.count
    var _inputs = [ccv_nnc_tensor_variable_t?](repeating: nil, count: parallel * inputSize)
    for (i, input) in inputs.enumerated() {
      guard let input = input else { continue }
      assert(input.untyped.count == parallel)
      for (j, tensor) in input.untyped.enumerated() {
        assert(tensor.graph === graph)
        _inputs[j * inputSize + i] = tensor._tensor
      }
    }
    let outputSize = outputs.count
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: outputSize * parallel)
    for (i, output) in outputs.enumerated() {
      assert(output.untyped.count == parallel)
      for (j, tensor) in output.untyped.enumerated() {
        assert(tensor.graph === graph)
        (_outputs + j * outputSize + i).initialize(to: tensor._tensor)
      }
    }
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, hint, 0, _inputs, Int32(inputSize * parallel), _outputs,
      Int32(outputSize * parallel), Int32(parallel), _streamContext)
    _outputs.deallocate()
  }

  public static func evaluate(
    model: OpaquePointer, isTest: Bool, dataParallel: inout Int?, inputs: [DynamicGraph_Any?],
    outputSize: Int32, streamContext: StreamContext?
  ) -> [AnyTensor] {
    precondition(inputs.count > 0)
    precondition(inputs.count > 0)
    let firstTensor = inputs.first(where: { $0 != nil })!!
    let graph = firstTensor.graph
    let parallel = firstTensor.untyped.count
    let inputSize = inputs.count
    var _inputs = [ccv_nnc_tensor_variable_t?](repeating: nil, count: parallel * inputSize)
    var outputsCanBeConstants = true
    for (i, input) in inputs.enumerated() {
      guard let input = input else { continue }
      assert(input.untyped.count == parallel)
      for (j, tensor) in input.untyped.enumerated() {
        assert(tensor.graph === graph)
        _inputs[j * inputSize + i] = tensor._tensor
        if !tensor.isConstant {
          outputsCanBeConstants = false
        }
      }
    }
    if let dataParallel = dataParallel {
      // You cannot run a model previously parallel and then not.
      assert(dataParallel == parallel)
    } else {
      ccv_cnnp_model_set_data_parallel(model, Int32(parallel))
      dataParallel = parallel
    }
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize) * parallel)
    let outputs: [DynamicGraph.Group<DynamicGraph.AnyTensor>]
    if outputsCanBeConstants {
      outputs = (0..<outputSize).map { _ in
        DynamicGraph.Group((0..<parallel).map { _ in graph.constant() })
      }
    } else {
      outputs = (0..<outputSize).map { _ in
        DynamicGraph.Group((0..<parallel).map { _ in graph.variable() })
      }
    }
    for (i, output) in outputs.enumerated() {
      for (j, tensor) in output.enumerated() {
        (_outputs + j * Int(outputSize) + i).initialize(to: tensor._tensor)
      }
    }
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_evaluate(
      _graph, model, isTest ? 1 : 0, _inputs, Int32(_inputs.count), _outputs,
      outputSize * Int32(parallel), nil, _streamContext)
    // Set gradient update to noop. These will be reset when we call Optimizer.step.
    let params = CmdParamsFactory.factory.newParams()
    let noop = ccv_nnc_cmd(CCV_NNC_NOOP, nil, params, 0)
    ccv_cnnp_model_set_minimizer(model, noop, 1, nil, 0)
    _outputs.deallocate()
    return outputs
  }

}

public enum Functional {
  internal static func exec<T: DynamicGraph.AnyTensorGroup>(
    _: T.Type, cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_Any?],
    outputSize: Int32,
    streamContext: StreamContext? = nil
  ) -> [T.AnyTensor] {
    return T.exec(
      cmd: cmd, hint: hint, inputs: inputs, outputSize: outputSize, streamContext: streamContext)
  }
  static func exec<T: DynamicGraph.AnyTensorGroup>(
    cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs firstInput: T?,
    _ restInputs: [DynamicGraph_Any?], outputSize: Int32, streamContext: StreamContext? = nil
  ) -> [T.AnyTensor] {
    let tensorInputs: [DynamicGraph_Any?] = [firstInput as DynamicGraph_Any?] + restInputs
    return exec(
      T.self, cmd: cmd, hint: hint, inputs: tensorInputs, outputSize: outputSize,
      streamContext: streamContext)
  }
  static func exec<T: DynamicGraph.AnyTensorGroup>(
    cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs firstInput: T?,
    _ restInputs: DynamicGraph_Any?..., outputSize: Int32, streamContext: StreamContext? = nil
  ) -> [T.AnyTensor] {
    exec(
      cmd: cmd, hint: hint, inputs: firstInput, restInputs, outputSize: outputSize,
      streamContext: streamContext)
  }
  internal static func exec<T: DynamicGraph.AnyTensorGroup>(
    _: T.Type, cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_Any?],
    outputs: [DynamicGraph_Any], streamContext: StreamContext? = nil
  ) {
    T.exec(cmd: cmd, hint: hint, inputs: inputs, outputs: outputs, streamContext: streamContext)
  }
  static func exec<T: DynamicGraph.AnyTensorGroup>(
    cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs firstInput: T?,
    _ restInputs: [DynamicGraph_Any?], outputs: [DynamicGraph_Any],
    streamContext: StreamContext? = nil
  ) {
    let tensorInputs: [DynamicGraph_Any?] = [firstInput as DynamicGraph_Any?] + restInputs
    return exec(
      T.self, cmd: cmd, hint: hint, inputs: tensorInputs, outputs: outputs,
      streamContext: streamContext)
  }
  static func exec<T: DynamicGraph.AnyTensorGroup>(
    cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs firstInput: T?,
    _ restInputs: DynamicGraph_Any?..., outputs: [DynamicGraph_Any],
    streamContext: StreamContext? = nil
  ) {
    exec(
      cmd: cmd, hint: hint, inputs: firstInput, restInputs, outputs: outputs,
      streamContext: streamContext)
  }
}

extension Model {
  fileprivate func callAsFunction<T: DynamicGraph.AnyTensorGroup>(
    _: T.Type, _ inputs: [DynamicGraph_Any], streamContext: StreamContext? = nil
  ) -> [T.AnyTensor] {
    let outputSize = ccv_cnnp_model_output_size(cModel)
    graph = inputs.first?.graph
    return T.evaluate(
      model: cModel, isTest: isTest, dataParallel: &dataParallel, inputs: inputs,
      outputSize: outputSize, streamContext: streamContext)
  }
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup>(
    inputs firstInput: T, _ restInputs: [DynamicGraph_Any], streamContext: StreamContext? = nil
  ) -> [T.AnyTensor] {
    let tensorInputs: [DynamicGraph_Any] = [firstInput as DynamicGraph_Any] + restInputs
    return self(T.self, tensorInputs, streamContext: streamContext)
  }

  public func callAsFunction<T: DynamicGraph.AnyTensorGroup>(
    inputs firstInput: T, _ restInputs: DynamicGraph_Any..., streamContext: StreamContext? = nil
  ) -> [T.AnyTensor] {
    self(inputs: firstInput, restInputs, streamContext: streamContext)
  }
}

extension AnyModelBuilder {
  fileprivate func apply<U: DynamicGraph.AnyTensorGroup>(
    ofType: U.Type, _ t: Any, _ inputs: [DynamicGraph_Any], streamContext: StreamContext? = nil
  ) -> [U.AnyTensor] {
    assert(inputs.count > 0)
    self.t = t
    self.inputs = inputs
    let outputSize = self.outputSize
    model!.graph = inputs.first?.graph
    let outputs = U.evaluate(
      model: model!.cModel, isTest: isTest, dataParallel: &model!.dataParallel, inputs: inputs,
      outputSize: Int32(outputSize), streamContext: streamContext)
    self.inputs = nil
    return outputs
  }
}

extension ModelBuilder {
  public func callAsFunction<U: DynamicGraph.AnyTensorGroup>(
    _ t: T, inputs firstInput: U, _ restInputs: DynamicGraph_Any...,
    streamContext: StreamContext? = nil
  ) -> [U.AnyTensor] {
    let tensorInputs: [DynamicGraph_Any] = [firstInput as DynamicGraph_Any] + restInputs
    return apply(ofType: U.self, t, tensorInputs, streamContext: streamContext)
  }
}

extension ModelBuilder where T == Void {
  public func callAsFunction<U: DynamicGraph.AnyTensorGroup>(
    inputs firstInput: U, _ restInputs: DynamicGraph_Any..., streamContext: StreamContext? = nil
  ) -> [U.AnyTensor] {
    let tensorInputs: [DynamicGraph_Any] = [firstInput as DynamicGraph_Any] + restInputs
    return apply(ofType: U.self, Void(), tensorInputs, streamContext: streamContext)
  }
}
