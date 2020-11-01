import C_nnc

public enum Functional {
  static func exec(cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph.AnyTensor], outputSize: Int32, streamContext: StreamContext? = nil) -> [DynamicGraph.AnyTensor] {
    assert(inputs.count > 0)
    let graph = inputs[0].graph
    for input in inputs {
      assert(input.graph === graph)
    }
    let _inputs: [ccv_nnc_tensor_variable_t?] = inputs.map { $0._tensor }
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: Int(outputSize))
    let outputs: [DynamicGraph.AnyTensor] = (0..<outputSize).map { _ in graph.variable() }
    for (i, variable) in outputs.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    let _graph = graph._graph
    let _streamContext = streamContext?._stream
    ccv_nnc_dynamic_graph_exec(_graph, cmd, hint, 0, _inputs, Int32(_inputs.count), _outputs, outputSize, 0, _streamContext)
    _outputs.deallocate()
    return outputs
  }
}

public extension Functional {
  static func mul<Element>(left: DynamicGraph.Tensor<Element>, right: DynamicGraph.Tensor<Element>, scalar: Float32 = 1, streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = ccv_nnc_cmd_param_t()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (scalar, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_MUL_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [left, right], outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }

  // Element-wise addition
  static func add<Element>(left: DynamicGraph.Tensor<Element>, right: DynamicGraph.Tensor<Element>, leftScalar: Float32 = 1, rightScalar: Float32 = 1, streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = ccv_nnc_cmd_param_t()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (leftScalar, rightScalar, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_ADD_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [left, right], outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }

  // Element-wise log
  static func log<Element>(_ one: DynamicGraph.Tensor<Element>, streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    let params = ccv_nnc_cmd_param_t()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWLOG_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [one], outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }

  // Matrix multiplication
  static func matmul<Element>(left: DynamicGraph.Tensor<Element>, right: DynamicGraph.Tensor<Element>, leftTranspose: (Int, Int) = (0, 0), rightTranspose: (Int, Int) = (0, 0), streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = ccv_nnc_cmd_param_t()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (1, 1, 0)
    params.blas.transpose_a = (Int32(leftTranspose.0), Int32(leftTranspose.1))
    params.blas.transpose_b = (Int32(rightTranspose.0), Int32(rightTranspose.1))
    let cmd = ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [left, right], outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }

  // Scalar-matrix multiplication
  static func scalmul<Element>(left: Float, right: DynamicGraph.Tensor<Element>, streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = ccv_nnc_cmd_param_t()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (left, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [right], outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

public extension DynamicGraph.Tensor {

  func transpose(_ axisA: Int, _ axisB: Int, streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = ccv_nnc_cmd_param_t()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.transpose.axis = (Int32(axisA), Int32(axisB))
    let cmd = ccv_nnc_cmd(CCV_NNC_TRANSPOSE_FORWARD, nil, params, 0)
    let outputs = Functional.exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [self], outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }

}

public extension DynamicGraph.Tensor {
  func toGPU(_ index: Int = 0, streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = ccv_nnc_cmd_param_t()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    var _input: ccv_nnc_tensor_variable_t? = self._tensor
    let rawInput = self.rawValue
    let output: DynamicGraph.Tensor<Element> = graph.variable(.GPU(index), format: rawInput.format, dimensions: rawInput.dimensions)
    var _output: ccv_nnc_tensor_variable_t? = output._tensor
    let _graph = graph._graph
    let _streamContext = streamContext?._stream
    ccv_nnc_dynamic_graph_exec(_graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
    return output
  }

  func toCPU(streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = ccv_nnc_cmd_param_t()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    var _input: ccv_nnc_tensor_variable_t? = self._tensor
    let rawInput = self.rawValue
    let output: DynamicGraph.Tensor<Element> = graph.variable(.CPU, format: rawInput.format, dimensions: rawInput.dimensions)
    var _output: ccv_nnc_tensor_variable_t? = output._tensor
    let _graph = graph._graph
    let _streamContext = streamContext?._stream
    ccv_nnc_dynamic_graph_exec(_graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
    return output
  }
}
