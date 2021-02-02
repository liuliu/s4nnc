import C_nnc

extension Functional {
  /// Element-wise addition
  public static func sum<T: DynamicGraph.TensorGroup>(
    _ inputs: T..., streamContext: StreamContext? = nil
  ) -> T {
    precondition(inputs.count >= 2)
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: inputs[0], Array(inputs.suffix(from: 1)),
      outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Broadcast element-wise multiplication
  public static func mul<T: DynamicGraph.TensorGroup>(
    left: T, right: T, scalar: Float32 = 1, streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (scalar, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_MUL_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: left, right, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Broadcast element-wise addition
  public static func add<T: DynamicGraph.TensorGroup>(
    left: T, right: T, leftScalar: Float32 = 1, rightScalar: Float32 = 1,
    streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (leftScalar, rightScalar, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_ADD_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: left, right, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Element-wise log
  public static func log<T: DynamicGraph.TensorGroup>(_ one: T, streamContext: StreamContext? = nil)
    -> T
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWLOG_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Matrix multiplication
  public static func matmul<T: DynamicGraph.TensorGroup>(
    left: T, right: T, leftTranspose: (Int, Int) = (0, 0), rightTranspose: (Int, Int) = (0, 0),
    streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (1, 1, 0)
    params.blas.transpose_a = (Int32(leftTranspose.0), Int32(leftTranspose.1))
    params.blas.transpose_b = (Int32(rightTranspose.0), Int32(rightTranspose.1))
    let cmd = ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: left, right, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Scalar-matrix multiplication.
  public static func scalmul<T: DynamicGraph.TensorGroup>(
    left: Float, right: T, streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (left, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: right, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Make a copy.
  public static func copy<T: DynamicGraph.TensorGroup>(
    from: T, to: T, streamContext: StreamContext? = nil
  ) {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: from, outputs: [to], streamContext: streamContext)
  }

  /// Select input tensor with another index tensor.
  public static func indexSelect<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    input: T, index: U, streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_INDEX_SELECT_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, index, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }
}

extension DynamicGraph.Tensor {
  /// Transpose from axisA to axisB.
  public func transposed(_ axisA: Int, _ axisB: Int, streamContext: StreamContext? = nil)
    -> DynamicGraph.Tensor<Element>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.transpose.axis = (Int32(axisA), Int32(axisB))
    let cmd = ccv_nnc_cmd(CCV_NNC_TRANSPOSE_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

extension DynamicGraph.Group {
  /// Transpose from axisA to axisB.
  public func transposed(_ axisA: Int, _ axisB: Int, streamContext: StreamContext? = nil)
    -> DynamicGraph.Group<Element>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.transpose.axis = (Int32(axisA), Int32(axisB))
    let cmd = ccv_nnc_cmd(CCV_NNC_TRANSPOSE_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Group<Element>(outputs[0])
  }
}

extension DynamicGraph.Tensor {
  /// Fill the given tensor with uniform random values.
  public func rand(
    _ lowerBound: Float = 0, _ upperBound: Float = 1, streamContext: StreamContext? = nil
  ) {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (lowerBound, upperBound, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_RANDOM_UNIFORM_FORWARD, nil, params, 0)
    let _graph = graph._graph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, nil, 0, &_output, 1, 0, _streamContext)
  }
}

extension DynamicGraph.Group {
  /// Fill the given tensor with uniform random values.
  public func rand(
    _ lowerBound: Float = 0, _ upperBound: Float = 1, streamContext: StreamContext? = nil
  ) {
    guard underlyingArray.count > 0 else { return }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (lowerBound, upperBound, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_RANDOM_UNIFORM_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph._graph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _output: ccv_nnc_tensor_variable_t? = underlyingArray[0]._tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, nil, 0, &_output, 1, 0, _streamContext)
    ccv_nnc_dynamic_graph_set_no_grad(_graph, 1)
    let copy = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    // Init the rest of them to be the same.
    for rest in underlyingArray.suffix(from: 1) {
      var _target: ccv_nnc_tensor_variable_t? = rest._tensor
      ccv_nnc_dynamic_graph_exec(
        _graph, copy, ccv_nnc_no_hint, 0, &_output, 1, &_target, 1, 0, _streamContext)
    }
    ccv_nnc_dynamic_graph_set_no_grad(_graph, 0)
  }
}

extension DynamicGraph.Tensor {
  /// Copy the given tensor to GPU.
  public func toGPU(_ ordinal: Int = 0, streamContext: StreamContext? = nil)
    -> DynamicGraph.Tensor<Element>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    var _input: ccv_nnc_tensor_variable_t? = self._tensor
    let rawInput = self.rawValue
    let output: DynamicGraph.Tensor<Element> = graph.variable(
      .GPU(ordinal), format: rawInput.format, dimensions: rawInput.dimensions)
    var _output: ccv_nnc_tensor_variable_t? = output._tensor
    let _graph = graph._graph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
    return output
  }

  /// Copy the given tensor to CPU.
  public func toCPU(streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    var _input: ccv_nnc_tensor_variable_t? = self._tensor
    let rawInput = self.rawValue
    let output: DynamicGraph.Tensor<Element> = graph.variable(
      .CPU, format: rawInput.format, dimensions: rawInput.dimensions)
    var _output: ccv_nnc_tensor_variable_t? = output._tensor
    let _graph = graph._graph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
    return output
  }
}

extension DynamicGraph.Tensor {
  /// Fill the given tensor with a value.
  public func full(
    _ value: Float = 0, streamContext: StreamContext? = nil
  ) {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (value, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SET_FORWARD, nil, params, 0)
    let _graph = graph._graph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, nil, 0, &_output, 1, 0, _streamContext)
  }
}

extension DynamicGraph.Group {
  /// Fill the given tensor with a value.
  public func full(
    _ value: Float = 0, streamContext: StreamContext? = nil
  ) {
    guard underlyingArray.count > 0 else { return }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (value, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SET_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph._graph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _output: ccv_nnc_tensor_variable_t? = underlyingArray[0]._tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, nil, 0, &_output, 1, 0, _streamContext)
    ccv_nnc_dynamic_graph_set_no_grad(_graph, 1)
    let copy = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    // Init the rest of them to be the same.
    for rest in underlyingArray.suffix(from: 1) {
      var _target: ccv_nnc_tensor_variable_t? = rest._tensor
      ccv_nnc_dynamic_graph_exec(
        _graph, copy, ccv_nnc_no_hint, 0, &_output, 1, &_target, 1, 0, _streamContext)
    }
    ccv_nnc_dynamic_graph_set_no_grad(_graph, 0)
  }
}
