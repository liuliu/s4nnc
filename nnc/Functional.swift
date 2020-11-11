import C_nnc

public protocol DynamicGraph_AnyTensorGroup {
  static func exec(cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_AnyTensorGroup], outputSize: Int32, streamContext: StreamContext?) -> [DynamicGraph_AnyTensorGroup]
}

public extension DynamicGraph {
  typealias AnyTensorGroup = DynamicGraph_AnyTensorGroup
}

extension DynamicGraph.AnyTensor: DynamicGraph.AnyTensorGroup {
  public static func exec(cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_AnyTensorGroup], outputSize: Int32, streamContext: StreamContext?) -> [DynamicGraph_AnyTensorGroup] {
    assert(inputs.count > 0)
    let tensorInputs = inputs as! [DynamicGraph.AnyTensor]
    let graph = tensorInputs[0].graph
    for input in tensorInputs {
      assert(input.graph === graph)
    }
    let _inputs: [ccv_nnc_tensor_variable_t?] = tensorInputs.map { $0._tensor }
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

extension Array: DynamicGraph.AnyTensorGroup where Element: DynamicGraph.AnyTensor {
  public static func exec(cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [DynamicGraph_AnyTensorGroup], outputSize: Int32, streamContext: StreamContext?) -> [DynamicGraph_AnyTensorGroup] {
    return []
  }
}

public protocol DynamicGraph_TensorGroup: DynamicGraph_AnyTensorGroup {
  associatedtype Element: TensorNumeric
  init(_: DynamicGraph_AnyTensorGroup)
}

public extension DynamicGraph {
  typealias TensorGroup = DynamicGraph_TensorGroup
}

public protocol _DynamicGraph_TensorGroup {
  associatedtype _Element: TensorNumeric
  init(_: DynamicGraph_AnyTensorGroup)
}

extension DynamicGraph.Tensor: _DynamicGraph_TensorGroup {
  public typealias _Element = Element
}

extension DynamicGraph.Tensor: DynamicGraph.TensorGroup {
  public typealias Element = Element
  public convenience init(_ obj: DynamicGraph.AnyTensorGroup) {
    guard obj is DynamicGraph.AnyTensor else {
      fatalError("Cannot convert from anything other than AnyTensor")
    }
    self.init(obj as! DynamicGraph.AnyTensor)
  }
}

extension Array: DynamicGraph.TensorGroup where Element: _DynamicGraph_TensorGroup, Element: DynamicGraph.AnyTensor {
  public typealias Element = Element._Element
  public init(_ obj: DynamicGraph.AnyTensorGroup) {
    guard obj is [DynamicGraph.AnyTensor] else {
      fatalError("Cannot convert from anything other than array of AnyTensor")
    }
    self.init((obj as! [DynamicGraph.AnyTensor]).map { Element($0) })
  }
}

public enum Functional {
  static func exec<T: DynamicGraph.AnyTensorGroup>(cmd: ccv_nnc_cmd_t, hint: ccv_nnc_hint_t, inputs: [T], outputSize: Int32, streamContext: StreamContext? = nil) -> [DynamicGraph.AnyTensorGroup] {
    return T.exec(cmd: cmd, hint: hint, inputs: inputs, outputSize: outputSize, streamContext: streamContext)
  }
}

public extension Functional {
  // Element-wise addition
  static func sum<T: DynamicGraph.TensorGroup>(left: T, right: T, scalar: Float32 = 1, streamContext: StreamContext? = nil) -> T {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [left, right], outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  // Broadcast element-wise multiplication
  static func mul<T: DynamicGraph.TensorGroup>(left: T, right: T, scalar: Float32 = 1, streamContext: StreamContext? = nil) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (scalar, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_MUL_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [left, right], outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  // Broadcast element-wise addition
  static func add<T: DynamicGraph.TensorGroup>(left: T, right: T, leftScalar: Float32 = 1, rightScalar: Float32 = 1, streamContext: StreamContext? = nil) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (leftScalar, rightScalar, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_ADD_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [left, right], outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  // Element-wise log
  static func log<T: DynamicGraph.TensorGroup>(_ one: T, streamContext: StreamContext? = nil) -> T {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWLOG_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [one], outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  // Matrix multiplication
  static func matmul<T: DynamicGraph.TensorGroup>(left: T, right: T, leftTranspose: (Int, Int) = (0, 0), rightTranspose: (Int, Int) = (0, 0), streamContext: StreamContext? = nil) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (1, 1, 0)
    params.blas.transpose_a = (Int32(leftTranspose.0), Int32(leftTranspose.1))
    params.blas.transpose_b = (Int32(rightTranspose.0), Int32(rightTranspose.1))
    let cmd = ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [left, right], outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  // Scalar-matrix multiplication
  static func scalmul<T: DynamicGraph.TensorGroup>(left: Float, right: T, streamContext: StreamContext? = nil) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (left, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [right], outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  static func indexSelect<Element, Long>(input: DynamicGraph.Tensor<Element>, index: DynamicGraph.Tensor<Long>, streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_INDEX_SELECT_FORWARD, nil, params, 0)
    let outputs = exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [input, index], outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

public extension DynamicGraph.Tensor {
  func transpose(_ axisA: Int, _ axisB: Int, streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.transpose.axis = (Int32(axisA), Int32(axisB))
    let cmd = ccv_nnc_cmd(CCV_NNC_TRANSPOSE_FORWARD, nil, params, 0)
    let outputs = Functional.exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [self], outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

public extension DynamicGraph.Tensor {
  func rand(_ lowerBound: Float = 0, _ upperBound: Float = 1, streamContext: StreamContext? = nil) {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.blas.a = (lowerBound, upperBound, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_RANDOM_UNIFORM_FORWARD, nil, params, 0)
    let _graph = graph._graph
    let _streamContext = streamContext?._stream
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(_graph, cmd, ccv_nnc_no_hint, 0, nil, 0, &_output, 1, 0, _streamContext)
  }
}

public extension DynamicGraph.Tensor {
  func toGPU(_ ordinal: Int = 0, streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    var _input: ccv_nnc_tensor_variable_t? = self._tensor
    let rawInput = self.rawValue
    let output: DynamicGraph.Tensor<Element> = graph.variable(.GPU(ordinal), format: rawInput.format, dimensions: rawInput.dimensions)
    var _output: ccv_nnc_tensor_variable_t? = output._tensor
    let _graph = graph._graph
    let _streamContext = streamContext?._stream
    ccv_nnc_dynamic_graph_exec(_graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
    return output
  }

  func toCPU(streamContext: StreamContext? = nil) -> DynamicGraph.Tensor<Element> {
    var params = CmdParamsFactory.factory.newParams()
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
