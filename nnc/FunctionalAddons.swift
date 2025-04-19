import C_nnc

public enum ReduceOp {
  case sum
  case mean
  case max
  case min
  case norm2
}

extension Functional {
  /// Element-wise addition
  public static func sum<T: DynamicGraph.TensorGroup>(
    _ inputs: [T], streamContext: StreamContext? = nil
  ) -> T {
    precondition(inputs.count >= 2)
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: inputs[0], Array(inputs.suffix(from: 1)),
      outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Element-wise addition
  public static func sum<T: DynamicGraph.TensorGroup>(
    _ inputs: T..., streamContext: StreamContext? = nil
  ) -> T {
    return sum(inputs, streamContext: streamContext)
  }

  /// Broadcast element-wise multiplication
  public static func mul<T: DynamicGraph.TensorGroup>(
    left: T, right: T, scalar: Float32 = 1, streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
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
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (leftScalar, rightScalar, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_ADD_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: left, right, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Element-wise division
  public static func div<T: DynamicGraph.TensorGroup>(
    left: T, right: T, streamContext: StreamContext? = nil
  ) -> T {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWDIV_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: left, right, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Element-wise reciprocal
  public static func reciprocal<T: DynamicGraph.TensorGroup>(
    _ one: T, streamContext: StreamContext? = nil
  )
    -> T
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWDIV_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: nil as T?, one, outputSize: 1,
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

  /// Element-wise exp
  public static func exp<T: DynamicGraph.TensorGroup>(_ one: T, streamContext: StreamContext? = nil)
    -> T
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWEXP_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Element-wise square root.
  public static func squareRoot<T: DynamicGraph.TensorGroup>(
    _ one: T, streamContext: StreamContext? = nil
  )
    -> T
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWSQRT_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Element-wise absolute.
  public static func abs<T: DynamicGraph.TensorGroup>(
    _ one: T, streamContext: StreamContext? = nil
  )
    -> T
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_EWABS_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Softmax activation
  public static func softmax<T: DynamicGraph.TensorGroup>(
    _ one: T, streamContext: StreamContext? = nil
  )
    -> T
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_SOFTMAX_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// ReLU activation
  public static func ReLU<T: DynamicGraph.TensorGroup>(
    _ one: T, streamContext: StreamContext? = nil
  )
    -> T
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_RELU_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Sigmoid activation
  public static func sigmoid<T: DynamicGraph.TensorGroup>(
    _ one: T, streamContext: StreamContext? = nil
  )
    -> T
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_SIGMOID_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Tanh activation
  public static func tanh<T: DynamicGraph.TensorGroup>(
    _ one: T, streamContext: StreamContext? = nil
  )
    -> T
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_TANH_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Swish activation
  public static func swish<T: DynamicGraph.TensorGroup>(
    _ one: T, streamContext: StreamContext? = nil
  )
    -> T
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_SWISH_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Average pool
  public static func averagePool<T: DynamicGraph.TensorGroup>(
    _ one: T, filterSize: [Int], hint: Hint = Hint(), streamContext: StreamContext? = nil
  )
    -> T
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim.0 = Int32(filterSize[0])
    params.size.dim.1 = Int32(filterSize[1])
    params.size.dim.2 = 1
    let cmd = ccv_nnc_cmd(CCV_NNC_AVERAGE_POOL_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: hint.toCHint(), inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Max pool
  public static func maxPool<T: DynamicGraph.TensorGroup>(
    _ one: T, filterSize: [Int], hint: Hint = Hint(), streamContext: StreamContext? = nil
  )
    -> T
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim.0 = Int32(filterSize[0])
    params.size.dim.1 = Int32(filterSize[1])
    params.size.dim.2 = 1
    let cmd = ccv_nnc_cmd(CCV_NNC_MAX_POOL_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: hint.toCHint(), inputs: one, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Argmax
  public static func argmax(
    _ one: DynamicGraph.AnyTensor, axis: Int, streamContext: StreamContext? = nil
  )
    -> DynamicGraph.Tensor<Int32>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.reduce.axis.0 = Int32(axis)
    params.reduce.count = 1
    let cmd = ccv_nnc_cmd(CCV_NNC_ARGMAX_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Int32>(outputs[0])
  }

  /// Argmax
  public static func argmax(
    _ one: DynamicGraph.Group<DynamicGraph.AnyTensor>, axis: Int,
    streamContext: StreamContext? = nil
  )
    -> DynamicGraph.Group<DynamicGraph.Tensor<Int32>>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.reduce.axis.0 = Int32(axis)
    params.reduce.count = 1
    let cmd = ccv_nnc_cmd(CCV_NNC_ARGMAX_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Group<DynamicGraph.Tensor<Int32>>(outputs[0])
  }

  /// Argmin
  public static func argmin(
    _ one: DynamicGraph.AnyTensor, axis: Int, streamContext: StreamContext? = nil
  )
    -> DynamicGraph.Tensor<Int32>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.reduce.axis.0 = Int32(axis)
    params.reduce.count = 1
    let cmd = ccv_nnc_cmd(CCV_NNC_ARGMIN_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Int32>(outputs[0])
  }

  /// Argmax
  public static func argmin(
    _ one: DynamicGraph.Group<DynamicGraph.AnyTensor>, axis: Int,
    streamContext: StreamContext? = nil
  )
    -> DynamicGraph.Group<DynamicGraph.Tensor<Int32>>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.reduce.axis.0 = Int32(axis)
    params.reduce.count = 1
    let cmd = ccv_nnc_cmd(CCV_NNC_ARGMIN_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: one, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Group<DynamicGraph.Tensor<Int32>>(outputs[0])
  }

  public struct GEMMFlag: OptionSet {
    public let rawValue: Int32
    public init(rawValue: Int32) {
      self.rawValue = rawValue
    }
    public static let Float32 = GEMMFlag(rawValue: Int32(CCV_NNC_GEMM_32F))
    public static let Float16 = GEMMFlag(rawValue: Int32(CCV_NNC_GEMM_16F))
    public static let TF32 = GEMMFlag(rawValue: Int32(CCV_NNC_GEMM_32TF))
  }

  /// Matrix multiplication
  public static func matmul<T: DynamicGraph.TensorGroup>(
    left: T, right: T, leftTranspose: (Int, Int) = (0, 0), rightTranspose: (Int, Int) = (0, 0),
    flags: Functional.GEMMFlag = [], streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (1, 1, 0)
    params.blas.transpose_a = (Int32(leftTranspose.0), Int32(leftTranspose.1))
    params.blas.transpose_b = (Int32(rightTranspose.0), Int32(rightTranspose.1))
    params.blas.flags = Int32(flags.rawValue)
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
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (left, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: right, outputSize: 1, streamContext: streamContext)
    return T(outputs[0])
  }

  /// Complex number multiplication
  public static func cmul<T: DynamicGraph.TensorGroup>(
    left: T, right: T, streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_CMUL_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: left, right, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Make a copy.
  public static func copy<T: DynamicGraph.TensorGroup>(
    from: T, to: T, streamContext: StreamContext? = nil
  ) {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, params, 0)
    exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: from, outputs: [to], streamContext: streamContext)
  }

  /// Select input tensor with another index tensor.
  public static func indexSelect<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    input: T, index: U, streamContext: StreamContext? = nil
  ) -> T where U.ElementNumeric == Int32, T.AnyTensor == U.AnyTensor {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_INDEX_SELECT_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, index, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Select input tensor with another index tensor.
  public static func indexSelect<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    input: T, index: U, streamContext: StreamContext? = nil
  ) -> T where U.ElementNumeric == Float32, T.AnyTensor == U.AnyTensor {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_INDEX_SELECT_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, index, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Masked fill a tensor based on other tensor's content equal to another.
  public static func maskedFill<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    input: T, mask: U, equalTo: Float, fillWith: Float, streamContext: StreamContext? = nil
  ) -> T where T.AnyTensor == U.AnyTensor {
    var params = CmdParamsFactory.factory.newParams()
    params.blas.a = (equalTo, fillWith, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_MASKED_FILL_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, mask, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Element-wise min for two input tensors
  public static func min<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_MIN_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: left, right, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Element-wise max for two input tensors
  public static func max<T: DynamicGraph.TensorGroup>(
    _ left: T, _ right: T, streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_MAX_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: left, right, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }

  /// Scatter add tensor with a index tensor.
  public static func scatterAdd<T: DynamicGraph.TensorGroup, U: DynamicGraph.TensorGroup>(
    count: Int, input: T, index: U, streamContext: StreamContext? = nil
  ) -> T where U.ElementNumeric == Int32, T.AnyTensor == U.AnyTensor {
    var params = CmdParamsFactory.factory.newParams()
    params.scatter_add.bincount = Int32(count)
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SCATTER_ADD_FORWARD, nil, params, 0)
    let outputs = exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, index, outputSize: 1,
      streamContext: streamContext)
    return T(outputs[0])
  }
}

extension DynamicGraph.Tensor {
  public subscript(ranges: [Range<Int>], streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get {
      precondition(ranges.count < CCV_NNC_MAX_DIM_ALLOC)
      let shape = self.shape
      precondition(ranges.count == shape.count)
      let offset = TensorShape(ranges.map { $0.lowerBound })
      let newShape = TensorShape(ranges.map { $0.count })
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < shape[i])
        assert(range.upperBound > 0 && range.upperBound <= shape[i])
      }
      let strides = self.strides
      precondition(ranges.count == strides.count)
      return reshaped(
        format: format, shape: newShape, offset: offset, strides: strides)
    }
    set(v) {
      precondition(v.graph === graph)
      precondition(ranges.count < CCV_NNC_MAX_DIM_ALLOC)
      let shape = self.shape
      precondition(ranges.count == shape.count)
      let offset = TensorShape(ranges.map { $0.lowerBound })
      let newShape = TensorShape(ranges.map { $0.count })
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < shape[i])
        assert(range.upperBound > 0 && range.upperBound <= shape[i])
      }
      let strides = self.strides
      precondition(ranges.count == strides.count)
      // Intentionally use the format of the input so we don't do unnecessary format conversion.
      let output = reshaped(
        format: v.format, shape: newShape, offset: offset, strides: strides
      )
      let params = CmdParamsFactory.factory.newParams()
      let cmd = ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, params, 0)
      let _graph = graph.cGraph
      let _streamContext = (streamContext ?? graph.streamContext)?._stream
      var _input: ccv_nnc_tensor_variable_t? = v._tensor
      var _output: ccv_nnc_tensor_variable_t? = output._tensor
      ccv_nnc_dynamic_graph_exec(
        _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
      withExtendedLifetime((v, output)) {}
    }
  }

  @inlinable
  public subscript(ranges: Range<Int>..., streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get { self[ranges, streamContext: streamContext] }
    set(v) { self[ranges, streamContext: streamContext] = v }
  }

  @usableFromInline
  subscript(indices: [Int], range: Range<Int>, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get {
      precondition(indices.count + 1 < CCV_NNC_MAX_DIM_ALLOC)
      let shape = self.shape
      precondition(indices.count + 1 == shape.count)
      let offset = TensorShape(indices + [range.lowerBound])
      let newShape = TensorShape(Array(repeating: 1, count: indices.count) + [range.count])
      assert(range.lowerBound >= 0 && range.lowerBound < shape[indices.count])
      let strides = self.strides
      return reshaped(
        format: format, shape: newShape, offset: offset, strides: strides)
    }
    set(v) {
      precondition(v.graph === graph)
      precondition(indices.count + 1 < CCV_NNC_MAX_DIM_ALLOC)
      let shape = self.shape
      precondition(indices.count + 1 == shape.count)
      let offset = TensorShape(indices + [range.lowerBound])
      let newShape = TensorShape(Array(repeating: 1, count: indices.count) + [range.count])
      assert(range.lowerBound >= 0 && range.lowerBound < shape[indices.count])
      let strides = self.strides
      // Intentionally use the format of the input so we don't do unnecessary format conversion.
      let output = reshaped(
        format: v.format, shape: newShape, offset: offset, strides: strides
      )
      let params = CmdParamsFactory.factory.newParams()
      let cmd = ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, params, 0)
      let _graph = graph.cGraph
      let _streamContext = (streamContext ?? graph.streamContext)?._stream
      var _input: ccv_nnc_tensor_variable_t? = v._tensor
      var _output: ccv_nnc_tensor_variable_t? = output._tensor
      ccv_nnc_dynamic_graph_exec(
        _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
      withExtendedLifetime((v, output)) {}
    }
  }

  @usableFromInline
  subscript(indices: [Int], range: UnboundedRange, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get {
      let shape = self.shape
      return self[indices, 0..<shape[indices.count], streamContext: streamContext]
    }
    set(v) {
      let shape = self.shape
      self[indices, 0..<shape[indices.count], streamContext: streamContext] = v
    }
  }

  @inlinable
  public subscript(range: Range<Int>, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get { self[[], range, streamContext: streamContext] }
    set { self[[], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, range: Range<Int>, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get { self[[i0], range, streamContext: streamContext] }
    set { self[[i0], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, range: Range<Int>, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get { self[[i0, i1], range, streamContext: streamContext] }
    set { self[[i0, i1], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  ) -> DynamicGraph.Tensor<Element> {
    get { self[[i0, i1, i2], range, streamContext: streamContext] }
    set { self[[i0, i1, i2], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Tensor<Element>
  {
    get { self[[i0, i1, i2, i3], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Tensor<
      Element
    >
  {
    get { self[[i0, i1, i2, i3, i4], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Tensor<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4, i5], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Tensor<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5, i6], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4, i5, i6], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(range: UnboundedRange, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get { self }
    set { self[[], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, range: UnboundedRange, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get { self[[i0], range, streamContext: streamContext] }
    set { self[[i0], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  ) -> DynamicGraph.Tensor<Element> {
    get { self[[i0, i1], range, streamContext: streamContext] }
    set { self[[i0, i1], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  ) -> DynamicGraph.Tensor<Element>
  {
    get { self[[i0, i1, i2], range, streamContext: streamContext] }
    set { self[[i0, i1, i2], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Tensor<Element>
  {
    get { self[[i0, i1, i2, i3], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Tensor<
      Element
    >
  {
    get { self[[i0, i1, i2, i3, i4], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Tensor<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4, i5], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int,
    range: UnboundedRange, streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Tensor<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5, i6], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4, i5, i6], range, streamContext: streamContext] = newValue }
  }
}

extension DynamicGraph.Group where Element: DynamicGraph.AnyTensor {
  public subscript(ranges: [Range<Int>], streamContext streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    get {
      precondition(ranges.count < CCV_NNC_MAX_DIM_ALLOC)
      let shape = self.shape
      precondition(ranges.count == shape.count)
      let offset = TensorShape(ranges.map { $0.lowerBound })
      let newShape = TensorShape(ranges.map { $0.count })
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < shape[i])
        assert(range.upperBound > 0 && range.upperBound <= shape[i])
      }
      let strides = self.strides
      precondition(ranges.count == strides.count)
      return reshaped(
        format: format, shape: newShape, offset: offset, strides: strides)
    }
    set(v) {
      precondition(v.count == count)
      guard count > 0 else { return }
      let graph = untyped[0].graph
      for x in v.untyped {
        precondition(x.graph === graph)
      }
      precondition(ranges.count < CCV_NNC_MAX_DIM_ALLOC)
      let shape = self.shape
      precondition(ranges.count == shape.count)
      let offset = TensorShape(ranges.map { $0.lowerBound })
      let newShape = TensorShape(ranges.map { $0.count })
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < shape[i])
        assert(range.upperBound > 0 && range.upperBound <= shape[i])
      }
      let strides = self.strides
      precondition(ranges.count == strides.count)
      // Intentionally use the format of the input so we don't do unnecessary format conversion.
      let outputs = reshaped(
        format: v.format, shape: newShape, offset: offset, strides: strides
      )
      let params = CmdParamsFactory.factory.newParams()
      let cmd = ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, params, 0)
      let _graph = graph.cGraph
      let _streamContext = (streamContext ?? graph.streamContext)?._stream
      let _inputs: [ccv_nnc_tensor_variable_t?] = v.untyped.map { $0._tensor }
      let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
        capacity: count)
      for (i, variable) in outputs.untyped.enumerated() {
        (_outputs + i).initialize(to: variable._tensor)
      }
      let outputSize = Int32(count)
      ccv_nnc_dynamic_graph_exec(
        _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize, _outputs, outputSize, outputSize,
        _streamContext)
      _outputs.deallocate()
      withExtendedLifetime((v, outputs)) {}
    }
  }

  @inlinable
  public subscript(ranges: Range<Int>..., streamContext streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    get { self[ranges, streamContext: streamContext] }
    set(v) { self[ranges, streamContext: streamContext] = v }
  }

  @usableFromInline
  subscript(indices: [Int], range: Range<Int>, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    get {
      precondition(indices.count + 1 < CCV_NNC_MAX_DIM_ALLOC)
      let shape = self.shape
      precondition(indices.count + 1 == shape.count)
      let offset = TensorShape(indices + [range.lowerBound])
      let newShape = TensorShape(Array(repeating: 1, count: indices.count) + [range.count])
      assert(range.lowerBound >= 0 && range.lowerBound < shape[indices.count])
      let strides = self.strides
      return reshaped(
        format: format, shape: newShape, offset: offset, strides: strides)
    }
    set(v) {
      precondition(v.count == count)
      guard count > 0 else { return }
      let graph = untyped[0].graph
      for x in v.untyped {
        precondition(x.graph === graph)
      }
      precondition(indices.count + 1 < CCV_NNC_MAX_DIM_ALLOC)
      let shape = self.shape
      precondition(indices.count + 1 == shape.count)
      let offset = TensorShape(indices + [range.lowerBound])
      let newShape = TensorShape(Array(repeating: 1, count: indices.count) + [range.count])
      assert(range.lowerBound >= 0 && range.lowerBound < shape[indices.count])
      let strides = self.strides
      // Intentionally use the format of the input so we don't do unnecessary format conversion.
      let outputs = reshaped(
        format: v.format, shape: newShape, offset: offset, strides: strides
      )
      let params = CmdParamsFactory.factory.newParams()
      let cmd = ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, params, 0)
      let _graph = graph.cGraph
      let _streamContext = (streamContext ?? graph.streamContext)?._stream
      let _inputs: [ccv_nnc_tensor_variable_t?] = v.untyped.map { $0._tensor }
      let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
        capacity: count)
      for (i, variable) in outputs.untyped.enumerated() {
        (_outputs + i).initialize(to: variable._tensor)
      }
      let outputSize = Int32(count)
      ccv_nnc_dynamic_graph_exec(
        _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize, _outputs, outputSize, outputSize,
        _streamContext)
      _outputs.deallocate()
      withExtendedLifetime((v, outputs)) {}
    }
  }

  @usableFromInline
  subscript(indices: [Int], range: UnboundedRange, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    get {
      let shape = self.shape
      return self[indices, 0..<shape[indices.count], streamContext: streamContext]
    }
    set(v) {
      let shape = self.shape
      self[indices, 0..<shape[indices.count], streamContext: streamContext] = v
    }
  }

  @inlinable
  public subscript(range: Range<Int>, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    get { self[[], range, streamContext: streamContext] }
    set { self[[], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, range: Range<Int>, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    get { self[[i0], range, streamContext: streamContext] }
    set { self[[i0], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, range: Range<Int>, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    get { self[[i0, i1], range, streamContext: streamContext] }
    set { self[[i0, i1], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  ) -> DynamicGraph.Group<Element> {
    get { self[[i0, i1, i2], range, streamContext: streamContext] }
    set { self[[i0, i1, i2], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Group<Element>
  {
    get { self[[i0, i1, i2, i3], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Group<
      Element
    >
  {
    get { self[[i0, i1, i2, i3, i4], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Group<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4, i5], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, range: Range<Int>,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Group<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5, i6], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4, i5, i6], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(range: UnboundedRange, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    get { self }
    set { self[[], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, range: UnboundedRange, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    get { self[[i0], range, streamContext: streamContext] }
    set { self[[i0], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  ) -> DynamicGraph.Group<Element> {
    get { self[[i0, i1], range, streamContext: streamContext] }
    set { self[[i0, i1], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  ) -> DynamicGraph.Group<Element>
  {
    get { self[[i0, i1, i2], range, streamContext: streamContext] }
    set { self[[i0, i1, i2], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Group<Element>
  {
    get { self[[i0, i1, i2, i3], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Group<
      Element
    >
  {
    get { self[[i0, i1, i2, i3, i4], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, range: UnboundedRange,
    streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Group<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4, i5], range, streamContext: streamContext] = newValue }
  }

  @inlinable
  public subscript(i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int,
    range: UnboundedRange, streamContext streamContext: StreamContext?
  )
    -> DynamicGraph.Group<Element>
  {
    get { self[[i0, i1, i2, i3, i4, i5, i6], range, streamContext: streamContext] }
    set { self[[i0, i1, i2, i3, i4, i5, i6], range, streamContext: streamContext] = newValue }
  }
}

extension DynamicGraph.Tensor {
  /// Transpose from axisA to axisB.
  public func transposed(_ axisA: Int, _ axisB: Int, streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.transpose.axis = (Int32(axisA), Int32(axisB))
    let cmd = ccv_nnc_cmd(CCV_NNC_TRANSPOSE_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

extension DynamicGraph.Group {
  /// Transpose from axisA to axisB.
  public func transposed(_ axisA: Int, _ axisB: Int, streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
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
    _ range: ClosedRange<Float>, streamContext: StreamContext?
  ) {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (range.lowerBound, range.upperBound, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_RANDOM_UNIFORM_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, nil, 0, &_output, 1, 0, _streamContext)
  }
}

extension DynamicGraph.Group {
  /// Fill the given tensor with uniform random values.
  public func rand(
    _ range: ClosedRange<Float>, streamContext: StreamContext?
  ) {
    guard underlyingArray.count > 0 else { return }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (range.lowerBound, range.upperBound, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_RANDOM_UNIFORM_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
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
  /// Fill the given tensor with normal-distributed random values.
  public func randn(
    std: Float, mean: Float, streamContext: StreamContext?
  ) {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (std, mean, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_RANDOM_NORMAL_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, nil, 0, &_output, 1, 0, _streamContext)
  }
}

extension DynamicGraph.Group {
  /// Fill the given tensor with normal-distributed random values.
  public func randn(
    std: Float, mean: Float, streamContext: StreamContext?
  ) {
    guard underlyingArray.count > 0 else { return }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (std, mean, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_RANDOM_NORMAL_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
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
  public func toGPU(_ ordinal: Int, streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    guard kind != .GPU(ordinal) else { return self }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    var _input: ccv_nnc_tensor_variable_t? = self._tensor
    let rawInput = self.rawValue
    let output: DynamicGraph.Tensor<Element> = graph.variable(
      .GPU(ordinal), format: rawInput.format, shape: rawInput.shape)
    var _output: ccv_nnc_tensor_variable_t? = output._tensor
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
    return output
  }

  /// Copy the given tensor to CPU.
  public func toCPU(streamContext: StreamContext?) -> DynamicGraph.Tensor<Element> {
    guard kind != .CPU else { return self }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    var _input: ccv_nnc_tensor_variable_t? = self._tensor
    let rawInput = self.rawValue
    let output: DynamicGraph.Tensor<Element> = graph.variable(
      .CPU, format: rawInput.format, shape: rawInput.shape)
    var _output: ccv_nnc_tensor_variable_t? = output._tensor
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
    return output
  }
}

extension DynamicGraph.Group
where Element: DynamicGraph.TensorGroup, Element: DynamicGraph.AnyTensor {
  /// Copy the given tensor to GPU.
  public func toGPU(_ ordinal: Int, streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    fatalError(
      "toGPU() cannot be reasonably implemented for Group as Group would be most effective to resides on different GPUs"
    )
  }

  /// Copy the given tensor to CPU.
  public func toCPU(streamContext: StreamContext?) -> DynamicGraph.Group<Element> {
    guard underlyingArray.count > 0 else { return self }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    let _inputs: [ccv_nnc_tensor_variable_t?] = underlyingArray.map { $0._tensor }
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let outputSize = Int32(underlyingArray.count)
    let outputs: [DynamicGraph.Tensor<ElementNumeric>] = untyped.map {
      graph.variable(.CPU, format: $0.format, shape: $0.shape)
    }
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    for (i, variable) in outputs.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize, _outputs, outputSize, outputSize,
      _streamContext)
    _outputs.deallocate()
    return DynamicGraph.Group(outputs) as! DynamicGraph.Group<Element>
  }
}

extension DynamicGraph.Tensor {
  /// Fill the given tensor with a value.
  public func full(_ value: Float, streamContext: StreamContext?) {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (value, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SET_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, nil, 0, &_output, 1, 0, _streamContext)
  }
}

extension DynamicGraph.Group {
  /// Fill the given tensor with a value.
  public func full(_ value: Float, streamContext: StreamContext?) {
    guard underlyingArray.count > 0 else { return }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (value, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SET_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let outputSize = Int32(underlyingArray.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    for (i, variable) in underlyingArray.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, nil, 0, _outputs, outputSize, outputSize, _streamContext)
    _outputs.deallocate()
  }
}

extension DynamicGraph.Tensor {
  /// Interpolate from this tensor to the other tensor.
  public func lerp(
    _ weight: Float, to: DynamicGraph.Tensor<Element>, streamContext: StreamContext?
  ) {
    precondition(weight >= 0 && weight <= 1)
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (1 - weight, weight, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_ADD_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let _inputs: [ccv_nnc_tensor_variable_t?] = [_tensor, to._tensor]
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, _inputs, 2, &_output, 1, 0, _streamContext)
  }
}

extension DynamicGraph.Group {
  /// Interpolate from this tensor to the other tensor.
  public func lerp(
    _ weight: Float, to: DynamicGraph.Group<Element>, streamContext: StreamContext?
  ) {
    precondition(weight >= 0 && weight <= 1)
    guard underlyingArray.count > 0 else { return }
    precondition(to.underlyingArray.count == underlyingArray.count)
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (1 - weight, weight, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_ADD_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let _inputs: [ccv_nnc_tensor_variable_t?] = zip(underlyingArray, to.underlyingArray).flatMap {
      [$0.0._tensor, $0.1._tensor]
    }
    let outputSize = Int32(underlyingArray.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    for (i, variable) in underlyingArray.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize * 2, _outputs, outputSize, outputSize,
      _streamContext)
    _outputs.deallocate()
  }
}

extension DynamicGraph.Tensor {
  func clamp(
    min: Float?, max: Float?, streamContext: StreamContext?
  ) {
    precondition(min != nil || max != nil)
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.clamp.min = min ?? Float.nan
    params.clamp.max = max ?? Float.nan
    let cmd = ccv_nnc_cmd(CCV_NNC_CLAMP_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _input: ccv_nnc_tensor_variable_t? = _tensor
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
  }

  /// Clamp the given tensor between two values.
  public func clamp(_ range: ClosedRange<Float>, streamContext: StreamContext?) {
    clamp(min: range.lowerBound, max: range.upperBound, streamContext: streamContext)
  }

  /// Clamp the given tensor with a lower bound.
  public func clamp(_ range: PartialRangeFrom<Float>, streamContext: StreamContext?) {
    clamp(min: range.lowerBound, max: nil, streamContext: streamContext)
  }

  /// Clamp the given tensor with an upper bound.
  public func clamp(_ range: PartialRangeThrough<Float>, streamContext: StreamContext?) {
    clamp(min: nil, max: range.upperBound, streamContext: streamContext)
  }
}

extension DynamicGraph.Group {
  func clamp(
    min: Float?, max: Float?, streamContext: StreamContext?
  ) {
    guard underlyingArray.count > 0 else { return }
    precondition(min != nil || max != nil)
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.clamp.min = min ?? Float.nan
    params.clamp.max = max ?? Float.nan
    let cmd = ccv_nnc_cmd(CCV_NNC_CLAMP_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let _inputs: [ccv_nnc_tensor_variable_t?] = underlyingArray.map { $0._tensor }
    let outputSize = Int32(underlyingArray.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    for (i, variable) in underlyingArray.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize, _outputs, outputSize, outputSize,
      _streamContext)
    _outputs.deallocate()
  }

  /// Clamp the given tensor between two values.
  public func clamp(_ range: ClosedRange<Float>, streamContext: StreamContext?) {
    clamp(min: range.lowerBound, max: range.upperBound, streamContext: streamContext)
  }

  /// Clamp the given tensor with a lower bound.
  public func clamp(_ range: PartialRangeFrom<Float>, streamContext: StreamContext?) {
    clamp(min: range.lowerBound, max: nil, streamContext: streamContext)
  }

  /// Clamp the given tensor with an upper bound.
  public func clamp(_ range: PartialRangeThrough<Float>, streamContext: StreamContext?) {
    clamp(min: nil, max: range.upperBound, streamContext: streamContext)
  }
}

extension DynamicGraph.Tensor {
  /// Detach current tensor from the graph. Afterwards, it is always "isConstant" and cannot requiresGrad.
  public func detach() {
    let _graph = graph.cGraph
    ccv_nnc_tensor_variable_detach(_graph, _tensor)
    requiresGrad = false
  }
}

extension DynamicGraph.Group {
  /// Detach tensors in this group from the graph. Afterwards, it is always "isConstant" and cannot requiresGrad.
  public mutating func detach() {
    for variable in underlyingArray {
      let _graph = variable.graph.cGraph
      ccv_nnc_tensor_variable_detach(_graph, variable._tensor)
    }
    requiresGrad = false
  }
}

extension DynamicGraph.Tensor {
  func clamped(
    min: Float?, max: Float?, streamContext: StreamContext?
  ) -> DynamicGraph.Tensor<Element> {
    precondition(min != nil || max != nil)
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.clamp.min = min ?? Float.nan
    params.clamp.max = max ?? Float.nan
    let cmd = ccv_nnc_cmd(CCV_NNC_CLAMP_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }

  /// Clamp the given tensor between two values.
  public func clamped(_ range: ClosedRange<Float>, streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    return clamped(min: range.lowerBound, max: range.upperBound, streamContext: streamContext)
  }

  /// Clamp the given tensor with a lower bound.
  public func clamped(_ range: PartialRangeFrom<Float>, streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    return clamped(min: range.lowerBound, max: nil, streamContext: streamContext)
  }

  /// Clamp the given tensor with an upper bound.
  public func clamped(_ range: PartialRangeThrough<Float>, streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    return clamped(min: nil, max: range.upperBound, streamContext: streamContext)
  }
}

extension DynamicGraph.Group {
  func clamped(
    min: Float?, max: Float?, streamContext: StreamContext?
  ) -> DynamicGraph.Group<Element> {
    precondition(min != nil || max != nil)
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.clamp.min = min ?? Float.nan
    params.clamp.max = max ?? Float.nan
    let cmd = ccv_nnc_cmd(CCV_NNC_CLAMP_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Group<Element>(outputs[0])
  }

  /// Clamp the given tensor between two values.
  public func clamped(_ range: ClosedRange<Float>, streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    return clamped(min: range.lowerBound, max: range.upperBound, streamContext: streamContext)
  }

  /// Clamp the given tensor with a lower bound.
  public func clamped(_ range: PartialRangeFrom<Float>, streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    return clamped(min: range.lowerBound, max: nil, streamContext: streamContext)
  }

  /// Clamp the given tensor with an upper bound.
  public func clamped(_ range: PartialRangeThrough<Float>, streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    return clamped(min: nil, max: range.upperBound, streamContext: streamContext)
  }
}

extension DynamicGraph.Tensor {
  /// Make a copy of the existing tensor.
  public func copied(streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

extension DynamicGraph.Group {
  /// Make a copy of the existing tensor group.
  public func copied(streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Group<Element>(outputs[0])
  }
}

extension DynamicGraph.Tensor {
  /// Only make a copy of the existing tensor if it is not contiguous in memory.
  public func contiguous(streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    guard !isContiguous else {
      return self
    }
    return copied(streamContext: streamContext)
  }
}

extension DynamicGraph.Group {
  /// Only make a copy of the existing tensor group if it is not contiguous in memory.
  public func contiguous(streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    guard !isContiguous else {
      return self
    }
    return copied(streamContext: streamContext)
  }
}

extension DynamicGraph.Tensor {
  /// Reduce along a given dimension.
  public func reduced(_ op: ReduceOp, axis: [Int], streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.reduce.axis = toCDimensions(axis)
    params.reduce.count = Int32(axis.count)
    let cmd: ccv_nnc_cmd_t
    switch op {
    case .sum:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_SUM_FORWARD, nil, params, 0)
    case .mean:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_MEAN_FORWARD, nil, params, 0)
    case .max:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_MAX_FORWARD, nil, params, 0)
    case .min:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_MIN_FORWARD, nil, params, 0)
    case .norm2:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_NORM2_FORWARD, nil, params, 0)
    }
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Tensor<Element>(outputs[0])
  }
}

extension DynamicGraph.Group {
  /// Reduce along a given dimension.
  public func reduced(_ op: ReduceOp, axis: [Int], streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.reduce.axis = toCDimensions(axis)
    params.reduce.count = Int32(axis.count)
    let cmd: ccv_nnc_cmd_t
    switch op {
    case .sum:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_SUM_FORWARD, nil, params, 0)
    case .mean:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_MEAN_FORWARD, nil, params, 0)
    case .max:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_MAX_FORWARD, nil, params, 0)
    case .min:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_MIN_FORWARD, nil, params, 0)
    case .norm2:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_NORM2_FORWARD, nil, params, 0)
    }
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 1, streamContext: streamContext)
    return DynamicGraph.Group<Element>(outputs[0])
  }
}

extension DynamicGraph.Tensor {
  /// Scale the given tensor with a constant inplace.
  public func scale(by a: Float, streamContext: StreamContext?) {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (a, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _input: ccv_nnc_tensor_variable_t? = _tensor
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
  }

  /// Scale the given tensor with a constant.
  public func scaled(by a: Float, streamContext: StreamContext?) -> Self {
    Functional.scalmul(left: a, right: self, streamContext: streamContext)
  }
}

extension DynamicGraph.Group {
  /// Scale the given tensor with a constant inplace.
  public func scale(by a: Float, streamContext: StreamContext?) {
    guard underlyingArray.count > 0 else { return }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (a, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let _inputs: [ccv_nnc_tensor_variable_t?] = underlyingArray.map { $0._tensor }
    let outputSize = Int32(underlyingArray.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    for (i, variable) in underlyingArray.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize, _outputs, outputSize, outputSize,
      _streamContext)
    _outputs.deallocate()
  }
}

extension DynamicGraph.Group where Element: DynamicGraph_TensorGroup {
  /// Scale the given tensor with a constant.
  public func scaled(by a: Float, streamContext: StreamContext?) -> Self {
    Functional.scalmul(left: a, right: self, streamContext: streamContext)
  }
}

extension DynamicGraph.Tensor {
  /// Apply softmax activation to the given tensor inplace.
  public func softmax(streamContext: StreamContext?) {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_SOFTMAX_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _input: ccv_nnc_tensor_variable_t? = _tensor
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
  }

}

extension DynamicGraph.Group {
  /// Apply softmax activation to the given tensor inplace.
  public func softmax(streamContext: StreamContext?) {
    guard underlyingArray.count > 0 else { return }
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_SOFTMAX_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let _inputs: [ccv_nnc_tensor_variable_t?] = underlyingArray.map { $0._tensor }
    let outputSize = Int32(underlyingArray.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    for (i, variable) in underlyingArray.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize, _outputs, outputSize, outputSize,
      _streamContext)
    _outputs.deallocate()
  }
}

extension DynamicGraph.Tensor {
  /// Apply ReLU activation to the given tensor inplace.
  public func ReLU(streamContext: StreamContext?) {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_RELU_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _input: ccv_nnc_tensor_variable_t? = _tensor
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
  }

}

extension DynamicGraph.Group {
  /// Apply ReLU activation to the given tensor inplace.
  public func ReLU(streamContext: StreamContext?) {
    guard underlyingArray.count > 0 else { return }
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_RELU_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let _inputs: [ccv_nnc_tensor_variable_t?] = underlyingArray.map { $0._tensor }
    let outputSize = Int32(underlyingArray.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    for (i, variable) in underlyingArray.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize, _outputs, outputSize, outputSize,
      _streamContext)
    _outputs.deallocate()
  }
}

extension DynamicGraph.Tensor {
  /// Apply sigmoid activation to the given tensor inplace.
  public func sigmoid(streamContext: StreamContext?) {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_SIGMOID_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _input: ccv_nnc_tensor_variable_t? = _tensor
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
  }

}

extension DynamicGraph.Group {
  /// Apply sigmoid activation to the given tensor inplace.
  public func sigmoid(streamContext: StreamContext?) {
    guard underlyingArray.count > 0 else { return }
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_SIGMOID_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let _inputs: [ccv_nnc_tensor_variable_t?] = underlyingArray.map { $0._tensor }
    let outputSize = Int32(underlyingArray.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    for (i, variable) in underlyingArray.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize, _outputs, outputSize, outputSize,
      _streamContext)
    _outputs.deallocate()
  }
}

extension DynamicGraph.Tensor {
  /// Apply tanh activation to the given tensor inplace.
  public func tanh(streamContext: StreamContext?) {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_TANH_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _input: ccv_nnc_tensor_variable_t? = _tensor
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
  }

}

extension DynamicGraph.Group {
  /// Apply tanh activation to the given tensor inplace.
  public func tanh(streamContext: StreamContext?) {
    guard underlyingArray.count > 0 else { return }
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_TANH_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let _inputs: [ccv_nnc_tensor_variable_t?] = underlyingArray.map { $0._tensor }
    let outputSize = Int32(underlyingArray.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    for (i, variable) in underlyingArray.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize, _outputs, outputSize, outputSize,
      _streamContext)
    _outputs.deallocate()
  }
}

extension DynamicGraph.Tensor {
  /// Apply swish activation to the given tensor inplace.
  public func swish(streamContext: StreamContext?) {
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_SWISH_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    var _input: ccv_nnc_tensor_variable_t? = _tensor
    var _output: ccv_nnc_tensor_variable_t? = _tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
  }
}

extension DynamicGraph.Group {
  /// Apply swish activation to the given tensor inplace.
  public func swish(streamContext: StreamContext?) {
    guard underlyingArray.count > 0 else { return }
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_SWISH_FORWARD, nil, params, 0)
    let graph = underlyingArray[0].graph
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let _inputs: [ccv_nnc_tensor_variable_t?] = underlyingArray.map { $0._tensor }
    let outputSize = Int32(underlyingArray.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: Int(outputSize))
    for (i, variable) in underlyingArray.enumerated() {
      (_outputs + i).initialize(to: variable._tensor)
    }
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, _inputs, outputSize, _outputs, outputSize, outputSize,
      _streamContext)
    _outputs.deallocate()
  }
}

extension DynamicGraph.Tensor {
  /// Explicitly do conversion between types.
  public convenience init(from input: DynamicGraph.AnyTensor, streamContext: StreamContext? = nil) {
    guard input.dataType != Element.dataType else {
      self.init(input)
      return
    }
    let params = CmdParamsFactory.factory.newParams()
    let cmd = ccv_nnc_cmd(CCV_NNC_DATATYPE_CONVERSION_FORWARD, nil, params, 0)
    let output = input.graph.variable(
      input.kind, format: input.format, shape: input.shape, of: Element.self)
    Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, outputs: [output],
      streamContext: streamContext)
    self.init(output)
  }
}

extension DynamicGraph.AnyTensor {
  /// Explicitly cast the underlying storage to a specific type. It is a helper function than
  /// doing DynamicGraph.Tensor<SomeType>(something). Less code change required if we change this
  /// from tensor to group.
  public func `as`<Element: TensorNumeric>(of: Element.Type) -> DynamicGraph.Tensor<Element> {
    let result = DynamicGraph.Tensor<Element>(self)
    assert(result.dataType == Element.dataType)
    return result
  }
}

extension DynamicGraph.Group where Element == DynamicGraph.AnyTensor {
  /// Explicitly cast the underlying storage to a specific type. It is a helper function than
  /// doing DynamicGraph.Group<DynamicGraph.Tensor<SomeType>>(something). Less code change required
  /// if we change this from group to tensor.
  public func `as`<T: TensorNumeric>(of: T.Type) -> DynamicGraph.Group<DynamicGraph.Tensor<T>> {
    let result = DynamicGraph.Group<DynamicGraph.Tensor<T>>(self)
    assert(result.dataType == T.dataType)
    return result
  }
}

extension DynamicGraph.AnyTensor {
  public var isNaN: Bool {
    var params = CmdParamsFactory.factory.newParams()
    params.reduce.count = Int32(shape.count)
    params.reduce.axis = toCDimensions(Array(0..<Int(params.reduce.count)))
    let cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_ISNAN_FORWARD, nil, params, 0)
    let _graph = graph.cGraph
    let output = graph.variable(kind, format: format, shape: [1], of: Int32.self)
    var _input: ccv_nnc_tensor_variable_t? = _tensor
    var _output: ccv_nnc_tensor_variable_t? = output._tensor
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, nil)
    let isNaN = output.toCPU().rawValue[0]
    return isNaN != 0
  }
}

extension DynamicGraph.Tensor {
  public func chunked(_ numberOfChunks: Int, axis: Int, streamContext: StreamContext?)
    -> [DynamicGraph.Tensor<Element>]
  {
    var shape = shape
    precondition(axis < shape.count)
    precondition((shape[axis] % numberOfChunks) == 0)
    shape[axis] = shape[axis] / numberOfChunks
    var offset = TensorShape([])
    let strides = strides
    let format = format
    return (0..<numberOfChunks).map {
      offset[axis] = shape[axis] * $0
      return reshaped(format: format, shape: shape, offset: offset, strides: strides)
    }
  }
}

extension DynamicGraph.Group where Element: DynamicGraph.AnyTensor {
  public func chunked(_ numberOfChunks: Int, axis: Int, streamContext: StreamContext?) -> [Self] {
    var shape = shape
    precondition(axis < shape.count)
    precondition((shape[axis] % numberOfChunks) == 0)
    shape[axis] = shape[axis] / numberOfChunks
    let result = underlyingArray.map { tensor in
      var offset = TensorShape([])
      let strides = tensor.strides
      return (0..<numberOfChunks).map {
        offset[axis] = shape[axis] * $0
        return tensor.reshaped(format: format, shape: shape, offset: offset, strides: strides)
      }
    }
    return (0..<numberOfChunks).map { index in
      DynamicGraph.Group(result.map { $0[index] })
    }
  }
}

extension DynamicGraph.Tensor {
  /// Sort along a given dimension.
  public func sorted(axis: Int, descending: Bool, streamContext: StreamContext? = nil)
    -> (DynamicGraph.Tensor<Element>, indices: DynamicGraph.Tensor<Int32>)
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.sort.along_axis = Int32(axis)
    params.sort.descending = descending ? 1 : 0
    let cmd = ccv_nnc_cmd(CCV_NNC_SORT_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 2, streamContext: streamContext)
    return (
      DynamicGraph.Tensor<Element>(outputs[0]), indices: DynamicGraph.Tensor<Int32>(outputs[1])
    )
  }
}

extension DynamicGraph.Group {
  /// Sort along a given dimension.
  public func sorted(axis: Int, descending: Bool, streamContext: StreamContext? = nil)
    -> (DynamicGraph.Group<Element>, indices: DynamicGraph.Group<DynamicGraph.Tensor<Int32>>)
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.sort.along_axis = Int32(axis)
    params.sort.descending = descending ? 1 : 0
    let cmd = ccv_nnc_cmd(CCV_NNC_SORT_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 2, streamContext: streamContext)
    return (
      DynamicGraph.Group<Element>(outputs[0]),
      indices: DynamicGraph.Group<DynamicGraph.Tensor<Int32>>(outputs[1])
    )
  }
}

extension DynamicGraph.Tensor {
  /// Partition k items along a given dimension.
  public func partitioned(
    kth: Int, axis: Int, descending: Bool, streamContext: StreamContext? = nil
  )
    -> (DynamicGraph.Tensor<Element>, indices: DynamicGraph.Tensor<Int32>)
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.partition.kth = Int32(kth)
    params.partition.along_axis = Int32(axis)
    params.partition.descending = descending ? 1 : 0
    let cmd = ccv_nnc_cmd(CCV_NNC_PARTITION_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 2, streamContext: streamContext)
    return (
      DynamicGraph.Tensor<Element>(outputs[0]), indices: DynamicGraph.Tensor<Int32>(outputs[1])
    )
  }
}

extension DynamicGraph.Group {
  /// Partition k items along a given dimension.
  public func partitioned(
    kth: Int, axis: Int, descending: Bool, streamContext: StreamContext? = nil
  )
    -> (DynamicGraph.Group<Element>, indices: DynamicGraph.Group<DynamicGraph.Tensor<Int32>>)
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.partition.kth = Int32(kth)
    params.partition.along_axis = Int32(axis)
    params.partition.descending = descending ? 1 : 0
    let cmd = ccv_nnc_cmd(CCV_NNC_PARTITION_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 2, streamContext: streamContext)
    return (
      DynamicGraph.Group<Element>(outputs[0]),
      indices: DynamicGraph.Group<DynamicGraph.Tensor<Int32>>(outputs[1])
    )
  }
}

extension DynamicGraph.Tensor {
  /// Find unique consecutive elements and their counts.
  public func uniqueConsecutive(count: Int, streamContext: StreamContext? = nil)
    -> (DynamicGraph.Tensor<Element>, counts: DynamicGraph.Tensor<Int32>)
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.unique_consecutive.bincount = Int32(count)
    let cmd = ccv_nnc_cmd(CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 2, streamContext: streamContext)
    return (
      DynamicGraph.Tensor<Element>(outputs[0]), counts: DynamicGraph.Tensor<Int32>(outputs[1])
    )
  }
}

extension DynamicGraph.Group {
  /// Find unique consecutive elements and their counts.
  public func uniqueConsecutive(count: Int, streamContext: StreamContext? = nil)
    -> (DynamicGraph.Group<Element>, counts: DynamicGraph.Group<DynamicGraph.Tensor<Int32>>)
  {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.unique_consecutive.bincount = Int32(count)
    let cmd = ccv_nnc_cmd(CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD, nil, params, 0)
    let outputs = Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: self, outputSize: 2, streamContext: streamContext)
    return (
      DynamicGraph.Group<Element>(outputs[0]),
      counts: DynamicGraph.Group<DynamicGraph.Tensor<Int32>>(outputs[1])
    )
  }
}
