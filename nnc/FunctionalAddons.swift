import C_nnc

public enum ReduceOp {
  case sum
  case max
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

  /// Matrix multiplication
  public static func matmul<T: DynamicGraph.TensorGroup>(
    left: T, right: T, leftTranspose: (Int, Int) = (0, 0), rightTranspose: (Int, Int) = (0, 0),
    streamContext: StreamContext? = nil
  ) -> T {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
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
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
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
  ) -> T where U.ElementNumeric == Int32, T.AnyTensor == U.AnyTensor {
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
}

extension DynamicGraph.Tensor {
  public subscript(ranges: [Range<Int>], streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get {
      precondition(ranges.count < CCV_NNC_MAX_DIM_ALLOC)
      let dimensions = self.dimensions
      precondition(ranges.count == dimensions.count)
      let offset = ranges.map { $0.lowerBound }
      let newDimensions = ranges.map { $0.count }
      let increments = self.increments
      precondition(ranges.count == increments.count)
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < increments[i])
        assert(range.upperBound > 0 && range.upperBound <= increments[i])
      }
      return reshaped(
        format: format, dimensions: newDimensions, offset: offset, increments: increments)
    }
    set(v) {
      precondition(v.graph === graph)
      precondition(ranges.count < CCV_NNC_MAX_DIM_ALLOC)
      let dimensions = self.dimensions
      precondition(ranges.count == dimensions.count)
      let offset = ranges.map { $0.lowerBound }
      let newDimensions = ranges.map { $0.count }
      let increments = self.increments
      precondition(ranges.count == increments.count)
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < increments[i])
        assert(range.upperBound > 0 && range.upperBound <= increments[i])
      }
      // Intentionally use the format of the input so we don't do unnecessary format conversion.
      let output = reshaped(
        format: v.format, dimensions: newDimensions, offset: offset, increments: increments
      )
      let params = CmdParamsFactory.factory.newParams()
      let cmd = ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, params, 0)
      let _graph = graph.cGraph
      let _streamContext = (streamContext ?? graph.streamContext)?._stream
      var _input: ccv_nnc_tensor_variable_t? = v._tensor
      var _output: ccv_nnc_tensor_variable_t? = output._tensor
      ccv_nnc_dynamic_graph_exec(
        _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
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
      let dimensions = self.dimensions
      precondition(indices.count + 1 == dimensions.count)
      let offset = indices + [range.lowerBound]
      let newDimensions = Array(repeating: 1, count: indices.count) + [range.count]
      let increments = self.increments
      assert(range.lowerBound >= 0 && range.lowerBound < increments[indices.count])
      return reshaped(
        format: format, dimensions: newDimensions, offset: offset, increments: increments)
    }
    set(v) {
      precondition(v.graph === graph)
      precondition(indices.count + 1 < CCV_NNC_MAX_DIM_ALLOC)
      let dimensions = self.dimensions
      precondition(indices.count + 1 == dimensions.count)
      let offset = indices + [range.lowerBound]
      let newDimensions = Array(repeating: 1, count: indices.count) + [range.count]
      let increments = self.increments
      assert(range.lowerBound >= 0 && range.lowerBound < increments[indices.count])
      // Intentionally use the format of the input so we don't do unnecessary format conversion.
      let output = reshaped(
        format: v.format, dimensions: newDimensions, offset: offset, increments: increments
      )
      let params = CmdParamsFactory.factory.newParams()
      let cmd = ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, nil, params, 0)
      let _graph = graph.cGraph
      let _streamContext = (streamContext ?? graph.streamContext)?._stream
      var _input: ccv_nnc_tensor_variable_t? = v._tensor
      var _output: ccv_nnc_tensor_variable_t? = output._tensor
      ccv_nnc_dynamic_graph_exec(
        _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
    }
  }

  @usableFromInline
  subscript(indices: [Int], range: UnboundedRange, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Tensor<Element>
  {
    get {
      let dimensions = self.dimensions
      return self[indices, 0..<dimensions[indices.count], streamContext: streamContext]
    }
    set(v) {
      let dimensions = self.dimensions
      self[indices, 0..<dimensions[indices.count], streamContext: streamContext] = v
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
      let dimensions = self.dimensions
      precondition(ranges.count == dimensions.count)
      let offset = ranges.map { $0.lowerBound }
      let newDimensions = ranges.map { $0.count }
      let increments = self.increments
      precondition(ranges.count == increments.count)
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < increments[i])
        assert(range.upperBound > 0 && range.upperBound <= increments[i])
      }
      return reshaped(
        format: format, dimensions: newDimensions, offset: offset, increments: increments)
    }
    set(v) {
      precondition(v.count == count)
      guard count > 0 else { return }
      let graph = untyped[0].graph
      for x in v.untyped {
        precondition(x.graph === graph)
      }
      precondition(ranges.count < CCV_NNC_MAX_DIM_ALLOC)
      let dimensions = self.dimensions
      precondition(ranges.count == dimensions.count)
      let offset = ranges.map { $0.lowerBound }
      let newDimensions = ranges.map { $0.count }
      let increments = self.increments
      precondition(ranges.count == increments.count)
      for (i, range) in ranges.enumerated() {
        assert(range.lowerBound >= 0 && range.lowerBound < increments[i])
        assert(range.upperBound > 0 && range.upperBound <= increments[i])
      }
      // Intentionally use the format of the input so we don't do unnecessary format conversion.
      let outputs = reshaped(
        format: v.format, dimensions: newDimensions, offset: offset, increments: increments
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
      let dimensions = self.dimensions
      precondition(indices.count + 1 == dimensions.count)
      let offset = indices + [range.lowerBound]
      let newDimensions = Array(repeating: 1, count: indices.count) + [range.count]
      let increments = self.increments
      assert(range.lowerBound >= 0 && range.lowerBound < increments[indices.count])
      return reshaped(
        format: format, dimensions: newDimensions, offset: offset, increments: increments)
    }
    set(v) {
      precondition(v.count == count)
      guard count > 0 else { return }
      let graph = untyped[0].graph
      for x in v.untyped {
        precondition(x.graph === graph)
      }
      precondition(indices.count + 1 < CCV_NNC_MAX_DIM_ALLOC)
      let dimensions = self.dimensions
      precondition(indices.count + 1 == dimensions.count)
      let offset = indices + [range.lowerBound]
      let newDimensions = Array(repeating: 1, count: indices.count) + [range.count]
      let increments = self.increments
      assert(range.lowerBound >= 0 && range.lowerBound < increments[indices.count])
      // Intentionally use the format of the input so we don't do unnecessary format conversion.
      let outputs = reshaped(
        format: v.format, dimensions: newDimensions, offset: offset, increments: increments
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
    }
  }

  @usableFromInline
  subscript(indices: [Int], range: UnboundedRange, streamContext streamContext: StreamContext?)
    -> DynamicGraph.Group<Element>
  {
    get {
      let dimensions = self.dimensions
      return self[indices, 0..<dimensions[indices.count], streamContext: streamContext]
    }
    set(v) {
      let dimensions = self.dimensions
      self[indices, 0..<dimensions[indices.count], streamContext: streamContext] = v
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
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    var _input: ccv_nnc_tensor_variable_t? = self._tensor
    let rawInput = self.rawValue
    let output: DynamicGraph.Tensor<Element> = graph.variable(
      .GPU(ordinal), format: rawInput.format, dimensions: rawInput.dimensions)
    var _output: ccv_nnc_tensor_variable_t? = output._tensor
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
    return output
  }

  /// Copy the given tensor to CPU.
  public func toCPU(streamContext: StreamContext?) -> DynamicGraph.Tensor<Element> {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, nil, params, 0)
    var _input: ccv_nnc_tensor_variable_t? = self._tensor
    let rawInput = self.rawValue
    let output: DynamicGraph.Tensor<Element> = graph.variable(
      .CPU, format: rawInput.format, dimensions: rawInput.dimensions)
    var _output: ccv_nnc_tensor_variable_t? = output._tensor
    let _graph = graph.cGraph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_exec(
      _graph, cmd, ccv_nnc_no_hint, 0, &_input, 1, &_output, 1, 0, _streamContext)
    return output
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
    case .max:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_MAX_FORWARD, nil, params, 0)
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
    case .max:
      cmd = ccv_nnc_cmd(CCV_NNC_REDUCE_MAX_FORWARD, nil, params, 0)
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

extension DynamicGraph.AnyTensor {
  /// Explicitly cast the underlying storage to a specific type. It is a helper function than
  /// doing DynamicGraph.Tensor<SomeType>(something). Less code change required if we change this
  /// from tensor to group.
  public func `as`<Element: TensorNumeric>(of: Element.Type) -> DynamicGraph.Tensor<Element> {
    return DynamicGraph.Tensor(self)
  }
}

extension DynamicGraph.Group where Element == DynamicGraph.AnyTensor {
  /// Explicitly cast the underlying storage to a specific type. It is a helper function than
  /// doing DynamicGraph.Group<DynamicGraph.Tensor<SomeType>>(something). Less code change required
  /// if we change this from group to tensor.
  public func `as`<T: TensorNumeric>(of: T.Type) -> DynamicGraph.Group<DynamicGraph.Tensor<T>> {
    return DynamicGraph.Group(self)
  }
}
