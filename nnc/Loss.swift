import C_nnc

/// A generic loss function protocol.
public protocol Loss {
  func callAsFunction<T: DynamicGraph.AnyTensorGroup, U: DynamicGraph.AnyTensorGroup>(
    _ input: T, target: U, streamContext: StreamContext?
  ) -> [T.AnyTensor] where T.AnyTensor == U.AnyTensor
}

extension Loss {
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup, U: DynamicGraph.AnyTensorGroup>(
    _ input: T, target: U
  ) -> [T.AnyTensor] where T.AnyTensor == U.AnyTensor {
    return callAsFunction(input, target: target, streamContext: nil)
  }
}

/// Softmax cross-entropy loss. This combines softmax with cross-entropy loss to maximize
/// numerical stability.
public struct SoftmaxCrossEntropyLoss: Loss {
  public var trim0: Float
  public var trim1: Float
  public init(trim0: Float = 0, trim1: Float = 1) {
    self.trim0 = trim0
    self.trim1 = trim1
  }
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup, U: DynamicGraph.AnyTensorGroup>(
    _ input: T, target: U, streamContext: StreamContext?
  ) -> [T.AnyTensor] where T.AnyTensor == U.AnyTensor {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.label_smoothing.trim0 = trim0
    params.label_smoothing.trim1 = trim1
    let cmd = ccv_nnc_cmd(CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD, nil, params, 0)
    return Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, target, outputSize: 2,
      streamContext: streamContext)
  }
}

/// Sigmoid cross-entropy loss. This combines sigmoid with binary cross-entropy loss to maximize
/// numerical stability.
public struct SigmoidBinaryCrossEntropyLoss: Loss {
  public var posWeight: Float
  public init(posWeight: Float = 1) {
    self.posWeight = posWeight
  }
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup, U: DynamicGraph.AnyTensorGroup>(
    _ input: T, target: U, streamContext: StreamContext?
  ) -> [T.AnyTensor] where T.AnyTensor == U.AnyTensor {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.binary_crossentropy.pos_weight = posWeight
    let cmd = ccv_nnc_cmd(CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD, nil, params, 0)
    return Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, target, outputSize: 2,
      streamContext: streamContext)
  }
}

/// Binary cross-entropy loss.
public struct BinaryCrossEntropyLoss: Loss {
  public var posWeight: Float
  public init(posWeight: Float = 1) {
    self.posWeight = posWeight
  }
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup, U: DynamicGraph.AnyTensorGroup>(
    _ input: T, target: U, streamContext: StreamContext?
  ) -> [T.AnyTensor] where T.AnyTensor == U.AnyTensor {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.binary_crossentropy.pos_weight = posWeight
    let cmd = ccv_nnc_cmd(CCV_NNC_BINARY_CROSSENTROPY_FORWARD, nil, params, 0)
    return Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, target, outputSize: 1,
      streamContext: streamContext)
  }
}

/// Multi-class cross-entropy loss.
public struct CategoricalCrossEntropyLoss: Loss {
  public var trim0: Float
  public var trim1: Float
  public init(trim0: Float = 0, trim1: Float = 1) {
    self.trim0 = trim0
    self.trim1 = trim1
  }
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup, U: DynamicGraph.AnyTensorGroup>(
    _ input: T, target: U, streamContext: StreamContext?
  ) -> [T.AnyTensor] where T.AnyTensor == U.AnyTensor {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.label_smoothing.trim0 = trim0
    params.label_smoothing.trim1 = trim1
    let cmd = ccv_nnc_cmd(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, nil, params, 0)
    return Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, target, outputSize: 1,
      streamContext: streamContext)
  }
}

/// Smooth L1 loss (for object detection).
public struct SmoothL1Loss: Loss {
  public init() {}
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup, U: DynamicGraph.AnyTensorGroup>(
    _ input: T, target: U, streamContext: StreamContext?
  ) -> [T.AnyTensor] where T.AnyTensor == U.AnyTensor {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SMOOTH_L1_FORWARD, nil, params, 0)
    return Functional.exec(
      cmd: cmd, hint: ccv_nnc_no_hint, inputs: input, target, outputSize: 1,
      streamContext: streamContext)
  }
}
