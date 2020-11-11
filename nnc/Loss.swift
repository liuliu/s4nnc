import C_nnc

public protocol Loss {
  func callAsFunction<T: DynamicGraph.AnyTensorGroup>(_ input: T, target: T, streamContext: StreamContext?) -> [T]
}

public extension Loss {
  func callAsFunction<T: DynamicGraph.AnyTensorGroup>(_ input: T, target: T) -> [T] {
    return callAsFunction(input, target: target, streamContext: nil)
  }
}

public struct SoftmaxCrossEntropyLoss: Loss {
  public var trim0: Float
  public var trim1: Float
  public init(trim0: Float = 0, trim1: Float = 1) {
    self.trim0 = trim0
    self.trim1 = trim1
  }
  public func callAsFunction<T: DynamicGraph.AnyTensorGroup>(_ input: T, target: T, streamContext: StreamContext?) -> [T] {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.label_smoothing.trim0 = trim0
    params.label_smoothing.trim1 = trim1
    let cmd = ccv_nnc_cmd(CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD, nil, params, 0)
    return Functional.exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [input, target], outputSize: 2, streamContext: streamContext) as! [DynamicGraph.AnyTensor]
  }
}

public struct SigmoidBinaryCrossEntropyLoss: Loss {
  public var posWeight: Float
  public init(posWeight: Float = 1) {
    self.posWeight = posWeight
  }
  public func callAsFunction(_ input: DynamicGraph.AnyTensor, target: DynamicGraph.AnyTensor, streamContext: StreamContext?) -> [DynamicGraph.AnyTensor] {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.binary_crossentropy.pos_weight = posWeight
    let cmd = ccv_nnc_cmd(CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD, nil, params, 0)
    return Functional.exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [input, target], outputSize: 2, streamContext: streamContext) as! [DynamicGraph.AnyTensor]
  }
}

public struct BinaryCrossEntropyLoss: Loss {
  public var posWeight: Float
  public init(posWeight: Float = 1) {
    self.posWeight = posWeight
  }
  public func callAsFunction(_ input: DynamicGraph.AnyTensor, target: DynamicGraph.AnyTensor, streamContext: StreamContext?) -> [DynamicGraph.AnyTensor] {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.binary_crossentropy.pos_weight = posWeight
    let cmd = ccv_nnc_cmd(CCV_NNC_BINARY_CROSSENTROPY_FORWARD, nil, params, 0)
    return Functional.exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [input, target], outputSize: 1, streamContext: streamContext) as! [DynamicGraph.AnyTensor]
  }
}

public struct CategoricalCrossEntropyLoss: Loss {
  public var trim0: Float
  public var trim1: Float
  public init(trim0: Float = 0, trim1: Float = 1) {
    self.trim0 = trim0
    self.trim1 = trim1
  }
  public func callAsFunction(_ input: DynamicGraph.AnyTensor, target: DynamicGraph.AnyTensor, streamContext: StreamContext?) -> [DynamicGraph.AnyTensor] {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    params.label_smoothing.trim0 = trim0
    params.label_smoothing.trim1 = trim1
    let cmd = ccv_nnc_cmd(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, nil, params, 0)
    return Functional.exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [input, target], outputSize: 1, streamContext: streamContext) as! [DynamicGraph.AnyTensor]
  }
}

public struct SmoothL1Loss: Loss {
  public func callAsFunction(_ input: DynamicGraph.AnyTensor, target: DynamicGraph.AnyTensor, streamContext: StreamContext?) -> [DynamicGraph.AnyTensor] {
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_SMOOTH_L1_FORWARD, nil, params, 0)
    return Functional.exec(cmd: cmd, hint: ccv_nnc_no_hint, inputs: [input, target], outputSize: 1, streamContext: streamContext) as! [DynamicGraph.AnyTensor]
  }
}
