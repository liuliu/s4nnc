import C_nnc

public struct SGDOptimizer: Optimizer, OptimizerAddons {
  public let graph: DynamicGraph
  public var nesterov: Bool
  public var rate: Float
  public var scale: Float
  public var decay: Float
  public var momentum: Float
  public var dampening: Float
  public var parameters = [DynamicGraph_AnyParameters]() {
    willSet {
      for var parameter in parameters.compactMap({ $0 as? DynamicGraph_Any }) {
        parameter.requiresGrad = false
      }
    }
    didSet {
      for var parameter in parameters.compactMap({ $0 as? DynamicGraph_Any }) {
        precondition(!parameter.isConstant)
        parameter.requiresGrad = true
      }
      savedAux = savedAux(minimizer: minimizer)
    }
  }

  var savedAux = [DynamicGraph_Any]()
  var minimizer: ccv_nnc_cmd_t {
    var params = CmdParamsFactory.factory.newParams()
    params.sgd.nesterov = nesterov ? 1 : 0
    params.sgd.rate = rate
    params.sgd.scale = scale
    params.sgd.decay = decay
    params.sgd.momentum = momentum
    params.sgd.dampening = dampening
    return ccv_nnc_cmd(CCV_NNC_SGD_FORWARD, nil, params, 0)
  }

  public init(
    _ graph: DynamicGraph, nesterov: Bool, rate: Float, scale: Float, decay: Float, momentum: Float,
    dampening: Float
  ) {
    self.graph = graph
    self.nesterov = nesterov
    self.rate = rate
    self.scale = scale
    self.decay = decay
    self.momentum = momentum
    self.dampening = dampening
  }

  public func step(streamContext: StreamContext?) {
    optimizerStep(
      graph: graph, minimizer: minimizer, parameters: parameters, savedAux: savedAux,
      streamContext: streamContext)
  }
}

public struct AdamOptimizer: Optimizer, OptimizerAddons {
  public let graph: DynamicGraph
  public var step: Int
  public var rate: Float
  public var beta1: Float
  public var beta2: Float
  public var decay: Float
  public var epsilon: Float
  public var parameters = [DynamicGraph_AnyParameters]() {
    willSet {
      for var parameter in parameters.compactMap({ $0 as? DynamicGraph_Any }) {
        parameter.requiresGrad = false
      }
    }
    didSet {
      for var parameter in parameters.compactMap({ $0 as? DynamicGraph_Any }) {
        precondition(!parameter.isConstant)
        parameter.requiresGrad = true
      }
      savedAux = savedAux(minimizer: minimizer)
    }
  }

  var savedAux = [DynamicGraph_Any]()
  var minimizer: ccv_nnc_cmd_t {
    var params = CmdParamsFactory.factory.newParams()
    params.adam.step = Int32(step)
    params.adam.rate = rate
    params.adam.beta1 = beta1
    params.adam.beta2 = beta2
    params.adam.decay = decay
    params.adam.epsilon = epsilon
    return ccv_nnc_cmd(CCV_NNC_ADAM_FORWARD, nil, params, 0)
  }

  public init(
    _ graph: DynamicGraph, step: Int, rate: Float, beta1: Float, beta2: Float, decay: Float,
    epsilon: Float
  ) {
    self.graph = graph
    self.step = step
    self.rate = rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.decay = decay
    self.epsilon = epsilon
  }

  public func step(streamContext: StreamContext?) {
    optimizerStep(
      graph: graph, minimizer: minimizer, parameters: parameters, savedAux: savedAux,
      streamContext: streamContext)
  }
}
