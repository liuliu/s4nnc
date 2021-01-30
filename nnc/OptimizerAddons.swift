import C_nnc

/// Stochastic gradient descent optimizer.
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

/// Adam optimizer.
public struct AdamOptimizer: Optimizer, OptimizerAddons {
  public let graph: DynamicGraph
  public var step: Int
  public var rate: Float
  public var betas: (Float, Float)
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
    params.adam.beta1 = betas.0
    params.adam.beta2 = betas.1
    params.adam.decay = decay
    params.adam.epsilon = epsilon
    return ccv_nnc_cmd(CCV_NNC_ADAM_FORWARD, nil, params, 0)
  }

  public init(
    _ graph: DynamicGraph, rate: Float, step: Int = 1, betas: (Float, Float) = (0.9, 0.999),
    decay: Float = 0,
    epsilon: Float = 1e-8
  ) {
    self.graph = graph
    self.step = step
    self.rate = rate
    self.betas = betas
    self.decay = decay
    self.epsilon = epsilon
  }

  public mutating func step(streamContext: StreamContext?) {
    optimizerStep(
      graph: graph, minimizer: minimizer, parameters: parameters, savedAux: savedAux,
      streamContext: streamContext)
    step += 1
  }
}
