import C_nnc

extension Model.Parameters {
  /**
   * Copy parameters from one model to another.
   *
   * - Parameter parameters: The parameters of another model, it must match the parameters copy to.
   */
  public func copy(from parameters: Model.Parameters) {
    guard var fromModel = parameters.model,
      var toModel = model
    else {
      fatalError()
    }
    // We can only copy parameters from fully compiled model, i.e., the owner of the sub-models.
    // Try to find them.
    while let owner = fromModel.owner {
      fromModel = owner
    }
    while let owner = toModel.owner {
      toModel = owner
    }
    ccv_cnnp_model_set_parameters(toModel.cModel, _io, fromModel.cModel, parameters._io)
  }

  /**
   * Copy parameters from a tensor.
   *
   * - Parameter tensor: The tensor contains some values, it must match the parameters copy to.
   */
  public func copy(from tensor: AnyTensor) {
    guard var toModel = model else {
      fatalError()
    }
    while let owner = toModel.owner {
      toModel = owner
    }
    ccv_cnnp_model_set_parameter(toModel.cModel, _io, tensor.cTensor)
  }

  /**
   * Copy parameters from a tensor.
   *
   * - Parameter tensor: The tensor contains some values, it must match the parameters copy to.
   */
  public func copy<Element: TensorNumeric>(from tensor: DynamicGraph.Tensor<Element>) {
    copy(from: tensor.rawValue)
  }

  /**
   * Copy parameters to a tensor.
   *
   * - Parameter tensor: The tensor to copy to, it must match the parameters copy from.
   */
  public func copy(to tensor: AnyTensor) {
    guard var toModel = model else {
      fatalError()
    }
    while let owner = toModel.owner {
      toModel = owner
    }
    ccv_cnnp_model_parameter_copy(toModel.cModel, _io, tensor.cTensor)
  }

  /**
   * Copy parameter out into a tensor.
   *
   * - Parameter type: The element type of a tensor.
   */
  public func copied<Element: TensorNumeric>(_ type: Element.Type = Element.self) -> Tensor<Element>
  {
    guard var toModel = model else {
      fatalError()
    }
    while let owner = toModel.owner {
      toModel = owner
    }
    let params = ccv_cnnp_model_parameter_tensor_params(toModel.cModel, _io)
    let output = ccv_nnc_tensor_new(nil, params, 0)
    ccv_cnnp_model_parameter_copy(toModel.cModel, _io, output)
    return AnyTensorStorage(output!).toTensor(Tensor<Element>.self)
  }

  /**
   * Copy parameters to a tensor.
   *
   * - Parameter tensor: The tensor to copy to, it must match the parameters copy from.
   */
  public func copy<Element: TensorNumeric>(to tensor: DynamicGraph.Tensor<Element>) {
    copy(to: tensor.rawValue)
  }
}

extension Model.Parameters {
  public enum ModelParametersShareResult {
    /// Continue to load parameter with the given name.
    case `continue`(String)
    /// Nothing is loaded.
    case fail
  }
  private final class ModelParametersShareRenameHelper {
    let renamer: (String, String) -> ModelParametersShareResult
    init(renamer: @escaping (String, String) -> ModelParametersShareResult) {
      self.renamer = renamer
    }
  }
  /**
   * Share parameters from another model. This is a specific memory optimization.
   *
   * - Parameter parameters: The model parameter to share from.
   */
  public func share(
    from parameters: Model.Parameters,
    renamer: ((String, String) -> ModelParametersShareResult)? = nil
  ) {
    guard var fromModel = parameters.model,
      var toModel = model
    else {
      fatalError()
    }
    // We can only copy parameters from fully compiled model, i.e., the owner of the sub-models.
    // Try to find them.
    while let owner = fromModel.owner {
      fromModel = owner
    }
    while let owner = toModel.owner {
      toModel = owner
    }
    toModel.originals.append(fromModel)
    guard let renamer = renamer else {
      ccv_cnnp_model_share_parameters(
        toModel.cModel, _io, fromModel.cModel, parameters._io, nil, nil)
      return
    }
    let renameHelper = ModelParametersShareRenameHelper(renamer: renamer)
    let unmanaged = Unmanaged.passRetained(renameHelper)
    ccv_cnnp_model_share_parameters(
      toModel.cModel, _io, fromModel.cModel, parameters._io,
      { handle, srcName, updatedName, providedSize in
        let renameHelper = Unmanaged<ModelParametersShareRenameHelper>.fromOpaque(handle!)
          .takeUnretainedValue()
        let result = renameHelper.renamer(
          srcName.map { String(cString: $0) } ?? "", updatedName.map { String(cString: $0) } ?? "")
        switch result {
        case .fail:
          return -1
        case .continue(var name):
          name.withUTF8 {
            if $0.count > 0 {
              memcpy(updatedName!, $0.baseAddress!, min($0.count, providedSize - 1))
            }
            updatedName?[min($0.count, providedSize - 1)] = 0
          }
          return 0
        }
      }, unmanaged.toOpaque())
    unmanaged.release()
  }
}

extension Model.Parameters {
  /**
   * Interpolate from current parameters to the another.
   *
   * parameters = (1 - weight) * parameters + weight * other
   *
   * - Parameter weight: How much the other parameter should weight, it must be between [0, 1].
   * - Parameter parameters: The parameters of another model, it must match the parameters to update.
   * - Parameter streamContext: The stream context to apply the lerp operation.
   */
  public func lerp(
    _ weight: Float, to parameters: Model.Parameters, streamContext: StreamContext? = nil
  ) {
    precondition(weight >= 0 && weight <= 1)
    guard var fromModel = parameters.model,
      var toModel = model
    else {
      fatalError()
    }
    // We can only copy parameters from fully compiled model, i.e., the owner of the sub-models.
    // Try to find them.
    while let owner = fromModel.owner {
      fromModel = owner
    }
    while let owner = toModel.owner {
      toModel = owner
    }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (1 - weight, weight, 0)
    let cmd = ccv_nnc_cmd(CCV_NNC_ADD_FORWARD, nil, params, 0)
    let graph = toModel.graph
    let _streamContext = (streamContext ?? graph?.streamContext)?._stream
    ccv_cnnp_model_parameters_zip_map(
      toModel.cModel, _io, cmd, ccv_nnc_no_hint, 0, nil, 0, nil, 0, _streamContext,
      fromModel.cModel, parameters._io
    )
  }
}

extension Model.Parameters {
  func clamp(min: Float?, max: Float?, streamContext: StreamContext?) {
    precondition(min != nil || max != nil)
    guard var toModel = model else {
      fatalError()
    }
    // We can only copy parameters from fully compiled model, i.e., the owner of the sub-models.
    // Try to find them.
    while let owner = toModel.owner {
      toModel = owner
    }
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.clamp.min = min ?? Float.nan
    params.clamp.max = max ?? Float.nan
    let cmd = ccv_nnc_cmd(CCV_NNC_CLAMP_FORWARD, nil, params, 0)
    let graph = toModel.graph
    let _streamContext = (streamContext ?? graph?.streamContext)?._stream
    ccv_cnnp_model_parameters_map(
      toModel.cModel, _io, cmd, ccv_nnc_no_hint, 0, nil, 0, nil, 0, _streamContext)
  }

  /**
   * Clamp current parameters between two values.
   */
  public func clamp(_ range: ClosedRange<Float>, streamContext: StreamContext? = nil) {
    clamp(min: range.lowerBound, max: range.upperBound, streamContext: streamContext)
  }

  /**
   * Clamp current parameters with a lower bound.
   */
  public func clamp(_ range: PartialRangeFrom<Float>, streamContext: StreamContext? = nil) {
    clamp(min: range.lowerBound, max: nil, streamContext: streamContext)
  }

  /**
   * Clamp current parameters with an upper bound.
   */
  public func clamp(_ range: PartialRangeThrough<Float>, streamContext: StreamContext? = nil) {
    clamp(min: nil, max: range.upperBound, streamContext: streamContext)
  }
}

extension Model.Parameters {
  public enum NormType: Int32 {
    case norm2 = 2
  }
  public func clipGradNorm(
    maxNorm: Float, normType: NormType = .norm2, streamContext: StreamContext? = nil
  ) {
    precondition(maxNorm >= 0)
    guard var toModel = model else {
      fatalError()
    }
    // We can only copy parameters from fully compiled model, i.e., the owner of the sub-models.
    // Try to find them.
    while let owner = toModel.owner {
      toModel = owner
    }
    let graph = toModel.graph
    let _streamContext = (streamContext ?? graph?.streamContext)?._stream
    ccv_cnnp_model_parameters_clip_grad_norm(
      toModel.cModel, _io, normType.rawValue, maxNorm, _streamContext)
  }
}
