infix operator .* : MultiplicationPrecedence
infix operator ./ : MultiplicationPrecedence
infix operator .+ : AdditionPrecedence

// Element-wise addition
public func .+ <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.sum(left, right)
}

public func .+ (left: ModelIOConvertible, right: ModelIOConvertible) -> Model.IO {
  return Sum()(left, right)
}

// Element-wise division
public func ./ <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.div(left: left, right: right)
}

public func ./ (left: ModelIOConvertible, right: ModelIOConvertible) -> Model.IO {
  return Div(reciprocal: false)(left, right)
}

// Broadcast element-wise multiplication
public func .* <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.mul(left: left, right: right)
}

public func .* (left: ModelIOConvertible, right: ModelIOConvertible) -> Model.IO {
  return Mul()(left, right)
}

// Broadcast element-wise addition
public func + <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.add(left: left, right: right)
}

public func + <Element: TensorNumeric>(left: Float, right: DynamicGraph.Tensor<Element>)
  -> DynamicGraph.Tensor<Element>
{
  let graph = right.graph
  let leftConstant = graph.constant(right.kind, format: right.format, shape: [1], of: Element.self)
  leftConstant.full(left)
  return Functional.add(left: leftConstant, right: right)
}

public func + <Element: TensorNumeric>(left: DynamicGraph.Tensor<Element>, right: Float)
  -> DynamicGraph.Tensor<Element>
{
  let graph = left.graph
  let rightConstant = graph.constant(left.kind, format: left.format, shape: [1], of: Element.self)
  rightConstant.full(right)
  return Functional.add(left: left, right: rightConstant)
}

public func + <Element: TensorNumeric>(
  left: Float, right: DynamicGraph.Group<DynamicGraph.Tensor<Element>>
) -> DynamicGraph.Group<DynamicGraph.Tensor<Element>> {
  let graph = right.graph
  let leftConstant = DynamicGraph.Group(
    right.map {
      graph.constant($0.kind, format: $0.format, shape: [1], of: Element.self)
    })
  leftConstant.full(left)
  return Functional.add(left: leftConstant, right: right)
}

public func + <Element: TensorNumeric>(
  left: DynamicGraph.Group<DynamicGraph.Tensor<Element>>, right: Float
) -> DynamicGraph.Group<DynamicGraph.Tensor<Element>> {
  let graph = left.graph
  let rightConstant = DynamicGraph.Group(
    left.map {
      graph.constant($0.kind, format: $0.format, shape: [1], of: Element.self)
    })
  rightConstant.full(right)
  return Functional.add(left: left, right: rightConstant)
}

public func + (left: ModelIOConvertible, right: ModelIOConvertible) -> Model.IO {
  return Add()(left, right)
}

public func + (left: Float, right: ModelIOConvertible) -> Model.IO {
  return Add()(Scalar(value: left)(right), right)
}

public func + (left: ModelIOConvertible, right: Float) -> Model.IO {
  return Add()(left, Scalar(value: right)(left))
}

public func / (left: Float, right: ModelIOConvertible) -> Model.IO {
  if left == 1 {
    return Div(reciprocal: true)(right)
  } else {
    return left * Div(reciprocal: true)(right)
  }
}

public func / (left: ModelIOConvertible, right: Float) -> Model.IO {
  if right == 1 {
    return left.io
  } else {
    return Scalmul(1.0 / right)(left)
  }
}

// Broadcast element-wise subtraction.
public func - <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.add(left: left, right: right, rightScalar: -1)
}

public func - <Element: TensorNumeric>(left: Float, right: DynamicGraph.Tensor<Element>)
  -> DynamicGraph.Tensor<Element>
{
  let graph = right.graph
  let leftConstant = graph.constant(right.kind, format: right.format, shape: [1], of: Element.self)
  leftConstant.full(left)
  return Functional.add(left: leftConstant, right: right, rightScalar: -1)
}

public func - <Element: TensorNumeric>(left: DynamicGraph.Tensor<Element>, right: Float)
  -> DynamicGraph.Tensor<Element>
{
  let graph = left.graph
  let rightConstant = graph.constant(left.kind, format: left.format, shape: [1], of: Element.self)
  rightConstant.full(-right)
  return Functional.add(left: left, right: rightConstant)
}

public func - <Element: TensorNumeric>(
  left: Float, right: DynamicGraph.Group<DynamicGraph.Tensor<Element>>
) -> DynamicGraph.Group<DynamicGraph.Tensor<Element>> {
  let graph = right.graph
  let leftConstant = DynamicGraph.Group(
    right.map {
      graph.constant($0.kind, format: $0.format, shape: [1], of: Element.self)
    })
  leftConstant.full(left)
  return Functional.add(left: leftConstant, right: right, rightScalar: -1)
}

public func - <Element: TensorNumeric>(
  left: DynamicGraph.Group<DynamicGraph.Tensor<Element>>, right: Float
) -> DynamicGraph.Group<DynamicGraph.Tensor<Element>> {
  let graph = left.graph
  let rightConstant = DynamicGraph.Group(
    left.map {
      graph.constant($0.kind, format: $0.format, shape: [1], of: Element.self)
    })
  rightConstant.full(-right)
  return Functional.add(left: left, right: rightConstant)
}

public func - (left: ModelIOConvertible, right: ModelIOConvertible) -> Model.IO {
  return Add(rightScalar: -1)(left, right)
}

public func - (left: Float, right: ModelIOConvertible) -> Model.IO {
  return Add(rightScalar: -1)(Scalar(value: left)(right), right)
}

public func - (left: ModelIOConvertible, right: Float) -> Model.IO {
  return Add()(left, Scalar(value: -right)(left))
}

// Matrix multiplication
public func * <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.matmul(left: left, right: right)
}

public func * (left: ModelIOConvertible, right: ModelIOConvertible) -> Model.IO {
  return Matmul()(left, right)
}

// Scalar division
public func / <T: DynamicGraph.TensorGroup>(left: Float, right: T) -> T {
  if left == 1 {
    return Functional.reciprocal(right)
  } else {
    return Functional.scalmul(left: left, right: Functional.reciprocal(right))
  }
}

public func / <T: DynamicGraph.TensorGroup>(left: T, right: Float) -> T {
  if right == 1 {
    return left
  } else {
    return Functional.scalmul(left: 1.0 / right, right: left)
  }
}

// Scalar-matrix multiplication
public func * <T: DynamicGraph.TensorGroup>(left: Float, right: T) -> T {
  return Functional.scalmul(left: left, right: right)
}

public func * <T: DynamicGraph.TensorGroup>(left: T, right: Float) -> T {
  return Functional.scalmul(left: right, right: left)
}

public func * (left: Float, right: ModelIOConvertible) -> Model.IO {
  return Scalmul(left)(right)
}

public func * (left: ModelIOConvertible, right: Float) -> Model.IO {
  return Scalmul(right)(left)
}

public prefix func - <T: DynamicGraph.TensorGroup>(tensor: T) -> T {
  return Functional.scalmul(left: -1, right: tensor)
}

public prefix func - (tensor: ModelIOConvertible) -> Model.IO {
  return Scalmul(-1)(tensor)
}
