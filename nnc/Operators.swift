infix operator .*: MultiplicationPrecedence
infix operator ./: MultiplicationPrecedence
infix operator .+: AdditionPrecedence

// Element-wise addition
public func .+ <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.sum(left, right)
}

public func .+ (left: Model.IO, right: Model.IO) -> Model.IO {
  return Sum()(left, right)
}

// Element-wise division
public func ./ <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.div(left: left, right: right)
}

// Broadcast element-wise multiplication
public func .* <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.mul(left: left, right: right)
}

public func .* (left: Model.IO, right: Model.IO) -> Model.IO {
  return Mul()(left, right)
}

// Broadcast element-wise addition
public func + <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.add(left: left, right: right)
}

public func + (left: Model.IO, right: Model.IO) -> Model.IO {
  return Add()(left, right)
}

// Broadcast element-wise subtraction.
public func - <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.add(left: left, right: right, rightScalar: -1)
}

public func - (left: Model.IO, right: Model.IO) -> Model.IO {
  return Add(rightScalar: -1)(left, right)
}

// Matrix multiplication
public func * <T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.matmul(left: left, right: right)
}

public func * (left: Model.IO, right: Model.IO) -> Model.IO {
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

public func * (left: Float, right: Model.IO) -> Model.IO {
  return Scalmul(left)(right)
}

public func * (left: Model.IO, right: Float) -> Model.IO {
  return Scalmul(right)(left)
}

public prefix func - <T: DynamicGraph.TensorGroup>(tensor: T) -> T {
  return Functional.scalmul(left: -1, right: tensor)
}

public prefix func - (tensor: Model.IO) -> Model.IO {
  return Scalmul(-1)(tensor)
}
