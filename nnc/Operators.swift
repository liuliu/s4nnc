infix operator .*: MultiplicationPrecedence
infix operator .+: AdditionPrecedence

// Element-wise addition
public func .+<T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.sum(left, right)
}

public func .+(left: Model.IO, right: Model.IO) -> Model.IO {
  return Sum()(left, right)
}

// Broadcast element-wise multiplication
public func .*<T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.mul(left: left, right: right)
}

public func .*(left: Model.IO, right: Model.IO) -> Model.IO {
  return Mul()(left, right)
}

// Broadcast element-wise addition
public func +<T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.add(left: left, right: right)
}

public func +(left: Model.IO, right: Model.IO) -> Model.IO {
  return Add()(left, right)
}

// Broadcast element-wise subtraction.
public func -<T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.add(left: left, right: right, rightScalar: -1)
}

public func -(left: Model.IO, right: Model.IO) -> Model.IO {
  return Add(rightScalar: -1)(left, right)
}

// Matrix multiplication
public func *<T: DynamicGraph.TensorGroup>(left: T, right: T) -> T {
  return Functional.matmul(left: left, right: right)
}

public func *(left: Model.IO, right: Model.IO) -> Model.IO {
  return Matmul()(left, right)
}

// Scalar-matrix multiplication
public func *<T: DynamicGraph.TensorGroup>(left: Float, right: T) -> T {
  return Functional.scalmul(left: left, right: right)
}

public func *<T: DynamicGraph.TensorGroup>(left: T, right: Float) -> T {
  return Functional.scalmul(left: right, right: left)
}

public func *(left: Float, right: Model.IO) -> Model.IO {
  return Scalmul(left)(right)
}

public func *(left: Model.IO, right: Float) -> Model.IO {
  return Scalmul(right)(left)
}
