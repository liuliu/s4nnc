infix operator .*: MultiplicationPrecedence

// Element-wise multiplication
public func .*<Element>(left: DynamicGraph.Tensor<Element>, right: DynamicGraph.Tensor<Element>) -> DynamicGraph.Tensor<Element> {
  return Functional.mul(left: left, right: right)
}

public func .*(left: Model.IO, right: Model.IO) -> Model.IO {
  return Mul()(left, right)
}

// Element-wise addition
public func +<Element>(left: DynamicGraph.Tensor<Element>, right: DynamicGraph.Tensor<Element>) -> DynamicGraph.Tensor<Element> {
  return Functional.add(left: left, right: right)
}

public func +(left: Model.IO, right: Model.IO) -> Model.IO {
  return Add()(left, right)
}

// Matrix multiplication
public func *<Element>(left: DynamicGraph.Tensor<Element>, right: DynamicGraph.Tensor<Element>) -> DynamicGraph.Tensor<Element> {
  return Functional.matmul(left: left, right: right)
}

public func *(left: Model.IO, right: Model.IO) -> Model.IO {
  return Matmul()(left, right)
}
