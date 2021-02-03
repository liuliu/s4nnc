@_functionBuilder
public struct Sequential {

  public typealias Expression = Model

  public typealias Component = [Model]

  public typealias FinalResult = Model

  public static func buildExpression(_ expression: Expression) -> Component {
    return [expression]
  }

  public static func buildBlock(_ children: Component...) -> Component {
    return children.flatMap { $0 }
  }

  public static func buildArray(_ components: [Component]) -> Component {
    return components.flatMap { $0 }
  }

  public static func buildBlock(_ component: Component) -> Component {
    return component
  }

  public static func buildOptional(_ children: Component?) -> Component {
    return children ?? []
  }

  public static func buildEither(first child: Component) -> Component {
    return child
  }

  public static func buildEither(second child: Component) -> Component {
    return child
  }

  public static func buildFinalResult(_ component: Component) -> FinalResult {
    return Model(component)
  }
}
