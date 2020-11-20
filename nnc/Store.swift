import C_nnc
import SQLite3

extension DynamicGraph {

  final class _Store {
    let sqlite: UnsafeMutableRawPointer
    init(sqlite: OpaquePointer) {
      self.sqlite = UnsafeMutableRawPointer(sqlite)
    }
    deinit {
      sqlite3_close(OpaquePointer(sqlite))
    }
  }

  public struct Store {
    private let store: _Store

    public func read(_ key: String, variable: DynamicGraph_Any) {
      switch variable {
      case let tensor as DynamicGraph.AnyTensor:
        let _graph = tensor.graph._graph
        let _tensor = tensor._tensor
        let raw = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)
        if raw != nil {
          var underlying = raw
          let result = ccv_nnc_tensor_read(store.sqlite, key, &underlying)
          if result == CCV_IO_FINAL {
            assert(underlying == raw)
          }
          return
        }
        var underlying: UnsafeMutablePointer<ccv_nnc_tensor_t>? = nil
        let result = ccv_nnc_tensor_read(store.sqlite, key, &underlying)
        guard result == CCV_IO_FINAL else { return }
        let anyTensor = _AnyTensor(underlying!)
        ccv_nnc_tensor_variable_set(_graph, _tensor, anyTensor._tensor)
        // Retain the tensor until we freed the variable.
        ccv_nnc_tensor_variable_destructor_hook(_graph, _tensor, { _, _, ctx in
          // No longer need to retain the tensor.
          Unmanaged<nnc._AnyTensor>.fromOpaque(ctx!).release()
        }, Unmanaged.passRetained(anyTensor).toOpaque())
        break
      case let group as DynamicGraph.AnyGroup:
        for (i, tensor) in group.underlying.enumerated() {
          read("\(key)(\(i))", variable: tensor)
        }
      default:
        fatalError("Cannot recognize the variable")
      }
    }
    public func read(_ key: String, model: Model) {
      ccv_cnnp_model_read(store.sqlite, key, model._model)
    }
    public func read(_ key: String, model: AnyModelBuilder) {
      model.read(key, from: store)
    }

    public func write(_ key: String, variable: DynamicGraph_Any) {
      switch variable {
      case let tensor as DynamicGraph.AnyTensor:
        let _graph = tensor.graph._graph
        let _tensor = tensor._tensor
        let raw = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)!
        ccv_nnc_tensor_write(raw, store.sqlite, key)
      case let group as DynamicGraph.AnyGroup:
        for (i, tensor) in group.underlying.enumerated() {
          write("\(key)(\(i))", variable: tensor)
        }
      default:
        fatalError("Cannot recognize the variable")
      }
    }
    public func write(_ key: String, model: Model) {
      ccv_cnnp_model_write(model._model, store.sqlite, key)
    }
    public func write(_ key: String, model: AnyModelBuilder) {
      write(key, model: model.model!)
    }

    init(_ store: _Store) {
      self.store = store
    }

  }

  @discardableResult
  public func openStore(_ filePath: String, procedure: (_ store: Store) -> Void) -> Bool {
    var _sqlite: OpaquePointer? = nil
    sqlite3_open_v2(filePath, &_sqlite, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil)
    guard let sqlite = _sqlite else { return false }
    let store = Store(_Store(sqlite: sqlite))
    procedure(store)
    return true
  }

}
