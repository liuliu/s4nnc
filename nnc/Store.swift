import C_nnc
import SQLite3

extension DynamicGraph {

  final class _Store {
    let sqlite: UnsafeMutableRawPointer
    let flags: Store.OpenFlag
    init(sqlite: OpaquePointer, flags: Store.OpenFlag) {
      self.sqlite = UnsafeMutableRawPointer(sqlite)
      self.flags = flags
    }
    deinit {
      // If the database is opened with WAL mode, this makes sure everything write back to the main
      // database, much easier to operate without worrying the data left in the wal log.
      if flags.contains(.truncateWhenClose) {
        sqlite3_wal_checkpoint_v2(OpaquePointer(sqlite), nil, SQLITE_CHECKPOINT_TRUNCATE, nil, nil)
      }
      sqlite3_close(OpaquePointer(sqlite))
    }
  }

  /**
   * A key-value based parameter store.
   */
  public struct Store {
    public struct OpenFlag: OptionSet {
      public let rawValue: Int
      public init(rawValue: Int) {
        self.rawValue = rawValue
      }
      public static let truncateWhenClose = OpenFlag(rawValue: 1 << 0)
    }
    private let graph: DynamicGraph
    private let store: _Store

    /**
     * Read a type-erased tensor from the store.
     *
     * - Parameter key: The key corresponding to that particular tensor.
     */
    public func read(_ key: String) -> NNC.AnyTensor? {
      var underlying: UnsafeMutablePointer<ccv_nnc_tensor_t>? = nil
      let result = ccv_nnc_tensor_read(store.sqlite, key, nil, &underlying)
      guard result == CCV_IO_FINAL else { return nil }
      let anyTensor = AnyTensorStorage(underlying!)
      return anyTensor.toAnyTensor()
    }

    /**
     * Read a tensor from the store into tensor variable from dynamic graph.
     *
     * - Parameters:
     *   - key: The key corresponding to that particular tensor.
     *   - variable: The tensor variable to be initialized with.
     * - Returns whether we successfully initialized the tensor variable.
     */
    @discardableResult
    public func read(_ key: String, variable: DynamicGraph_Any) -> Bool {
      switch variable {
      case let tensor as DynamicGraph.AnyTensor:
        assert(tensor.graph === graph)
        let _graph = graph.cGraph
        let _tensor = tensor._tensor
        let raw = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)
        if raw != nil {
          var underlying = raw
          let result = ccv_nnc_tensor_read(store.sqlite, key, nil, &underlying)
          if result == CCV_IO_FINAL {
            assert(underlying == raw)
          }
          return result == CCV_IO_FINAL
        }
        var underlying: UnsafeMutablePointer<ccv_nnc_tensor_t>? = nil
        let result = ccv_nnc_tensor_read(store.sqlite, key, nil, &underlying)
        guard result == CCV_IO_FINAL else { return false }
        let anyTensor = AnyTensorStorage(underlying!)
        ccv_nnc_tensor_variable_set(_graph, _tensor, underlying)
        // Retain the tensor until we freed the variable.
        ccv_nnc_tensor_variable_destructor_hook(
          _graph, _tensor,
          { _, _, ctx in
            // No longer need to retain the tensor.
            Unmanaged<NNC.AnyTensorStorage>.fromOpaque(ctx!).release()
          }, Unmanaged.passRetained(anyTensor).toOpaque())
      case let group as DynamicGraph.AnyGroup:
        for (i, tensor) in group.untyped.enumerated() {
          guard read("\(key)(\(i))", variable: tensor) else {
            return false
          }
        }
      default:
        fatalError("Cannot recognize the variable")
      }
      return true
    }
    public enum ModelReaderResult {
      /// Continue to load parameter with the given name.
      case `continue`(String)
      /// The parameter is loaded, no futher operation need.
      case final(NNC.AnyTensor)
      /// Nothing is loaded.
      case fail
    }
    class ModelReaderHelper {
      let reader: (String, DataType, TensorFormat, TensorShape) -> ModelReaderResult
      let sqlite: UnsafeMutableRawPointer
      init(
        reader: @escaping (String, DataType, TensorFormat, TensorShape) -> ModelReaderResult,
        sqlite: UnsafeMutableRawPointer
      ) {
        self.reader = reader
        self.sqlite = sqlite
      }
    }
    /**
     * Read parameters into a given model.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model.
     *   - model: The model to be initialized with parameters from a given key.
     *   - reader: You can customize your reader to load parameter with a different name etc.
     */
    public func read(
      _ key: String, model: Model,
      reader: ((String, DataType, TensorFormat, TensorShape) -> ModelReaderResult)? = nil
    ) {
      guard let reader = reader else {
        ccv_cnnp_model_read(store.sqlite, key, model.cModel)
        return
      }
      let readerHelper = ModelReaderHelper(reader: reader, sqlite: store.sqlite)
      ccv_cnnp_model_set_io(
        model.cModel,
        { (handle, name, dir, tensorOut) -> Int32 in
          let readerHelper = Unmanaged<ModelReaderHelper>.fromOpaque(handle!).takeUnretainedValue()
          let cTensorOut = tensorOut!.pointee
          let params = cTensorOut!.pointee.info
          let result = readerHelper.reader(
            name.map { String(cString: $0) } ?? "", DataType.from(cTensorParams: params),
            TensorFormat.from(cTensorParams: params), TensorShape(dims: params.dim))
          switch result {
          case .final(let tensor):
            let cTensor = tensor.cTensor
            let dataSize = ccv_nnc_tensor_data_size(cTensor.pointee.info)
            ccv_nnc_tensor_swap(cTensorOut, name, dir, cTensor.pointee.data.ptr, dataSize)
            return Int32(CCV_IO_FINAL)
          case .continue(let name):
            return ccv_nnc_tensor_read(readerHelper.sqlite, name, dir, tensorOut)
          case .fail:
            return Int32(CCV_IO_ERROR)
          }
        }, nil)
      let unmanaged = Unmanaged.passRetained(readerHelper)
      ccv_cnnp_model_read(unmanaged.toOpaque(), key, model.cModel)
      ccv_cnnp_model_set_io(model.cModel, nil, nil)
      unmanaged.release()
    }
    /**
     * Read parameters into a given model builder.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model.
     *   - model: The model builder to be initialized with parameters from a given key.
     */
    public func read(_ key: String, model: AnyModelBuilder) {
      model.read(key, from: store)
    }

    /**
     * Write a tensor to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular tensor.
     *   - tensor: The tensor to be persisted.
     */
    public func write(_ key: String, tensor: NNC.AnyTensor) {
      ccv_nnc_tensor_write(tensor.cTensor, store.sqlite, key)
    }
    /**
     * Write a tensor variable to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular tensor.
     *   - variable: The tensor variable to be persisted.
     */
    public func write(_ key: String, variable: DynamicGraph_Any) {
      switch variable {
      case let tensor as DynamicGraph.AnyTensor:
        assert(tensor.graph === graph)
        let _graph = graph.cGraph
        let _tensor = tensor._tensor
        let raw = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)!
        ccv_nnc_tensor_write(raw, store.sqlite, key)
      case let group as DynamicGraph.AnyGroup:
        for (i, tensor) in group.untyped.enumerated() {
          write("\(key)(\(i))", variable: tensor)
        }
      default:
        fatalError("Cannot recognize the variable")
      }
    }
    /**
     * Write a model to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model.
     *   - model: The model where its parameters to be persisted.
     */
    public func write(_ key: String, model: Model) {
      ccv_cnnp_model_write(model.cModel, store.sqlite, key)
    }
    /**
     * Write a model builder to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model builder.
     *   - model builder: The model where its parameters to be persisted.
     */
    public func write(_ key: String, model: AnyModelBuilder) {
      write(key, model: model.model!)
    }

    init(_ store: _Store, graph: DynamicGraph) {
      self.store = store
      self.graph = graph
    }

    /**
     * Retrieve a list of all tensors in this file. This reads from the disk
     * and could take some time to finish.
     */
    public var keys: [String] {
      var stmt: OpaquePointer? = nil
      sqlite3_prepare_v2(OpaquePointer(store.sqlite), "SELECT name FROM tensors", -1, &stmt, nil)
      guard let stmt = stmt else { return [] }
      var keys = [String]()
      while SQLITE_ROW == sqlite3_step(stmt) {
        guard let cString = sqlite3_column_text(stmt, 0) else { continue }
        keys.append(String(cString: cString))
      }
      sqlite3_finalize(stmt)
      return keys
    }

    /**
     * Remove one tensor by its key.
     */
    public func remove(_ key: String) {
      var deletion: OpaquePointer? = nil
      sqlite3_prepare_v2(
        OpaquePointer(store.sqlite), "DELETE FROM tensors WHERE name=?1", -1, &deletion, nil)
      let SQLITE_TRANSIENT = unsafeBitCast(
        OpaquePointer(bitPattern: -1), to: sqlite3_destructor_type.self)
      sqlite3_bind_text(deletion, 1, key, -1, SQLITE_TRANSIENT)
      sqlite3_step(deletion)
      sqlite3_finalize(deletion)
    }

    /**
     * Remove all tensors from the store. This also vacuums the store to minimize its size.
     */
    public func removeAll() {
      sqlite3_exec(OpaquePointer(store.sqlite), "DELETE FROM tensors", nil, nil, nil)
      sqlite3_exec(OpaquePointer(store.sqlite), "VACUUM", nil, nil, nil)
    }

  }

  /**
   * Open the store from a file.
   *
   * - Parameters:
   *   - filePath: The file path for the store.
   *   - flags: The flags for the opening store. Default to truncateWhenClose.
   *   - procedure: When the store is open, you can access it from this closure.
   * - Returns: Wether this store can be successfully open or not.
   */
  @discardableResult
  public func openStore(
    _ filePath: String, flags: Store.OpenFlag = .truncateWhenClose,
    procedure: (_ store: Store) throws -> Void
  ) rethrows -> Bool {
    var _sqlite: OpaquePointer? = nil
    sqlite3_open_v2(filePath, &_sqlite, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil)
    guard let sqlite = _sqlite else { return false }
    sqlite3_busy_timeout(sqlite, 30_000)  // This is essential to have real-world usages.
    let store = Store(_Store(sqlite: sqlite, flags: flags), graph: self)
    try procedure(store)
    return true
  }

}
