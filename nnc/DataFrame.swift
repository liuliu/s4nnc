import C_nnc

public final class DataFrame {

  final class Wrapped<T> {
    let value: T
    init(_ value: T) {
      self.value = value
    }
  }

  public final class Iterator {
    public func prefetch(_ i: Int) {
    }
  }

  private let underlying: Wrapped<[Any]>
  private let _dataframe: OpaquePointer

  public init<S: Sequence>(from sequence: S) {
    underlying = Wrapped(Array(sequence) as [Any])
    var column_data = ccv_cnnp_column_data_t()
    column_data.data_enum = { _, row_idxs, row_size, data, context, _ in
      
    }
    column_data.context = Unmanaged.passRetained(underlying).toOpaque()
    _dataframe = ccv_cnnp_dataframe_new(&column_data, 1, Int32(underlying.value.count))!
  }

  public subscript(firstIndex: String, indices: String...) -> Iterator {
    get {
      return Iterator()
    }
  }

  public subscript(index: String) -> Iterator {
    get {
      return Iterator()
    }
    set (v) {
    }
  }

  public var count: Int {
    return 0
  }
}
