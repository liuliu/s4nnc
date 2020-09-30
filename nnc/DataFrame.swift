import C_nnc

public final class DataFrame {

  final class Wrapped<T> {
    let value: T
    init(_ value: T) {
      self.value = value
    }
  }

  public final class Series: Sequence {

    public final class Iterator: IteratorProtocol {
      public typealias Element = AnyObject
      private weak var series: Series?
      public init(_ series: Series) {
        self.series = series
      }

      public func next() -> AnyObject? {
        return series?.next()
      }
    }

    public typealias Element = AnyObject

    public func makeIterator() -> Iterator {
      return Iterator(self)
    }

    public func prefetch(_ i: Int, streamContext: StreamContext? = nil) {
      ccv_cnnp_dataframe_iter_prefetch(_iter, Int32(i), streamContext?._stream)
    }

    public func next(_ streamContext: StreamContext? = nil) -> AnyObject? {
      var data: UnsafeMutableRawPointer? = nil
      let retval = ccv_cnnp_dataframe_iter_next(_iter, &data, 1, streamContext?._stream)
      guard retval == 0 else { return nil }
      if data == nil {
        return nil
      }
      return Unmanaged<AnyObject>.fromOpaque(data!).takeUnretainedValue()
    }

    public var underestmiatedCount: Int {
      return count
    }

    public let count: Int

    private let _iter: OpaquePointer

    fileprivate init(_ iter: OpaquePointer, count: Int) {
      _iter = iter
      self.count = count
    }

    deinit {
      ccv_cnnp_dataframe_iter_free(_iter)
    }
  }

  private let underlying: Wrapped<[AnyObject]>
  private let _dataframe: OpaquePointer

  public init<S: Sequence>(from sequence: S) {
    underlying = Wrapped(Array(sequence) as [AnyObject])
    var column_data = ccv_cnnp_column_data_t()
    column_data.data_enum = { _, row_idxs, row_size, data, context, _ in
      guard let row_idxs = row_idxs else { return }
      guard let data = data else { return }
      let underlying = Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context!).takeUnretainedValue()
      for i in 0..<Int(row_size) {
        let idx = Int((row_idxs + i).pointee)
        let value = underlying.value[idx]
        (data + i).initialize(to: Unmanaged.passRetained(value).toOpaque())
      }
    }
    column_data.data_deinit = { data, _ in
      guard let data = data else { return }
      Unmanaged<AnyObject>.fromOpaque(data).release()
    }
    column_data.context_deinit = { context in
      guard let context = context else { return }
      Unmanaged<Wrapped<[AnyObject]>>.fromOpaque(context).release()
    }
    column_data.context = Unmanaged.passRetained(underlying).toOpaque()
    _dataframe = ccv_cnnp_dataframe_new(&column_data, 1, Int32(underlying.value.count))!
  }

  public subscript(firstIndex: String, indices: String...) -> Series {
    get {
      var i: Int32 = 0
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, &i, 1)!
      let rowCount = ccv_cnnp_dataframe_row_count(_dataframe)
      return Series(iter, count: Int(rowCount))
    }
  }

  public subscript(index: String) -> Series {
    get {
      var i: Int32 = 0
      let iter = ccv_cnnp_dataframe_iter_new(_dataframe, &i, 1)!
      let rowCount = ccv_cnnp_dataframe_row_count(_dataframe)
      return Series(iter, count: Int(rowCount))
    }
    set (v) {
    }
  }

  public var count: Int {
    let rowCount = ccv_cnnp_dataframe_row_count(_dataframe)
    return Int(rowCount)
  }

  deinit {
    ccv_cnnp_dataframe_free(_dataframe)
  }
}
