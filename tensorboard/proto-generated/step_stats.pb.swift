// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorboard/compat/proto/step_stats.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that you are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

/// An allocation/de-allocation operation performed by the allocator.
struct Tensorboard_AllocationRecord: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// The timestamp of the operation.
  var allocMicros: Int64 = 0

  /// Number of bytes allocated, or de-allocated if negative.
  var allocBytes: Int64 = 0

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

struct Tensorboard_AllocatorMemoryUsed: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var allocatorName: String = String()

  /// These are per-node allocator memory stats.
  var totalBytes: Int64 = 0

  var peakBytes: Int64 = 0

  /// The bytes that are not deallocated.
  var liveBytes: Int64 = 0

  /// The allocation and deallocation timeline.
  var allocationRecords: [Tensorboard_AllocationRecord] = []

  /// These are snapshots of the overall allocator memory stats.
  /// The number of live bytes currently allocated by the allocator.
  var allocatorBytesInUse: Int64 = 0

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

/// Output sizes recorded for a single execution of a graph node.
struct Tensorboard_NodeOutput: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var slot: Int32 = 0

  var tensorDescription: Tensorboard_TensorDescription {
    get {return _tensorDescription ?? Tensorboard_TensorDescription()}
    set {_tensorDescription = newValue}
  }
  /// Returns true if `tensorDescription` has been explicitly set.
  var hasTensorDescription: Bool {return self._tensorDescription != nil}
  /// Clears the value of `tensorDescription`. Subsequent reads from it will return its default value.
  mutating func clearTensorDescription() {self._tensorDescription = nil}

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}

  fileprivate var _tensorDescription: Tensorboard_TensorDescription? = nil
}

/// For memory tracking.
struct Tensorboard_MemoryStats: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var tempMemorySize: Int64 = 0

  var persistentMemorySize: Int64 = 0

  var persistentTensorAllocIds: [Int64] = []

  /// NOTE: This field was marked as deprecated in the .proto file.
  var deviceTempMemorySize: Int64 = 0

  /// NOTE: This field was marked as deprecated in the .proto file.
  var devicePersistentMemorySize: Int64 = 0

  /// NOTE: This field was marked as deprecated in the .proto file.
  var devicePersistentTensorAllocIds: [Int64] = []

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

/// Time/size stats recorded for a single execution of a graph node.
struct Tensorboard_NodeExecStats: @unchecked Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  /// TODO(tucker): Use some more compact form of node identity than
  /// the full string name.  Either all processes should agree on a
  /// global id (cost_id?) for each node, or we should use a hash of
  /// the name.
  var nodeName: String {
    get {return _storage._nodeName}
    set {_uniqueStorage()._nodeName = newValue}
  }

  var allStartMicros: Int64 {
    get {return _storage._allStartMicros}
    set {_uniqueStorage()._allStartMicros = newValue}
  }

  var opStartRelMicros: Int64 {
    get {return _storage._opStartRelMicros}
    set {_uniqueStorage()._opStartRelMicros = newValue}
  }

  var opEndRelMicros: Int64 {
    get {return _storage._opEndRelMicros}
    set {_uniqueStorage()._opEndRelMicros = newValue}
  }

  var allEndRelMicros: Int64 {
    get {return _storage._allEndRelMicros}
    set {_uniqueStorage()._allEndRelMicros = newValue}
  }

  var memory: [Tensorboard_AllocatorMemoryUsed] {
    get {return _storage._memory}
    set {_uniqueStorage()._memory = newValue}
  }

  var output: [Tensorboard_NodeOutput] {
    get {return _storage._output}
    set {_uniqueStorage()._output = newValue}
  }

  var timelineLabel: String {
    get {return _storage._timelineLabel}
    set {_uniqueStorage()._timelineLabel = newValue}
  }

  var scheduledMicros: Int64 {
    get {return _storage._scheduledMicros}
    set {_uniqueStorage()._scheduledMicros = newValue}
  }

  var threadID: UInt32 {
    get {return _storage._threadID}
    set {_uniqueStorage()._threadID = newValue}
  }

  var referencedTensor: [Tensorboard_AllocationDescription] {
    get {return _storage._referencedTensor}
    set {_uniqueStorage()._referencedTensor = newValue}
  }

  var memoryStats: Tensorboard_MemoryStats {
    get {return _storage._memoryStats ?? Tensorboard_MemoryStats()}
    set {_uniqueStorage()._memoryStats = newValue}
  }
  /// Returns true if `memoryStats` has been explicitly set.
  var hasMemoryStats: Bool {return _storage._memoryStats != nil}
  /// Clears the value of `memoryStats`. Subsequent reads from it will return its default value.
  mutating func clearMemoryStats() {_uniqueStorage()._memoryStats = nil}

  var allStartNanos: Int64 {
    get {return _storage._allStartNanos}
    set {_uniqueStorage()._allStartNanos = newValue}
  }

  var opStartRelNanos: Int64 {
    get {return _storage._opStartRelNanos}
    set {_uniqueStorage()._opStartRelNanos = newValue}
  }

  var opEndRelNanos: Int64 {
    get {return _storage._opEndRelNanos}
    set {_uniqueStorage()._opEndRelNanos = newValue}
  }

  var allEndRelNanos: Int64 {
    get {return _storage._allEndRelNanos}
    set {_uniqueStorage()._allEndRelNanos = newValue}
  }

  var scheduledNanos: Int64 {
    get {return _storage._scheduledNanos}
    set {_uniqueStorage()._scheduledNanos = newValue}
  }

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}

  fileprivate var _storage = _StorageClass.defaultInstance
}

struct Tensorboard_DeviceStepStats: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var device: String = String()

  var nodeStats: [Tensorboard_NodeExecStats] = []

  /// Its key is thread id.
  var threadNames: Dictionary<UInt32,String> = [:]

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

struct Tensorboard_StepStats: Sendable {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var devStats: [Tensorboard_DeviceStepStats] = []

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorboard"

extension Tensorboard_AllocationRecord: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".AllocationRecord"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "alloc_micros"),
    2: .standard(proto: "alloc_bytes"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularInt64Field(value: &self.allocMicros) }()
      case 2: try { try decoder.decodeSingularInt64Field(value: &self.allocBytes) }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.allocMicros != 0 {
      try visitor.visitSingularInt64Field(value: self.allocMicros, fieldNumber: 1)
    }
    if self.allocBytes != 0 {
      try visitor.visitSingularInt64Field(value: self.allocBytes, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: Tensorboard_AllocationRecord, rhs: Tensorboard_AllocationRecord) -> Bool {
    if lhs.allocMicros != rhs.allocMicros {return false}
    if lhs.allocBytes != rhs.allocBytes {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Tensorboard_AllocatorMemoryUsed: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".AllocatorMemoryUsed"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "allocator_name"),
    2: .standard(proto: "total_bytes"),
    3: .standard(proto: "peak_bytes"),
    4: .standard(proto: "live_bytes"),
    6: .standard(proto: "allocation_records"),
    5: .standard(proto: "allocator_bytes_in_use"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularStringField(value: &self.allocatorName) }()
      case 2: try { try decoder.decodeSingularInt64Field(value: &self.totalBytes) }()
      case 3: try { try decoder.decodeSingularInt64Field(value: &self.peakBytes) }()
      case 4: try { try decoder.decodeSingularInt64Field(value: &self.liveBytes) }()
      case 5: try { try decoder.decodeSingularInt64Field(value: &self.allocatorBytesInUse) }()
      case 6: try { try decoder.decodeRepeatedMessageField(value: &self.allocationRecords) }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.allocatorName.isEmpty {
      try visitor.visitSingularStringField(value: self.allocatorName, fieldNumber: 1)
    }
    if self.totalBytes != 0 {
      try visitor.visitSingularInt64Field(value: self.totalBytes, fieldNumber: 2)
    }
    if self.peakBytes != 0 {
      try visitor.visitSingularInt64Field(value: self.peakBytes, fieldNumber: 3)
    }
    if self.liveBytes != 0 {
      try visitor.visitSingularInt64Field(value: self.liveBytes, fieldNumber: 4)
    }
    if self.allocatorBytesInUse != 0 {
      try visitor.visitSingularInt64Field(value: self.allocatorBytesInUse, fieldNumber: 5)
    }
    if !self.allocationRecords.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.allocationRecords, fieldNumber: 6)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: Tensorboard_AllocatorMemoryUsed, rhs: Tensorboard_AllocatorMemoryUsed) -> Bool {
    if lhs.allocatorName != rhs.allocatorName {return false}
    if lhs.totalBytes != rhs.totalBytes {return false}
    if lhs.peakBytes != rhs.peakBytes {return false}
    if lhs.liveBytes != rhs.liveBytes {return false}
    if lhs.allocationRecords != rhs.allocationRecords {return false}
    if lhs.allocatorBytesInUse != rhs.allocatorBytesInUse {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Tensorboard_NodeOutput: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".NodeOutput"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "slot"),
    3: .standard(proto: "tensor_description"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularInt32Field(value: &self.slot) }()
      case 3: try { try decoder.decodeSingularMessageField(value: &self._tensorDescription) }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    if self.slot != 0 {
      try visitor.visitSingularInt32Field(value: self.slot, fieldNumber: 1)
    }
    try { if let v = self._tensorDescription {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
    } }()
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: Tensorboard_NodeOutput, rhs: Tensorboard_NodeOutput) -> Bool {
    if lhs.slot != rhs.slot {return false}
    if lhs._tensorDescription != rhs._tensorDescription {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Tensorboard_MemoryStats: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".MemoryStats"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "temp_memory_size"),
    3: .standard(proto: "persistent_memory_size"),
    5: .standard(proto: "persistent_tensor_alloc_ids"),
    2: .standard(proto: "device_temp_memory_size"),
    4: .standard(proto: "device_persistent_memory_size"),
    6: .standard(proto: "device_persistent_tensor_alloc_ids"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularInt64Field(value: &self.tempMemorySize) }()
      case 2: try { try decoder.decodeSingularInt64Field(value: &self.deviceTempMemorySize) }()
      case 3: try { try decoder.decodeSingularInt64Field(value: &self.persistentMemorySize) }()
      case 4: try { try decoder.decodeSingularInt64Field(value: &self.devicePersistentMemorySize) }()
      case 5: try { try decoder.decodeRepeatedInt64Field(value: &self.persistentTensorAllocIds) }()
      case 6: try { try decoder.decodeRepeatedInt64Field(value: &self.devicePersistentTensorAllocIds) }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.tempMemorySize != 0 {
      try visitor.visitSingularInt64Field(value: self.tempMemorySize, fieldNumber: 1)
    }
    if self.deviceTempMemorySize != 0 {
      try visitor.visitSingularInt64Field(value: self.deviceTempMemorySize, fieldNumber: 2)
    }
    if self.persistentMemorySize != 0 {
      try visitor.visitSingularInt64Field(value: self.persistentMemorySize, fieldNumber: 3)
    }
    if self.devicePersistentMemorySize != 0 {
      try visitor.visitSingularInt64Field(value: self.devicePersistentMemorySize, fieldNumber: 4)
    }
    if !self.persistentTensorAllocIds.isEmpty {
      try visitor.visitPackedInt64Field(value: self.persistentTensorAllocIds, fieldNumber: 5)
    }
    if !self.devicePersistentTensorAllocIds.isEmpty {
      try visitor.visitPackedInt64Field(value: self.devicePersistentTensorAllocIds, fieldNumber: 6)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: Tensorboard_MemoryStats, rhs: Tensorboard_MemoryStats) -> Bool {
    if lhs.tempMemorySize != rhs.tempMemorySize {return false}
    if lhs.persistentMemorySize != rhs.persistentMemorySize {return false}
    if lhs.persistentTensorAllocIds != rhs.persistentTensorAllocIds {return false}
    if lhs.deviceTempMemorySize != rhs.deviceTempMemorySize {return false}
    if lhs.devicePersistentMemorySize != rhs.devicePersistentMemorySize {return false}
    if lhs.devicePersistentTensorAllocIds != rhs.devicePersistentTensorAllocIds {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Tensorboard_NodeExecStats: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".NodeExecStats"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "node_name"),
    2: .standard(proto: "all_start_micros"),
    3: .standard(proto: "op_start_rel_micros"),
    4: .standard(proto: "op_end_rel_micros"),
    5: .standard(proto: "all_end_rel_micros"),
    6: .same(proto: "memory"),
    7: .same(proto: "output"),
    8: .standard(proto: "timeline_label"),
    9: .standard(proto: "scheduled_micros"),
    10: .standard(proto: "thread_id"),
    11: .standard(proto: "referenced_tensor"),
    12: .standard(proto: "memory_stats"),
    13: .standard(proto: "all_start_nanos"),
    14: .standard(proto: "op_start_rel_nanos"),
    15: .standard(proto: "op_end_rel_nanos"),
    16: .standard(proto: "all_end_rel_nanos"),
    17: .standard(proto: "scheduled_nanos"),
  ]

  fileprivate class _StorageClass {
    var _nodeName: String = String()
    var _allStartMicros: Int64 = 0
    var _opStartRelMicros: Int64 = 0
    var _opEndRelMicros: Int64 = 0
    var _allEndRelMicros: Int64 = 0
    var _memory: [Tensorboard_AllocatorMemoryUsed] = []
    var _output: [Tensorboard_NodeOutput] = []
    var _timelineLabel: String = String()
    var _scheduledMicros: Int64 = 0
    var _threadID: UInt32 = 0
    var _referencedTensor: [Tensorboard_AllocationDescription] = []
    var _memoryStats: Tensorboard_MemoryStats? = nil
    var _allStartNanos: Int64 = 0
    var _opStartRelNanos: Int64 = 0
    var _opEndRelNanos: Int64 = 0
    var _allEndRelNanos: Int64 = 0
    var _scheduledNanos: Int64 = 0

    #if swift(>=5.10)
      // This property is used as the initial default value for new instances of the type.
      // The type itself is protecting the reference to its storage via CoW semantics.
      // This will force a copy to be made of this reference when the first mutation occurs;
      // hence, it is safe to mark this as `nonisolated(unsafe)`.
      static nonisolated(unsafe) let defaultInstance = _StorageClass()
    #else
      static let defaultInstance = _StorageClass()
    #endif

    private init() {}

    init(copying source: _StorageClass) {
      _nodeName = source._nodeName
      _allStartMicros = source._allStartMicros
      _opStartRelMicros = source._opStartRelMicros
      _opEndRelMicros = source._opEndRelMicros
      _allEndRelMicros = source._allEndRelMicros
      _memory = source._memory
      _output = source._output
      _timelineLabel = source._timelineLabel
      _scheduledMicros = source._scheduledMicros
      _threadID = source._threadID
      _referencedTensor = source._referencedTensor
      _memoryStats = source._memoryStats
      _allStartNanos = source._allStartNanos
      _opStartRelNanos = source._opStartRelNanos
      _opEndRelNanos = source._opEndRelNanos
      _allEndRelNanos = source._allEndRelNanos
      _scheduledNanos = source._scheduledNanos
    }
  }

  fileprivate mutating func _uniqueStorage() -> _StorageClass {
    if !isKnownUniquelyReferenced(&_storage) {
      _storage = _StorageClass(copying: _storage)
    }
    return _storage
  }

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    _ = _uniqueStorage()
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      while let fieldNumber = try decoder.nextFieldNumber() {
        // The use of inline closures is to circumvent an issue where the compiler
        // allocates stack space for every case branch when no optimizations are
        // enabled. https://github.com/apple/swift-protobuf/issues/1034
        switch fieldNumber {
        case 1: try { try decoder.decodeSingularStringField(value: &_storage._nodeName) }()
        case 2: try { try decoder.decodeSingularInt64Field(value: &_storage._allStartMicros) }()
        case 3: try { try decoder.decodeSingularInt64Field(value: &_storage._opStartRelMicros) }()
        case 4: try { try decoder.decodeSingularInt64Field(value: &_storage._opEndRelMicros) }()
        case 5: try { try decoder.decodeSingularInt64Field(value: &_storage._allEndRelMicros) }()
        case 6: try { try decoder.decodeRepeatedMessageField(value: &_storage._memory) }()
        case 7: try { try decoder.decodeRepeatedMessageField(value: &_storage._output) }()
        case 8: try { try decoder.decodeSingularStringField(value: &_storage._timelineLabel) }()
        case 9: try { try decoder.decodeSingularInt64Field(value: &_storage._scheduledMicros) }()
        case 10: try { try decoder.decodeSingularUInt32Field(value: &_storage._threadID) }()
        case 11: try { try decoder.decodeRepeatedMessageField(value: &_storage._referencedTensor) }()
        case 12: try { try decoder.decodeSingularMessageField(value: &_storage._memoryStats) }()
        case 13: try { try decoder.decodeSingularInt64Field(value: &_storage._allStartNanos) }()
        case 14: try { try decoder.decodeSingularInt64Field(value: &_storage._opStartRelNanos) }()
        case 15: try { try decoder.decodeSingularInt64Field(value: &_storage._opEndRelNanos) }()
        case 16: try { try decoder.decodeSingularInt64Field(value: &_storage._allEndRelNanos) }()
        case 17: try { try decoder.decodeSingularInt64Field(value: &_storage._scheduledNanos) }()
        default: break
        }
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every if/case branch local when no optimizations
      // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
      // https://github.com/apple/swift-protobuf/issues/1182
      if !_storage._nodeName.isEmpty {
        try visitor.visitSingularStringField(value: _storage._nodeName, fieldNumber: 1)
      }
      if _storage._allStartMicros != 0 {
        try visitor.visitSingularInt64Field(value: _storage._allStartMicros, fieldNumber: 2)
      }
      if _storage._opStartRelMicros != 0 {
        try visitor.visitSingularInt64Field(value: _storage._opStartRelMicros, fieldNumber: 3)
      }
      if _storage._opEndRelMicros != 0 {
        try visitor.visitSingularInt64Field(value: _storage._opEndRelMicros, fieldNumber: 4)
      }
      if _storage._allEndRelMicros != 0 {
        try visitor.visitSingularInt64Field(value: _storage._allEndRelMicros, fieldNumber: 5)
      }
      if !_storage._memory.isEmpty {
        try visitor.visitRepeatedMessageField(value: _storage._memory, fieldNumber: 6)
      }
      if !_storage._output.isEmpty {
        try visitor.visitRepeatedMessageField(value: _storage._output, fieldNumber: 7)
      }
      if !_storage._timelineLabel.isEmpty {
        try visitor.visitSingularStringField(value: _storage._timelineLabel, fieldNumber: 8)
      }
      if _storage._scheduledMicros != 0 {
        try visitor.visitSingularInt64Field(value: _storage._scheduledMicros, fieldNumber: 9)
      }
      if _storage._threadID != 0 {
        try visitor.visitSingularUInt32Field(value: _storage._threadID, fieldNumber: 10)
      }
      if !_storage._referencedTensor.isEmpty {
        try visitor.visitRepeatedMessageField(value: _storage._referencedTensor, fieldNumber: 11)
      }
      try { if let v = _storage._memoryStats {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 12)
      } }()
      if _storage._allStartNanos != 0 {
        try visitor.visitSingularInt64Field(value: _storage._allStartNanos, fieldNumber: 13)
      }
      if _storage._opStartRelNanos != 0 {
        try visitor.visitSingularInt64Field(value: _storage._opStartRelNanos, fieldNumber: 14)
      }
      if _storage._opEndRelNanos != 0 {
        try visitor.visitSingularInt64Field(value: _storage._opEndRelNanos, fieldNumber: 15)
      }
      if _storage._allEndRelNanos != 0 {
        try visitor.visitSingularInt64Field(value: _storage._allEndRelNanos, fieldNumber: 16)
      }
      if _storage._scheduledNanos != 0 {
        try visitor.visitSingularInt64Field(value: _storage._scheduledNanos, fieldNumber: 17)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: Tensorboard_NodeExecStats, rhs: Tensorboard_NodeExecStats) -> Bool {
    if lhs._storage !== rhs._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((lhs._storage, rhs._storage)) { (_args: (_StorageClass, _StorageClass)) in
        let _storage = _args.0
        let rhs_storage = _args.1
        if _storage._nodeName != rhs_storage._nodeName {return false}
        if _storage._allStartMicros != rhs_storage._allStartMicros {return false}
        if _storage._opStartRelMicros != rhs_storage._opStartRelMicros {return false}
        if _storage._opEndRelMicros != rhs_storage._opEndRelMicros {return false}
        if _storage._allEndRelMicros != rhs_storage._allEndRelMicros {return false}
        if _storage._memory != rhs_storage._memory {return false}
        if _storage._output != rhs_storage._output {return false}
        if _storage._timelineLabel != rhs_storage._timelineLabel {return false}
        if _storage._scheduledMicros != rhs_storage._scheduledMicros {return false}
        if _storage._threadID != rhs_storage._threadID {return false}
        if _storage._referencedTensor != rhs_storage._referencedTensor {return false}
        if _storage._memoryStats != rhs_storage._memoryStats {return false}
        if _storage._allStartNanos != rhs_storage._allStartNanos {return false}
        if _storage._opStartRelNanos != rhs_storage._opStartRelNanos {return false}
        if _storage._opEndRelNanos != rhs_storage._opEndRelNanos {return false}
        if _storage._allEndRelNanos != rhs_storage._allEndRelNanos {return false}
        if _storage._scheduledNanos != rhs_storage._scheduledNanos {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Tensorboard_DeviceStepStats: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".DeviceStepStats"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "device"),
    2: .standard(proto: "node_stats"),
    3: .standard(proto: "thread_names"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularStringField(value: &self.device) }()
      case 2: try { try decoder.decodeRepeatedMessageField(value: &self.nodeStats) }()
      case 3: try { try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufUInt32,SwiftProtobuf.ProtobufString>.self, value: &self.threadNames) }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.device.isEmpty {
      try visitor.visitSingularStringField(value: self.device, fieldNumber: 1)
    }
    if !self.nodeStats.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.nodeStats, fieldNumber: 2)
    }
    if !self.threadNames.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufUInt32,SwiftProtobuf.ProtobufString>.self, value: self.threadNames, fieldNumber: 3)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: Tensorboard_DeviceStepStats, rhs: Tensorboard_DeviceStepStats) -> Bool {
    if lhs.device != rhs.device {return false}
    if lhs.nodeStats != rhs.nodeStats {return false}
    if lhs.threadNames != rhs.threadNames {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension Tensorboard_StepStats: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".StepStats"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "dev_stats"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeRepeatedMessageField(value: &self.devStats) }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.devStats.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.devStats, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: Tensorboard_StepStats, rhs: Tensorboard_StepStats) -> Bool {
    if lhs.devStats != rhs.devStats {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
