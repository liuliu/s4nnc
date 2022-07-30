import Foundation
import SystemPackage

/// A logger that writes protobuf events into a tensorboard-readable file.
struct EventLogger {
  /// File descriptor.
  private let fd: FileDescriptor

  /// Creates an instance with log located at `logDirectory`; creates
  /// the log file and add an initial event as well.
  init(logDirectory: String) throws {
    // Create the directory if it is missing.
    try FileManager.default.createDirectory(atPath: logDirectory, withIntermediateDirectories: true)

    // Create the file.
    let timestamp = Date().timeIntervalSince1970
    let filePath = URL(fileURLWithPath: logDirectory, isDirectory: true).appendingPathComponent(
      "events.out.tfevents." + String(timestamp).split(separator: ".")[0]
    ).path

    fd = try FileDescriptor.open(
      filePath, .writeOnly, options: [.create, .truncate], permissions: .ownerReadWrite)
    // Add an initial event.
    var initialEvent = Tensorboard_Event()
    initialEvent.wallTime = timestamp
    initialEvent.fileVersion = "brain.Event:2"
    try add(initialEvent)
  }

  func close() throws {
    try fd.close()
  }

  /// Add an event to the log.
  func add(_ event: Tensorboard_Event) throws {
    let data: Data = try event.serializedData()
    var header: Data = Data()
    header.append(contentsOf: UInt64(data.count).littleEndianBuffer)
    var payload = header
    payload.append(contentsOf: header.maskedCRC32C().littleEndianBuffer)
    payload.append(contentsOf: data)
    payload.append(contentsOf: data.maskedCRC32C().littleEndianBuffer)
    try fd.writeAll(payload)
  }
}
