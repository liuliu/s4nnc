import Foundation

/// A writer for writing model execution summaries to a tensorboard-readable file; the
/// summaries include scalars for logging statistics, graphs for visualizing model etc.
public struct SummaryWriter {
  /// Logger for writing the summaries as protobuf events to the file.
  let eventLogger: EventLogger

  /// Creates an instance with log located at `logDirectory`.
  public init(logDirectory: String) {
    eventLogger = try! EventLogger(logDirectory: logDirectory)
  }

  public func close() throws {
    try eventLogger.close()
  }
}

extension SummaryWriter {
  /// Add training and validation statistics for tensorboard scalars dashboard.
  public func addScalar(
    _ tag: String, _ value: Float, step: Int,
    wallTime: Double = Date().timeIntervalSince1970, displayName: String? = nil,
    description: String? = nil
  ) {
    var summaryMetadata = Tensorboard_SummaryMetadata()
    summaryMetadata.displayName = displayName ?? tag
    summaryMetadata.summaryDescription = description ?? ""

    var summaryValue = Tensorboard_Summary.Value()
    summaryValue.tag = tag
    summaryValue.simpleValue = value
    summaryValue.metadata = summaryMetadata

    var summary = Tensorboard_Summary()
    summary.value = [summaryValue]

    var event = Tensorboard_Event()
    event.summary = summary
    event.wallTime = wallTime
    event.step = Int64(step)
    do {
      try eventLogger.add(event)
    } catch {
      fatalError("Could not add \(event) to log: \(error)")
    }
  }
}
