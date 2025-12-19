#if canImport(C_ccv)
import C_ccv
#endif

#if canImport(C_swiftpm_ccv)
import C_swiftpm_ccv
#endif

import Foundation
import MuJoCo
import NIOCore
import NIOHTTP1
import NIOPosix
import NIOWebSocket

public protocol HTTPRenderProvider {
  var renderContextCallback: ((_: inout MjrContext, _ width: Int32, _ height: Int32) -> Void)? {
    get set
  }
  func sendEvent(_: GLContext.Event)
}

extension Simulate: HTTPRenderProvider {}

protocol HTTPHandlerFrameProvider {
  func fetchFrame() -> Data
  mutating func register(_: EventLoopPromise<Data>)
}

protocol WebSocketHandlerEventProvider {
  func sendEvent(_: GLContext.Event)
}

public class HTTPRenderServer {
  private var frame = Data()
  private var registeredPromisesForFrame = [EventLoopPromise<Data>]()
  private let queue = DispatchQueue(label: "data", qos: .default)
  private var provider: HTTPRenderProvider
  private let streamKey: String
  private let maxWidth: Int
  private let maxHeight: Int
  private let canResize: Bool
  private let numberOfThreads: Int
  private var image: UnsafeMutablePointer<ccv_dense_matrix_t>?
  private var buffer: UnsafeMutablePointer<UInt8>

  public init(
    _ provider: HTTPRenderProvider, streamKey: String? = nil, maxWidth: Int = 1920,
    maxHeight: Int = 1080, maxFrameRate: Int = 30, canResize: Bool = true,
    numberOfThreads: Int = System.coreCount
  ) {
    self.provider = provider
    self.streamKey =
      streamKey
      ?? String(
        (0..<8).map { _ in
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".randomElement()!
        })
    self.maxWidth = maxWidth
    self.maxHeight = maxHeight
    self.canResize = canResize
    self.numberOfThreads = numberOfThreads
    image = ccv_dense_matrix_new(Int32(maxHeight), Int32(maxWidth), Int32(CCV_8U | CCV_C3), nil, 0)
    buffer = UnsafeMutablePointer<UInt8>.allocate(
      capacity: Int(image!.pointee.rows * image!.pointee.cols) * 2)
    var lasttime = GLContext.time
    self.provider.renderContextCallback = { [weak self] context, width, height in
      guard let self = self else { return }
      if self.image?.pointee.rows != height || self.image?.pointee.cols != width {
        ccv_matrix_free(self.image)
        self.image = ccv_dense_matrix_new(height, width, Int32(CCV_8U | CCV_C3), nil, 0)
        self.buffer.deallocate()
        self.buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: Int(width * height) * 2)
      }
      let nowtime = GLContext.time
      // No need to send more than 30fps.
      guard nowtime - lasttime >= 1.0 / Double(maxFrameRate) else {
        return
      }
      lasttime = nowtime
      context.readPixels(
        rgb: &self.image!.pointee.data.u8,
        viewport: MjrRect(left: 0, bottom: 0, width: width, height: height))
      ccv_flip(self.image, &self.image, 0, Int32(CCV_FLIP_Y))
      var count: Int = Int(width * height) * 2
      ccv_write(self.image, self.buffer, &count, Int32(CCV_IO_JPEG_STREAM), nil)
      self.queue.sync {
        self.frame = Data(bytes: self.buffer, count: count)
        let registeredPromisesForFrame = self.registeredPromisesForFrame
        self.registeredPromisesForFrame.removeAll()
        for promise in registeredPromisesForFrame {
          promise.succeed(self.frame)
        }
      }
    }
  }

  private var group: EventLoopGroup? = nil

  deinit {
    ccv_matrix_free(image)
    buffer.deallocate()
    try! group?.syncShutdownGracefully()
  }

  private var port: Int = 0

  public func bind(host: String, port: Int) -> EventLoopFuture<Channel> {
    let group = MultiThreadedEventLoopGroup(numberOfThreads: numberOfThreads)
    self.group = group
    self.port = port
    let upgrader = NIOWebSocketServerUpgrader(
      shouldUpgrade: { (channel: Channel, head: HTTPRequestHead) in
        channel.eventLoop.makeSucceededFuture(HTTPHeaders())
      },
      upgradePipelineHandler: { (channel: Channel, _: HTTPRequestHead) in
        channel.pipeline.addHandler(
          WebSocketHandler(
            self, maxWidth: self.maxWidth, maxHeight: self.maxHeight, canResize: self.canResize))
      })
    let socketBootstrap = ServerBootstrap(group: group)
      // Specify backlog and enable SO_REUSEADDR for the server itself
      .serverChannelOption(ChannelOptions.backlog, value: 256)
      .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)

      // Set the handlers that are applied to the accepted Channels
      .childChannelInitializer { channel in
        let httpHandler = HTTPHandler(self, streamKey: self.streamKey, canResize: self.canResize)
        let config: NIOHTTPServerUpgradeConfiguration = (
          upgraders: [upgrader],
          completionHandler: { _ in
            channel.pipeline.removeHandler(httpHandler, promise: nil)
          }
        )
        return channel.pipeline.configureHTTPServerPipeline(withServerUpgrade: config).flatMap {
          channel.pipeline.addHandler(httpHandler)
        }
      }
      // Enable SO_REUSEADDR for the accepted Channels
      .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
      .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 1)
      .childChannelOption(ChannelOptions.allowRemoteHalfClosure, value: true)
    return socketBootstrap.bind(host: host, port: port)
  }

  public var html: String {
    """
      <div id="\(self.streamKey)-motion-jpeg-div" style="margin:auto;display:block"></div>
      <script>
        var div = document.getElementById("\(self.streamKey)-motion-jpeg-div");
        div.innerHTML = '<img id="\(self.streamKey)-motion-jpeg-img" src="http://' + window.location.hostname + ':\(port)/\(streamKey).mjpg" style="margin:auto;display:block"><a href="http://' + window.location.hostname + ':\(port)" target="_blank" style="margin:auto;display:block;text-align:center;padding-top:4px">[Open in New Window]</a>';
        var script = document.createElement("script");
        script.src = "http://" + window.location.hostname + ":\(port)/\(streamKey).js";
        div.appendChild(script);
      </script>
    """
  }
}

extension HTTPRenderServer: HTTPHandlerFrameProvider {
  func register(_ promise: EventLoopPromise<Data>) {
    queue.sync {
      registeredPromisesForFrame.append(promise)
    }
  }
  func fetchFrame() -> Data {
    var data: Data? = nil
    queue.sync {
      data = frame
    }
    return data!
  }
}

extension HTTPRenderServer: WebSocketHandlerEventProvider {
  func sendEvent(_ event: GLContext.Event) {
    provider.sendEvent(event)
  }
}

private func httpResponseHead(
  request: HTTPRequestHead, status: HTTPResponseStatus, headers: HTTPHeaders = HTTPHeaders()
) -> HTTPResponseHead {
  var head = HTTPResponseHead(version: request.version, status: status, headers: headers)
  head.headers.add(name: "Connection", value: "close")
  return head
}

extension HTTPRenderServer {

  private struct JSEvent: Decodable {
    var keyCode: Int32?
    var ctrlKey: Bool?
    var altKey: Bool?
    var shiftKey: Bool?
    var mouseState: String?
    var buttons: Int32?
    var offsetX: Float?
    var offsetY: Float?
    var deltaX: Float?
    var deltaY: Float?
    var width: Int32?
    var height: Int32?
  }

  private final class HTTPHandler: ChannelInboundHandler, RemovableChannelHandler {
    public typealias InboundIn = HTTPServerRequestPart
    public typealias OutboundOut = HTTPServerResponsePart

    private enum State {
      case idle
      case waitingForRequestBody
      case sendingResponse

      mutating func requestReceived() {
        precondition(self == .idle, "Invalid state for request received: \(self)")
        self = .waitingForRequestBody
      }

      mutating func requestComplete() {
        precondition(self == .waitingForRequestBody, "Invalid state for request complete: \(self)")
        self = .sendingResponse
      }

      mutating func responseComplete() {
        precondition(self == .sendingResponse, "Invalid state for response complete: \(self)")
        self = .idle
      }
    }

    private var buffer: ByteBuffer! = nil
    private var state = State.idle
    private var continuousCount: Int = 0

    private var handler: ((ChannelHandlerContext, HTTPServerRequestPart) -> Void)?
    private var indexPageResponse: String {
      """
        <!DOCTYPE html><html lang="en"><head><meta charset="utf-8"></head><body style="margin:0"><img id="\(self.streamKey)-motion-jpeg-img" src="/\(self.streamKey).mjpg" style="margin:auto;display:block"><script src="/\(self.streamKey).js"></script>
      """
    }
    private var jsResponse: String {
      let firstPart = """
        (function () {
          var mjpeg = document.getElementById("\(self.streamKey)-motion-jpeg-img");
          var src = new URL(mjpeg.src)
          var wsconnection = new WebSocket("ws://" + src.host + "/\(self.streamKey).ws");
          var commonKeyCodes = {
            "Space": 32,
            "Quote": 39, /* ' */
            "Comma": 44, /* , */
            "Minus": 45, /* - */
            "Period": 46, /* . */
            "Slash": 47, /* / */
            "Digit0": 48,
            "Digit1": 49,
            "Digit2": 50,
            "Digit3": 51,
            "Digit4": 52,
            "Digit5": 53,
            "Digit6": 54,
            "Digit7": 55,
            "Digit8": 56,
            "Digit9": 57,
            "Semicolon": 59, /* ; */
            "Equal": 61, /* = */
            "KeyA": 65,
            "KeyB": 66,
            "KeyC": 67,
            "KeyD": 68,
            "KeyE": 69,
            "KeyF": 70,
            "KeyG": 71,
            "KeyH": 72,
            "KeyI": 73,
            "KeyJ": 74,
            "KeyK": 75,
            "KeyL": 76,
            "KeyM": 77,
            "KeyN": 78,
            "KeyO": 79,
            "KeyP": 80,
            "KeyQ": 81,
            "KeyR": 82,
            "KeyS": 83,
            "KeyT": 84,
            "KeyU": 85,
            "KeyV": 86,
            "KeyW": 87,
            "KeyX": 88,
            "KeyY": 89,
            "KeyZ": 90,
            "BracketLeft": 91, /* [ */
            "Backslash": 92, /* \\ */
            "BracketRight": 93, /* ] */
            "Backquote": 96, /* ` */
            "Escape": 256,
            "Enter": 257,
            "Tab": 258,
            "Backspace": 259,
            "Insert": 260,
            "Delete": 261,
            "ArrowRight": 262,
            "ArrowLeft": 263,
            "ArrowDown": 264,
            "ArrowUp": 265,
            "PageUp": 266,
            "PageDown": 267,
            "Home": 268,
            "End": 269,
            "CapsLock": 280,
            "ScrollLock": 281,
            "NumLock": 282,
            "PrintScreen": 283,
            "Pause": 284,
            "F1": 290,
            "F2": 291,
            "F3": 292,
            "F4": 293,
            "F5": 294,
            "F6": 295,
            "F7": 296,
            "F8": 297,
            "F9": 298,
            "F10": 299,
            "F11": 300,
            "F12": 301,
            "F13": 302,
            "F14": 303,
            "F15": 304,
            "F16": 305,
            "F17": 306,
            "F18": 307,
            "F19": 308,
            "F20": 309,
            "F21": 310,
            "F22": 311,
            "F23": 312,
            "F24": 313,
            "F25": 314,
            "Numpad0": 320,
            "Numpad1": 321,
            "Numpad2": 322,
            "Numpad3": 323,
            "Numpad4": 324,
            "Numpad5": 325,
            "Numpad6": 326,
            "Numpad7": 327,
            "Numpad8": 328,
            "Numpad9": 329,
            "NumpadDecimal": 330,
            "NumpadDivide": 331,
            "NumpadMultiply": 332,
            "NumpadSubstract": 333,
            "NumpadAdd": 334,
            "NumpadEnter": 335,
            "NumpadEqual": 336,
            "ShiftLeft": 340,
            "ControlLeft": 341,
            "AltLeft": 342,
            "ShiftRight": 344,
            "ControlRight": 345,
            "AltRight": 346
          };
          function onKeydown(e) {
            e.preventDefault();
            wsconnection.send(JSON.stringify({"keyCode": commonKeyCodes[e.code], "ctrlKey": e.ctrlKey, "altKey": e.altKey, "shiftKey": e.shiftKey}));
          }
          mjpeg.addEventListener("mouseenter", function (e) {
            e.preventDefault();
            wsconnection.send(JSON.stringify({"mouseState": "move", "buttons": e.buttons, "offsetX": e.offsetX, "offsetY": e.offsetY, "ctrlKey": e.ctrlKey, "altKey": e.altKey, "shiftKey": e.shiftKey}));
            window.addEventListener("keydown", onKeydown);
          });
          mjpeg.addEventListener("mouseleave", function (e) {
            window.removeEventListener("keydown", onKeydown);
          });
          mjpeg.addEventListener("mousemove", function (e) {
            e.preventDefault();
            wsconnection.send(JSON.stringify({"mouseState": "move", "buttons": e.buttons, "offsetX": e.offsetX, "offsetY": e.offsetY, "ctrlKey": e.ctrlKey, "altKey": e.altKey, "shiftKey": e.shiftKey}));
          });
          mjpeg.addEventListener("mousedown", function (e) {
            e.preventDefault();
            wsconnection.send(JSON.stringify({"mouseState": "press", "buttons": e.buttons, "offsetX": e.offsetX, "offsetY": e.offsetY, "ctrlKey": e.ctrlKey, "altKey": e.altKey, "shiftKey": e.shiftKey}));
          });
          mjpeg.addEventListener("mouseup", function (e) {
            e.preventDefault();
            wsconnection.send(JSON.stringify({"mouseState": "release", "buttons": e.buttons, "offsetX": e.offsetX, "offsetY": e.offsetY, "ctrlKey": e.ctrlKey, "altKey": e.altKey, "shiftKey": e.shiftKey}));
          });
          mjpeg.addEventListener("wheel", function (e) {
            e.preventDefault();
            wsconnection.send(JSON.stringify({"deltaX": e.deltaX / 16, "deltaY": -e.deltaY / 16}));
          });
          mjpeg.addEventListener("contextmenu", function (e) {
            e.preventDefault();
            e.stopPropagation();
          });
          mjpeg.addEventListener("dragstart", function (e) {
            e.preventDefault();
          });
          mjpeg.addEventListener("drop", function (e) {
            e.preventDefault();
          });
        """
      if self.canResize {
        return firstPart + """
          window.addEventListener("resize", function () {
            wsconnection.send(JSON.stringify({"width": Math.floor(window.innerWidth / 16) * 16, "height": Math.floor(window.innerHeight / 16) * 16}));
          });
          wsconnection.addEventListener("open", function () {
            wsconnection.send(JSON.stringify({"width": Math.floor(window.innerWidth / 16) * 16, "height": Math.floor(window.innerHeight / 16) * 16}));
          });
          })();
          """
      } else {
        return firstPart + "\n})();"
      }
    }

    private var frameProvider: HTTPHandlerFrameProvider
    private let streamKey: String
    private let canResize: Bool

    public init(_ frameProvider: HTTPHandlerFrameProvider, streamKey: String, canResize: Bool) {
      self.frameProvider = frameProvider
      self.streamKey = streamKey
      self.canResize = canResize
    }

    func handleContinuousWrites(
      context: ChannelHandlerContext, request: HTTPServerRequestPart
    ) {
      switch request {
      case .head(let request):
        self.continuousCount = 0
        self.state.requestReceived()
        func doNext(_ frame: Data) {
          guard self.buffer != nil else { return }
          self.buffer.clear()
          let promise = context.eventLoop.makePromise(of: Data.self)
          let future = promise.futureResult
          self.buffer.writeStaticString("--FRAME\r\n")
          self.buffer.writeStaticString("Content-Type: image/jpeg\r\n")
          self.buffer.writeString("Content-Length: \(frame.count)\r\n")
          self.buffer.writeStaticString("\r\n")
          self.buffer.writeBytes(frame)
          self.frameProvider.register(promise)
          self.buffer.writeStaticString("\r\n")
          context.writeAndFlush(self.wrapOutboundOut(.body(.byteBuffer(self.buffer)))).whenSuccess {
            future.whenSuccess(doNext)
          }
        }
        var responseHead = httpResponseHead(request: request, status: .ok)
        responseHead.headers.add(
          name: "Content-Type", value: "multipart/x-mixed-replace; boundary=FRAME")
        context.writeAndFlush(self.wrapOutboundOut(.head(responseHead)), promise: nil)
        doNext(frameProvider.fetchFrame())
      case .end:
        self.state.requestComplete()
      default:
        break
      }
    }

    func mjpgHandler(request reqHead: HTTPRequestHead) -> (
      (ChannelHandlerContext, HTTPServerRequestPart) -> Void
    )? {
      return {
        self.handleContinuousWrites(context: $0, request: $1)
      }
    }

    private func completeResponse(
      _ context: ChannelHandlerContext, trailers: HTTPHeaders?, promise: EventLoopPromise<Void>?
    ) {
      self.state.responseComplete()
      let promise = promise ?? context.eventLoop.makePromise()
      promise.futureResult.whenComplete { (_: Result<Void, Error>) in context.close(promise: nil) }
      self.handler = nil
      context.writeAndFlush(self.wrapOutboundOut(.end(trailers)), promise: promise)
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
      let reqPart = self.unwrapInboundIn(data)
      if let handler = self.handler {
        handler(context, reqPart)
        return
      }

      switch reqPart {
      case .head(let request):

        if request.uri == "/\(self.streamKey).mjpg" {
          self.handler = self.mjpgHandler(request: request)
          self.handler!(context, reqPart)
          return
        } else if request.uri == "/" {
          self.state.requestReceived()
          var responseHead = httpResponseHead(request: request, status: .ok)
          self.buffer.clear()
          self.buffer.writeString(self.indexPageResponse)
          responseHead.headers.add(name: "Content-Length", value: "\(self.buffer!.readableBytes)")
          responseHead.headers.add(name: "Content-Type", value: "text/html; charset=utf-8")
          let response = HTTPServerResponsePart.head(responseHead)
          context.write(self.wrapOutboundOut(response), promise: nil)
          return
        } else if request.uri == "/\(self.streamKey).js" {
          self.state.requestReceived()
          var responseHead = httpResponseHead(request: request, status: .ok)
          self.buffer.clear()
          self.buffer.writeString(self.jsResponse)
          responseHead.headers.add(name: "Content-Length", value: "\(self.buffer!.readableBytes)")
          responseHead.headers.add(name: "Content-Type", value: "text/javascript; charset=utf-8")
          let response = HTTPServerResponsePart.head(responseHead)
          context.write(self.wrapOutboundOut(response), promise: nil)
          return
        }
        self.state.requestReceived()
        var responseHead = httpResponseHead(request: request, status: .notFound)
        self.buffer.clear()
        self.buffer.writeString("Not Found")
        responseHead.headers.add(name: "Content-Length", value: "\(self.buffer!.readableBytes)")
        responseHead.headers.add(name: "Content-Type", value: "text/html; charset=utf-8")
        let response = HTTPServerResponsePart.head(responseHead)
        context.write(self.wrapOutboundOut(response), promise: nil)
      case .body:
        break
      case .end:
        self.state.requestComplete()
        let content = HTTPServerResponsePart.body(.byteBuffer(buffer!.slice()))
        context.write(self.wrapOutboundOut(content), promise: nil)
        self.completeResponse(context, trailers: nil, promise: nil)
      }
    }

    func channelReadComplete(context: ChannelHandlerContext) {
      context.flush()
    }

    func handlerAdded(context: ChannelHandlerContext) {
      self.buffer = context.channel.allocator.buffer(capacity: 0)
    }

    func handlerRemoved(context: ChannelHandlerContext) {
      self.buffer = nil
    }

    func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
      switch event {
      case let evt as ChannelEvent where evt == ChannelEvent.inputClosed:
        // The remote peer half-closed the channel. At this time, any
        // outstanding response will now get the channel closed, and
        // if we are idle or waiting for a request body to finish we
        // will close the channel immediately.
        switch self.state {
        case .idle, .waitingForRequestBody:
          context.close(promise: nil)
        case .sendingResponse:
          break
        }
      default:
        context.fireUserInboundEventTriggered(event)
      }
    }
  }

  private final class WebSocketHandler: ChannelInboundHandler {
    typealias InboundIn = WebSocketFrame
    typealias OutboundOut = WebSocketFrame

    private var awaitingClose: Bool = false
    private let decoder = JSONDecoder()

    private let eventProvider: WebSocketHandlerEventProvider
    private let maxWidth: Int
    private let maxHeight: Int
    private let canResize: Bool

    public init(
      _ eventProvider: WebSocketHandlerEventProvider, maxWidth: Int, maxHeight: Int, canResize: Bool
    ) {
      self.eventProvider = eventProvider
      self.maxWidth = maxWidth
      self.maxHeight = maxHeight
      self.canResize = canResize
    }

    public func handlerAdded(context: ChannelHandlerContext) {
    }

    public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
      let frame = self.unwrapInboundIn(data)

      switch frame.opcode {
      case .connectionClose:
        self.receivedClose(context: context, frame: frame)
      case .ping:
        self.pong(context: context, frame: frame)
      case .text:
        var data = frame.unmaskedData
        let text = data.readString(length: data.readableBytes) ?? ""
        if let textData = text.data(using: .utf8),
          let jsEvent = try? decoder.decode(JSEvent.self, from: textData)
        {
          if let keyCode = jsEvent.keyCode {
            eventProvider.sendEvent(
              .keyboard(
                .init(
                  keyCode: keyCode, control: jsEvent.ctrlKey ?? false,
                  shift: jsEvent.shiftKey ?? false, alt: jsEvent.altKey ?? false)))
          } else if let mouseState = jsEvent.mouseState, let buttons = jsEvent.buttons,
            let offsetX = jsEvent.offsetX, let offsetY = jsEvent.offsetY
          {
            switch mouseState {
            case "press":
              eventProvider.sendEvent(
                .mouse(
                  .init(
                    state: .press, x: offsetX, y: offsetY, left: (buttons | 1 == 1),
                    right: (buttons | 2 == 2), middle: (buttons | 4 == 4),
                    control: jsEvent.ctrlKey ?? false, shift: jsEvent.shiftKey ?? false,
                    alt: jsEvent.altKey ?? false)))
            case "release":
              eventProvider.sendEvent(
                .mouse(
                  .init(
                    state: .release, x: offsetX, y: offsetY, left: (buttons | 1 == 1),
                    right: (buttons | 2 == 2), middle: (buttons | 4 == 4),
                    control: jsEvent.ctrlKey ?? false, shift: jsEvent.shiftKey ?? false,
                    alt: jsEvent.altKey ?? false)))
            case "move":
              eventProvider.sendEvent(
                .mouse(
                  .init(
                    state: .move, x: offsetX, y: offsetY, left: (buttons | 1 == 1),
                    right: (buttons | 2 == 2), middle: (buttons | 4 == 4),
                    control: jsEvent.ctrlKey ?? false, shift: jsEvent.shiftKey ?? false,
                    alt: jsEvent.altKey ?? false)))
            default:
              break
            }
          } else if let deltaX = jsEvent.deltaX, let deltaY = jsEvent.deltaY {
            eventProvider.sendEvent(.scroll(.init(sx: deltaX, sy: deltaY)))
          } else if canResize, let width = jsEvent.width, let height = jsEvent.height {
            eventProvider.sendEvent(
              .resize(
                .init(width: min(width, Int32(maxWidth)), height: min(height, Int32(maxHeight)))))
          }
        }
      case .binary, .continuation, .pong:
        // We ignore these frames.
        break
      default:
        // Unknown frames are errors.
        self.closeOnError(context: context)
      }
    }

    public func channelReadComplete(context: ChannelHandlerContext) {
      context.flush()
    }

    private func receivedClose(context: ChannelHandlerContext, frame: WebSocketFrame) {
      // Handle a received close frame. In websockets, we're just going to send the close
      // frame and then close, unless we already sent our own close frame.
      if awaitingClose {
        // Cool, we started the close and were waiting for the user. We're done.
        context.close(promise: nil)
      } else {
        // This is an unsolicited close. We're going to send a response frame and
        // then, when we've sent it, close up shop. We should send back the close code the remote
        // peer sent us, unless they didn't send one at all.
        var data = frame.unmaskedData
        let closeDataCode = data.readSlice(length: 2) ?? ByteBuffer()
        let closeFrame = WebSocketFrame(fin: true, opcode: .connectionClose, data: closeDataCode)
        _ = context.write(self.wrapOutboundOut(closeFrame)).map { () in
          context.close(promise: nil)
        }
      }
    }

    private func pong(context: ChannelHandlerContext, frame: WebSocketFrame) {
      var frameData = frame.data
      let maskingKey = frame.maskKey

      if let maskingKey = maskingKey {
        frameData.webSocketUnmask(maskingKey)
      }

      let responseFrame = WebSocketFrame(fin: true, opcode: .pong, data: frameData)
      context.write(self.wrapOutboundOut(responseFrame), promise: nil)
    }

    private func closeOnError(context: ChannelHandlerContext) {
      // We have hit an error, we want to close. We do that by sending a close frame and then
      // shutting down the write side of the connection.
      var data = context.channel.allocator.buffer(capacity: 2)
      data.write(webSocketErrorCode: .protocolError)
      let frame = WebSocketFrame(fin: true, opcode: .connectionClose, data: data)
      context.write(self.wrapOutboundOut(frame)).whenComplete { (_: Result<Void, Error>) in
        context.close(mode: .output, promise: nil)
      }
      awaitingClose = true
    }
  }
}
