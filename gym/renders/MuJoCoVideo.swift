import C_ccv
import C_ffmpeg
import Foundation
import Gym
import MuJoCo

public final class MuJoCoVideo<EnvType: MuJoCoEnv> {
  var env: EnvType
  var filePath: String
  var framesPerSecond: Int
  var simulate: Simulate
  var cpugenesis: Double = 0
  var simgenesis: Double = 0
  var videoLock = pthread_mutex_t()
  var formatContext: UnsafeMutablePointer<AVFormatContext>? = nil
  var videoCodec: UnsafeMutablePointer<AVCodec>? = nil
  var videoStream: UnsafeMutablePointer<AVStream>? = nil
  var videoPacket: UnsafeMutablePointer<AVPacket>? = nil
  var videoFrame: UnsafeMutablePointer<AVFrame>? = nil
  var videoYUVFrame: UnsafeMutablePointer<AVFrame>? = nil
  var encoderContext: UnsafeMutablePointer<AVCodecContext>? = nil
  var swsContext: OpaquePointer? = nil
  var nextPts: Int64 = 0
  public init(
    env: EnvType, filePath: String, width: Int = 1280, height: Int = 720, framesPerSecond: Int = 30
  ) {
    self.env = env
    self.filePath = filePath
    self.framesPerSecond = framesPerSecond
    simulate = Simulate(width: width, height: height, title: "video")
    simulate.use(model: self.env.model, data: &self.env.data)
    simulate.ui0 = false
    simulate.ui1 = false
    pthread_mutex_init(&videoLock, nil)
  }

  deinit {
    close()
  }
}

extension MuJoCoVideo: Renderable {
  private func makeVideoOutput() {
    guard formatContext == nil else { return }
    avformat_alloc_output_context2(&formatContext, nil, nil, filePath)
    if formatContext == nil {  // Cannot infer format from file name, use MPEG.
      avformat_alloc_output_context2(&formatContext, nil, "mpeg", filePath)
    }
    guard let formatContext = formatContext else { fatalError() }
    guard let fmt = formatContext.pointee.oformat else { fatalError() }
    precondition(fmt.pointee.video_codec != AV_CODEC_ID_NONE)
    // add_stream
    videoCodec = avcodec_find_encoder(fmt.pointee.video_codec)
    guard let videoCodec = videoCodec else { fatalError() }
    videoPacket = av_packet_alloc()
    videoStream = avformat_new_stream(formatContext, nil)
    guard let videoStream = videoStream else { fatalError() }
    videoStream.pointee.id = 0
    videoStream.pointee.time_base.num = 1
    videoStream.pointee.time_base.den = Int32(framesPerSecond)
    encoderContext = avcodec_alloc_context3(videoCodec)
    guard let encoderContext = encoderContext else { fatalError() }
    encoderContext.pointee.codec_id = fmt.pointee.video_codec
    encoderContext.pointee.bit_rate = 4_000_000
    encoderContext.pointee.width = Int32(simulate.width)
    encoderContext.pointee.height = Int32(simulate.height)
    encoderContext.pointee.time_base = videoStream.pointee.time_base
    encoderContext.pointee.gop_size = 12
    encoderContext.pointee.pix_fmt = AV_PIX_FMT_YUV420P
    if encoderContext.pointee.codec_id == AV_CODEC_ID_MPEG2VIDEO {
      encoderContext.pointee.max_b_frames = 2
    } else if encoderContext.pointee.codec_id == AV_CODEC_ID_MPEG1VIDEO {
      encoderContext.pointee.mb_decision = 2
    }
    if fmt.pointee.flags & AVFMT_GLOBALHEADER != 0 {
      encoderContext.pointee.flags |= AV_CODEC_FLAG_GLOBAL_HEADER
    }
    // open_video
    avcodec_open2(encoderContext, videoCodec, nil)
    avcodec_parameters_from_context(videoStream.pointee.codecpar, encoderContext)
    // alloc_picture
    videoFrame = av_frame_alloc()
    guard let videoFrame = videoFrame else { fatalError() }
    videoFrame.pointee.format = AV_PIX_FMT_RGB24.rawValue
    videoFrame.pointee.width = Int32(simulate.width)
    videoFrame.pointee.height = Int32(simulate.height)
    av_frame_get_buffer(videoFrame, 0)
    // alloc_picture
    videoYUVFrame = av_frame_alloc()
    guard let videoYUVFrame = videoYUVFrame else { fatalError() }
    videoYUVFrame.pointee.format = AV_PIX_FMT_YUV420P.rawValue
    videoYUVFrame.pointee.width = Int32(simulate.width)
    videoYUVFrame.pointee.height = Int32(simulate.height)
    av_frame_get_buffer(videoYUVFrame, 0)
    // get_video_frame
    if swsContext == nil {
      swsContext = sws_getContext(
        encoderContext.pointee.width, encoderContext.pointee.height, AV_PIX_FMT_RGB24,
        encoderContext.pointee.width, encoderContext.pointee.height, AV_PIX_FMT_YUV420P,
        SWS_BICUBIC, nil, nil, nil)
    }
    if fmt.pointee.flags & AVFMT_NOFILE == 0 {
      avio_open(&formatContext.pointee.pb, filePath, AVIO_FLAG_WRITE)
    }
    let _ = avformat_write_header(formatContext, nil)
    simulate.renderContextCallback = { [weak self] context, _, _ in
      guard let self = self else { return }
      self.sendFrame(context: context)
    }
  }

  public func render() {
    if cpugenesis == 0 {
      cpugenesis = GLContext.time
      simgenesis = env.data.time
      makeVideoOutput()
    }
    let simsync = env.data.time
    simulate.yield()
    var cpusync = GLContext.time
    while simsync - simgenesis >= cpusync - cpugenesis {  // wait until reality catches up with simulation.
      simulate.yield()
      cpusync = GLContext.time
    }
  }

  private func sendFrame(context: MjrContext) {
    pthread_mutex_lock(&videoLock)
    guard let encoderContext = encoderContext, let videoFrame = videoFrame,
      let videoYUVFrame = videoYUVFrame, let swsContext = swsContext, let videoPacket = videoPacket,
      let videoStream = videoStream
    else {
      pthread_mutex_unlock(&videoLock)
      return
    }
    // Now caught up, check whether we want to push a new frame.
    let ms = Int64((GLContext.time - cpugenesis) * 1000)
    guard
      av_compare_ts(nextPts, encoderContext.pointee.time_base, ms, AVRational(num: 1, den: 1000))
        < 0
    else {
      pthread_mutex_unlock(&videoLock)
      return
    }
    context.readPixels(
      rgb: &videoFrame.pointee.data.0!,
      viewport: MjrRect(
        left: 0, bottom: 0, width: videoFrame.pointee.width, height: videoFrame.pointee.height))
    var image = ccv_dense_matrix(
      encoderContext.pointee.height, encoderContext.pointee.width, Int32(CCV_8U | CCV_C3),
      videoFrame.pointee.data.0, 0)
    withUnsafeMutablePointer(to: &image) {
      var imagePtr: UnsafeMutablePointer<ccv_dense_matrix_t>? = $0
      ccv_flip(imagePtr, &imagePtr, 0, Int32(CCV_FLIP_Y))
    }
    guard av_frame_make_writable(videoYUVFrame) >= 0 else { return }
    withUnsafePointer(to: videoFrame.pointee.data) { videoFrameData in
      let videoFrameData = UnsafeRawPointer(videoFrameData).assumingMemoryBound(
        to: UnsafePointer<UInt8>?.self)
      withUnsafePointer(to: videoFrame.pointee.linesize) { videoFrameLinesize in
        let videoFrameLinesize = UnsafeRawPointer(videoFrameLinesize).assumingMemoryBound(
          to: Int32.self)
        withUnsafePointer(to: videoYUVFrame.pointee.data) { videoYUVFrameData in
          let videoYUVFrameData = UnsafeRawPointer(videoYUVFrameData).assumingMemoryBound(
            to: UnsafeMutablePointer<UInt8>?.self)
          withUnsafePointer(to: videoYUVFrame.pointee.linesize) { videoYUVFrameLinesize in
            let videoYUVFrameLinesize = UnsafeRawPointer(videoYUVFrameLinesize).assumingMemoryBound(
              to: Int32.self)
            let _ = sws_scale(
              swsContext, videoFrameData, videoFrameLinesize, 0, encoderContext.pointee.height,
              videoYUVFrameData, videoYUVFrameLinesize)
          }
        }
      }
    }
    nextPts += 1
    videoYUVFrame.pointee.pts = nextPts
    var ret = avcodec_send_frame(encoderContext, videoYUVFrame)
    precondition(ret >= 0)
    while ret >= 0 {
      ret = avcodec_receive_packet(encoderContext, videoPacket)
      if averror_is_eagain_or_eof(ret) != 0 {
        break
      } else if ret < 0 {
        fatalError("Error encoding a frame: \(ret)")
      }
      /* rescale output packet timestamp values from codec to stream timebase */
      av_packet_rescale_ts(
        videoPacket, encoderContext.pointee.time_base, videoStream.pointee.time_base)
      videoPacket.pointee.stream_index = videoStream.pointee.index

      /* Write the compressed frame to the media file. */
      ret = av_interleaved_write_frame(formatContext, videoPacket)
      /* pkt is now blank (av_interleaved_write_frame() takes ownership of
       * its contents and resets pkt), so that no unreferencing is necessary.
       * This would be different if one used av_write_frame(). */
      if ret < 0 {
        fatalError("Error while writing output packet: \(ret)")
      }
    }
    pthread_mutex_unlock(&videoLock)
  }
}

extension MuJoCoVideo {
  public func close() {
    pthread_mutex_lock(&videoLock)
    simulate.renderContextCallback = nil
    // flush all frames.
    if let encoderContext = encoderContext, let videoStream = videoStream,
      let videoPacket = videoPacket
    {
      var ret = avcodec_send_frame(encoderContext, nil)
      precondition(ret >= 0)
      while ret >= 0 {
        ret = avcodec_receive_packet(encoderContext, videoPacket)
        if averror_is_eagain_or_eof(ret) != 0 {
          break
        } else if ret < 0 {
          fatalError("Error encoding a frame: \(ret)")
        }
        /* rescale output packet timestamp values from codec to stream timebase */
        av_packet_rescale_ts(
          videoPacket, encoderContext.pointee.time_base, videoStream.pointee.time_base)
        videoPacket.pointee.stream_index = videoStream.pointee.index

        /* Write the compressed frame to the media file. */
        ret = av_interleaved_write_frame(formatContext, videoPacket)
        /* pkt is now blank (av_interleaved_write_frame() takes ownership of
         * its contents and resets pkt), so that no unreferencing is necessary.
         * This would be different if one used av_write_frame(). */
        if ret < 0 {
          fatalError("Error while writing output packet: \(ret)")
        }
      }
    }
    av_write_trailer(formatContext)
    if encoderContext != nil {
      avcodec_free_context(&encoderContext)
      encoderContext = nil
    }
    if videoFrame != nil {
      av_frame_free(&videoFrame)
      videoFrame = nil
    }
    if videoYUVFrame != nil {
      av_frame_free(&videoYUVFrame)
      videoYUVFrame = nil
    }
    if videoPacket != nil {
      av_packet_free(&videoPacket)
      videoPacket = nil
    }
    if swsContext != nil {
      sws_freeContext(swsContext)
      swsContext = nil
    }
    if let formatContext = formatContext, let fmt = formatContext.pointee.oformat,
      fmt.pointee.flags & AVFMT_NOFILE == 0
    {
      avio_closep(&formatContext.pointee.pb)
    }
    avformat_free_context(formatContext)
    formatContext = nil
    pthread_mutex_unlock(&videoLock)
  }
}
