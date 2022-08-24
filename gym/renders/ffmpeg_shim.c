#include "ffmpeg_shim.h"

int averror_is_eagain_or_eof(int ret)
{
	return ret == AVERROR(EAGAIN) || ret == AVERROR_EOF;
}
