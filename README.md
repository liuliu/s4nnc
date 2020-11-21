# Swift for NNC

s4nnc is a Swift interface for [libnnc](https://libnnc.org) library.

From the very start, [libnnc](https://libnnc.org) is meant to be a common runtime that supports many language bindings. It becomes apparently during the development that for deep learning, a raw C interface would be unwieldy complex to use in real-life. For example, the training loop of a transformer model for sentiment analysis takes more than 400 lines of code: <https://github.com/liuliu/ccv/blob/unstable/bin/nnc/imdb.c#L268>

A high-level language that delegates most of the work to the C interface seems to be a win in brevity and usefulness. The same training loop of a transformer model in Swift takes less than 100 lines: <https://github.com/liuliu/s4nnc/blob/main/examples/imdb/main.swift#L197>

Because the heavy-lifting are done in the [libnnc](https://libnnc.org) library, the Swift portion can be light and even automatically generated. At the moment, we have about 3,000 lines of Swift code to run quite a few models on GPU, complete with data feeder, model specification and optimizer.

## Design


