2021-08-06
----------
Swift is memory-safe language in general. But for frameworks talking to C, even C side has no memory issues (really?!), it is a good idea to run valgrind against the Swift binary.

When upgraded to Swift 5.4.2, the unit tests with this repo now fails in `DataFrameTests.testToGPU()`. The error is because a pointer cannot be unregistered from CUDA side. When we moving memory from CPU to GPU, we first pin the memory such that there will be marginal speed up when doing the copy from CPU to GPU memory. It can be fixed by just ignoring the error, because at unregistering time, I will deallocate that memory anyway. However, I do want to solve it, it bothers me. Digging deeper, the pointer changed from when allocated to when freed. That's why we pinned one memory region, but tries to unpin another one. This hints a memory issue.

Probing through the codebase doesn't yield much. The strong suspicion would always be on the C side of things. But we don't have this issue for a long time, the pointer looks like simply changed without any interaction on that side. After probing for an hour, I started to try valgrind. It crashes on tests almost immediately. First I dismissed it as if something wrong with valgrind. But looking back, it suggests some data I freed shouldn't. This is interesting.

In DataFrame, I wrap data from an array in two ways. One is tensor, which I wrapped as the C tensor such that it can be passed to a C processor (such as `to_gpu`) for transformations. Another is object, which we created temporarily and will release when iteration is done. Due to implementation error, I called `Unmanaged<AnyObject>.fromOpaque().release()` in the case where I wrap the C tensor, causing modifications on the said pointer from Swift runtime. This error only done when we create DataFrame from an array, which is not the majority of use-case.

I guess it is also time to see whether `rules_swift` can support address sanitizer, which Swift itself supports. These tools, even for a memory-safe language, is quite useful, it turns out.


2020-12-22
----------
Fixed previous mentioned issue. This is possible by having a new method in libnnc: `ccv_nnc_dynamic_graph_has_effect_to_tensor_variables` which will answer questions such as whether a variable A can have effect to variable B, therefore, finding the ones with requiresGrad = true while has effect to the object that you call `backward(to:)` on.

Added a unit test that simulate implementing `Dense` model by using tensor variable directly, and validated the test works.


2020-12-20
----------
One part is not exposed is how requiresGrad works in coordination with `backward(to:)`. The underlying `ccv_nnc_dynamic_graph_backward` method takes input tensors and gives out gradients for the inputs. Thus, if we want to compute grad for parameters that is not in `backward(to:)`, we need to somehow get that tensors from DynamicGraph.

More over, because we also need to know if that tracked tensor is relevant at all, we need to expose a new method to query such information from `ccv_nnc_dynamic_graph_t`. This requires us to design a new API.

This is not a big problem for my current use, but will be if I want to release this.
