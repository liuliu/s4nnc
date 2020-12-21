2020-12-20
----------
One part is not exposed is how requiresGrad works in coordination with `backward(to:)`. The underlying `ccv_nnc_dynamic_graph_backward` method takes input tensors and gives out gradients for the inputs. Thus, if we want to compute grad for parameters that is not in `backward(to:)`, we need to somehow get that tensors from DynamicGraph.

More over, because we also need to know if that tracked tensor is relevant at all, we need to expose a new method to query such information from `ccv_nnc_dynamic_graph_t`. This requires us to design a new API.

This is not a big problem for my current use, but will be if I want to release this.
