2020-12-22
----------
Fixed previous mentioned issue. This is possible by having a new method in libnnc: `ccv_nnc_dynamic_graph_has_effect_to_tensor_variables` which will answer questions such as whether a variable A can have effect to variable B, therefore, finding the ones with requiresGrad = true while has effect to the object that you call `backward(to:)` on.

Added a unit test that simulate implementing `Dense` model by using tensor variable directly, and validated the test works.

2020-12-20
----------
One part is not exposed is how requiresGrad works in coordination with `backward(to:)`. The underlying `ccv_nnc_dynamic_graph_backward` method takes input tensors and gives out gradients for the inputs. Thus, if we want to compute grad for parameters that is not in `backward(to:)`, we need to somehow get that tensors from DynamicGraph.

More over, because we also need to know if that tracked tensor is relevant at all, we need to expose a new method to query such information from `ccv_nnc_dynamic_graph_t`. This requires us to design a new API.

This is not a big problem for my current use, but will be if I want to release this.
