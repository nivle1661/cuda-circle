# CUDA Renderer

This is my solution of a [CMU project](https://github.com/cmu15418/assignment2/tree/master/render) that renders different variety of circle patterns in CUDA. 
The main implementation resides in cudaRenderer.cu, which handles the `render` function.

## Optimization Approach

My approach relies on binning so that the number of circles to consider can be drastically lowered in each area. 
The pipeline is as follows:
1.  In parallel, find the bins that each circle belongs to.
2.  Using `thrust::stable_sort_by_key`, find the sorted list of circles that each bin contains (preserves render ordering)
3.  Render each bin on its own thread by looking through each pixel and sequentially checking if it lies within any of the bin's circles


