# EdgeMapVertexMap

This repository seeks to make it easy to design your own graph data structure and have it run many graph algorithms.

It does this by creating a simple interface that can be implemented and then algorithms can use this interface with the EdgeMap VertexMap API introduced in [ligra](https://dl.acm.org/doi/abs/10.1145/2442516.2442530) to run the algorithms.

Also, a few basic data structures have already been defined.

## Running Basic Static Algorithms
To be able to run static algorithms your structure needs to define 
```
  template <class F>
  void map_neighbors(node_t node, F f, void *d, bool parallel);
```
Which loops over the neighbors of vertex node and applies function f to each element.  `parallel` indicated if you are allowed to map over the elements in parallel. 

After this simply construct your graph however you want and can include and run the algorithms from the `algorithms` directory.  

## Advanced Usage
There are a few complications and features that can be used for better performance.

### Triangle Count
The triangle count algorithm also requires 

```uint64_t common_neighbors(node_t a, node_t b);```

be defined which counts the number of common neighbors there are of a and b which have id less than either a or b.

### map_range
For some data structures finding the data for a specific vertex can be expensive from scratch and it can be faster to operate on ranges of vertices.  These structure can instead implement 
``` 

  template <class F>
  void map_range(node_t node_start, node_t node_end, F f, void *d);
```
Which should be equivalent to, but more efficient than 
```
    for (node_t i = node_start; i < node_end; i++) {
      G.map_neighbors(i, f, d, false);
    }
```


### Extra Data
Some structures benefit from being able to precalculate, or structure the data.  These structures can define the function 
```
template <typename T> getExtraData(bool skip);
```
This function, when it exists will be called before algorithms are run and their output will be passed in to the `d` argument of the map functions.


# Examples
- AdjacencyMatrix: BitMap of dense matrix
- AdjacencySet: Vector of std::set
- AdjacencyVector: Vector of sorted std::vector
- CSR: Compressed Sparse Rows
- AdjacencyBSet: Vector of absl::btree_set
- AdjacencyCompressedVector: Vector of vectors, individually compressed with delta encoding
- AdjacencyFlatHashSet: Vector of absl::flat_hash_map
- AdjacencyHashMap: Vector of std::unordered_set
- AdjacencyDenseSparseSet: Vector of DenseSparseSets, which are sorted vectors when small and bit vectors when large


# Building and Modifying Graphs
For the most part the task of building and updating the graphs are left to the user.

`io_util.hpp` has some helper functions to read in graph files in .adj format

`GraphHelpers.hpp` Can build a few basic kinds of graphs, and defined `parallel_batch_insert` which will insert a batch of edges using parallelism and assumes that different nodes can be inserted into concurrently.

