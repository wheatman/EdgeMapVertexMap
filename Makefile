buildAdjacencyBSet:
	$(MAKE) -C AdjacencyBSet basic
buildAdjacencyCompressedVector:
	$(MAKE) -C AdjacencyCompressedVector basic
buildAdjacencyDenseSparseSet:
	$(MAKE) -C AdjacencyDenseSparseSet basic
buildAdjacencyFlatHashSet:
	$(MAKE) -C AdjacencyFlatHashSet basic
buildAdjacencyHashSet:
	$(MAKE) -C AdjacencyHashSet basic
buildAdjacencyMatrix:
	$(MAKE) -C AdjacencyMatrix basic
buildAdjacencySet:
	$(MAKE) -C AdjacencySet basic
buildAdjacencyVector:
	$(MAKE) -C AdjacencyVector basic
buildCSR:
	$(MAKE) -C CSR basic
build: buildAdjacencyBSet buildAdjacencyCompressedVector buildAdjacencyDenseSparseSet buildAdjacencyFlatHashSet buildAdjacencyHashSet buildAdjacencyMatrix buildAdjacencySet buildAdjacencyVector buildCSR


testAdjacencyBSet: 
	$(MAKE) -C AdjacencyBSet test
testAdjacencyCompressedVector: 
	$(MAKE) -C AdjacencyCompressedVector test
testAdjacencyDenseSparseSet: 
	$(MAKE) -C AdjacencyDenseSparseSet test
testAdjacencyFlatHashSet: 
	$(MAKE) -C AdjacencyFlatHashSet test
testAdjacencyHashSet: 
	$(MAKE) -C AdjacencyHashSet test
testAdjacencyMatrix: 
	$(MAKE) -C AdjacencyMatrix test
testAdjacencySet: 
	$(MAKE) -C AdjacencySet test
testAdjacencyVector: 
	$(MAKE) -C AdjacencyVector test
testCSR:
	$(MAKE) -C CSR test
test: testAdjacencyBSet testAdjacencyCompressedVector testAdjacencyDenseSparseSet testAdjacencyFlatHashSet testAdjacencyHashSet testAdjacencyMatrix testAdjacencySet testAdjacencyVector testCSR