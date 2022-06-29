OPT?=3
VALGRIND?=0
SANITIZE?=0
OPENMP?=0
PARALLEL=0

CFLAGS := -Wall -Wno-address-of-packed-member -Wextra -O$(OPT) -g  -std=c++20 -I../../ -I../../ParallelTools

LDFLAGS := -lrt -lm -lm -ldl 


ifeq ($(OPENMP),1)
PARALLEL=1
CILK=0
CFLAGS += -fopenmp
ONE_WORKER = OMP_NUM_THREADS=1
else
CILK?=0
endif

ifeq ($(CILK),1)
CFLAGS += -fopencilk
LDFLAGS += -Lx86_64/ -lopencilk
ONE_WORKER = CILK_NWORKERS=1
PARALLEL=1
endif



ifeq ($(SANITIZE),1)
ifeq ($(OPENMP),1)
CFLAGS += -fsanitize=undefined,thread -fno-omit-frame-pointer
else
ifeq ($(CILK),1)
CFLAGS += -fsanitize=cilk,undefined -fno-omit-frame-pointer
else
CFLAGS += -fsanitize=undefined,address -fno-omit-frame-pointer
endif
endif
endif

ifeq ($(OPT),3)
CFLAGS += -fno-signed-zeros  -freciprocal-math -ffp-contract=fast -fno-trapping-math  -ffinite-math-only
ifeq ($(VALGRIND),0)
CFLAGS += -march=native #-static
endif
endif





DEFINES := -DOPENMP=$(OPENMP) -DCILK=$(CILK)


ifeq ($(DEBUG),0)
CFLAGS += -DNDEBUG
endif



all: basic
 
basic: AdjacencyFlatHashSet.cpp
	mkdir -p build && cd build && CXXFLAGS="$(CFLAGS) $(DEFINES)" cmake .. && make -j && cp basic ..

clean:
	rm -f *.o opt profile basic code.profdata default.profraw
	rm -rf build

test_bfs: basic
	./basic ../data/slashdot.adj  bfs 1
	diff bfs.out ../correct_output/slashdot/bfs/source1 
	./basic ../data/slashdot.adj  bfs 2
	diff bfs.out ../correct_output/slashdot/bfs/source2
	./basic ../data/slashdot.adj  bfs 3
	diff bfs.out ../correct_output/slashdot/bfs/source3
	./basic ../data/slashdot.adj  bfs 4
	diff bfs.out ../correct_output/slashdot/bfs/source4 
	./basic ../data/slashdot.adj  bfs 5
	diff bfs.out ../correct_output/slashdot/bfs/source5 
test_bc: basic
	./basic ../data/slashdot.adj  bc 1
	diff bc.out ../correct_output/slashdot/bc/source1 
	./basic ../data/slashdot.adj  bc 2
	diff bc.out ../correct_output/slashdot/bc/source2
	./basic ../data/slashdot.adj  bc 3
	diff bc.out ../correct_output/slashdot/bc/source3
	./basic ../data/slashdot.adj  bc 4
	diff bc.out ../correct_output/slashdot/bc/source4 
	./basic ../data/slashdot.adj  bc 5
	diff bc.out ../correct_output/slashdot/bc/source5 
test_pr: basic
	./basic ../data/slashdot.adj  pr 10
	diff pr.out ../correct_output/slashdot/pr/iters10 
test_cc: basic
	./basic ../data/slashdot.adj  cc
	diff cc.out ../correct_output/slashdot/cc/output 
test: test_bfs test_bc test_cc test_pr

