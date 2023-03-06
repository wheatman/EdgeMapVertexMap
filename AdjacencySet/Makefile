OPT?=3
VALGRIND?=0
SANITIZE?=0
OPENMP?=0
PARALLEL=0

CFLAGS := -Wall -Wno-address-of-packed-member -Wextra -O$(OPT) -g -gdwarf-4 -std=c++20 -I../include -I../ParallelTools -I../cxxopts/include/

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
 
basic: AdjacencySet.cpp 
	$(CXX) $(CFLAGS) $(DEFINES) $(LDFLAGS) -o $@ AdjacencySet.cpp

clean:
	rm -f *.o opt profile basic code.profdata default.profraw

test_bfs: basic
	./basic -g ../data/slashdot.adj --algorithm bfs --src 1
	diff bfs.out ../correct_output/slashdot/bfs/source1 
	./basic -g ../data/slashdot.adj --algorithm bfs --src 2
	diff bfs.out ../correct_output/slashdot/bfs/source2
	./basic -g ../data/slashdot.adj --algorithm bfs --src 3
	diff bfs.out ../correct_output/slashdot/bfs/source3
	./basic -g ../data/slashdot.adj --algorithm bfs --src 4
	diff bfs.out ../correct_output/slashdot/bfs/source4 
	./basic -g ../data/slashdot.adj --algorithm bfs --src 5
	diff bfs.out ../correct_output/slashdot/bfs/source5 
test_bc: basic
	./basic -g ../data/slashdot.adj --algorithm bc --src 1
	diff bc.out ../correct_output/slashdot/bc/source1 
	./basic -g ../data/slashdot.adj --algorithm bc --src 2
	diff bc.out ../correct_output/slashdot/bc/source2
	./basic -g ../data/slashdot.adj --algorithm bc --src 3
	diff bc.out ../correct_output/slashdot/bc/source3
	./basic -g ../data/slashdot.adj --algorithm bc --src 4
	diff bc.out ../correct_output/slashdot/bc/source4 
	./basic -g ../data/slashdot.adj --algorithm bc --src 5
	diff bc.out ../correct_output/slashdot/bc/source5 
test_pr: basic
	./basic -g ../data/slashdot.adj --algorithm pr --priters 10
	diff pr.out ../correct_output/slashdot/pr/iters10 
test_cc: basic
	./basic -g ../data/slashdot.adj --algorithm cc
	diff cc.out ../correct_output/slashdot/cc/output 
test_bf: basic
	./basic -w -g ../data/slashdot_weights.adj --algorithm bf --src 1
	diff bf.out ../correct_output/slashdot_weights/bf/source1 
	./basic -w -g ../data/slashdot_weights.adj --algorithm bf --src 2
	diff bf.out ../correct_output/slashdot_weights/bf/source2
	./basic -w -g ../data/slashdot_weights.adj --algorithm bf --src 3
	diff bf.out ../correct_output/slashdot_weights/bf/source3
	./basic -w -g ../data/slashdot_weights.adj --algorithm bf --src 4
	diff bf.out ../correct_output/slashdot_weights/bf/source4 
	./basic -w -g ../data/slashdot_weights.adj --algorithm bf --src 5
	diff bf.out ../correct_output/slashdot_weights/bf/source5 
test_bfs_directed: basic test_bfs
	./basic -g ../data/small_er.adj --algorithm bfs --symetric=false --src 1
	diff bfs.out ../correct_output/small_er/bfs/source1 
	./basic -g ../data/small_er.adj --algorithm bfs --symetric=false --src 2
	diff bfs.out ../correct_output/small_er/bfs/source2
	./basic -g ../data/small_er.adj --algorithm bfs --symetric=false --src 3
	diff bfs.out ../correct_output/small_er/bfs/source3
	./basic -g ../data/small_er.adj --algorithm bfs --symetric=false --src 4
	diff bfs.out ../correct_output/small_er/bfs/source4
	./basic -g ../data/small_er.adj --algorithm bfs --symetric=false --src 5
	diff bfs.out ../correct_output/small_er/bfs/source5
test: test_bfs test_bc test_cc test_pr test_bf test_bfs_directed
