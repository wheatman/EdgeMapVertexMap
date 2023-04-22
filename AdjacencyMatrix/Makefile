OPT?=3
VALGRIND?=0
SANITIZE?=0
CILK?=0
PARLAY?=0
DEBUG?=0

CFLAGS := -Wall -Wno-address-of-packed-member -Wextra -O$(OPT) -g -gdwarf-4 -std=c++20 -I../include -I../ParallelTools -I../cxxopts/include/ -I../parlaylib/include/

LDFLAGS := -lrt -lm -lm -ldl 


ifeq ($(CILK),1)
CFLAGS += -fopencilk
PARLAY=0
endif



ifeq ($(SANITIZE),1)
ifeq ($(CILK),1)
CFLAGS += -fsanitize=cilk,undefined -fno-omit-frame-pointer
else
CFLAGS += -fsanitize=undefined,address -fno-omit-frame-pointer
endif
endif

ifeq ($(OPT),3)
CFLAGS += -fno-signed-zeros  -freciprocal-math -ffp-contract=fast -fno-trapping-math  -ffinite-math-only
ifeq ($(VALGRIND),0)
CFLAGS += -march=native #-static
endif
endif





DEFINES := -DPARLAY=$(PARLAY) -DCILK=$(CILK)


ifeq ($(DEBUG),0)
CFLAGS += -DNDEBUG
endif



all: basic
 
basic: BinaryAdjacencyMatrix.cpp 
	$(CXX) $(CFLAGS) $(DEFINES) $(LDFLAGS) -o $@ BinaryAdjacencyMatrix.cpp



clean:
	rm -f *.o opt profile basic code.profdata default.profraw

test_bfs: basic
	@./basic -g ../data/slashdot.adj --algorithm bfs --src 1 --iters 1
	@diff -q bfs.out ../correct_output/slashdot/bfs/source1 
	@./basic -g ../data/slashdot.adj --algorithm bfs --src 2 --iters 1
	@diff -q bfs.out ../correct_output/slashdot/bfs/source2
	@./basic -g ../data/slashdot.adj --algorithm bfs --src 3 --iters 1
	@diff -q bfs.out ../correct_output/slashdot/bfs/source3
	@./basic -g ../data/slashdot.adj --algorithm bfs --src 4 --iters 1
	@diff -q bfs.out ../correct_output/slashdot/bfs/source4 
	@./basic -g ../data/slashdot.adj --algorithm bfs --src 5 --iters 1
	@diff -q bfs.out ../correct_output/slashdot/bfs/source5 
test_bc: basic
	@./basic -g ../data/slashdot.adj --algorithm bc --src 1 --iters 1
	@diff -q bc.out ../correct_output/slashdot/bc/source1 
	@./basic -g ../data/slashdot.adj --algorithm bc --src 2 --iters 1
	@diff -q bc.out ../correct_output/slashdot/bc/source2
	@./basic -g ../data/slashdot.adj --algorithm bc --src 3 --iters 1
	@diff -q bc.out ../correct_output/slashdot/bc/source3
	@./basic -g ../data/slashdot.adj --algorithm bc --src 4 --iters 1
	@diff -q bc.out ../correct_output/slashdot/bc/source4 
	@./basic -g ../data/slashdot.adj --algorithm bc --src 5 --iters 1
	@diff -q bc.out ../correct_output/slashdot/bc/source5 
test_pr: basic
	@./basic -g ../data/slashdot.adj --algorithm pr --pr_rounds 10 --iters 1
	@diff -q pr.out ../correct_output/slashdot/pr/iters10 
test_cc: basic
	@./basic -g ../data/slashdot.adj --algorithm cc --iters 1
	@diff -q cc.out ../correct_output/slashdot/cc/output 
test: test_bfs test_bc test_cc test_pr
