OPT?=3
VALGRIND?=0
SANITIZE?=0
OPENMP?=0
PARALLEL=0

CFLAGS := -Wall -Wno-address-of-packed-member -Wextra -O$(OPT) -g  -std=c++17 -I../

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
 
basic: BinaryAdjacencyMatrix.cpp 
	$(CXX) $(CFLAGS) $(DEFINES) $(LDFLAGS) -o $@ BinaryAdjacencyMatrix.cpp



clean:
	rm -f *.o opt profile basic code.profdata default.profraw

