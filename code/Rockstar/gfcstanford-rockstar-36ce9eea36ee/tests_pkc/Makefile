# Tests for the PKC reader (io/pkc_core.c) against the reference snapcomp library of pkdgrav3.
# Not part of the Rockstar build. Needs a built snapcomp (cmake -S <pkdgrav3>/io/snapcomp -B <build>).
#   make SNAPCOMP_SRC=<pkdgrav3>/io/snapcomp SNAPCOMP_BUILD=<build> ZSTD_DIR=<prefix>
#   ./run_tests.sh <scratch dir>
SNAPCOMP_SRC ?= $(HOME)/Documents/Projects/pkdgrav3_dev/pkdgrav3/io/snapcomp
SNAPCOMP_BUILD ?= $(SNAPCOMP_SRC)/../../build_snapcomp
ZSTD_DIR ?= /opt/homebrew
CC ?= cc
CXX ?= c++
CFLAGS = -std=c99 -O2 -Wall -Wextra -DENABLE_PKC -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE -I$(ZSTD_DIR)/include
CXXFLAGS = -std=c++17 -O2 -Wall -DENABLE_PKC -I$(SNAPCOMP_SRC) -I../io -I$(ZSTD_DIR)/include
LIBS = $(SNAPCOMP_BUILD)/libsnapcomp.a -L$(ZSTD_DIR)/lib -lzstd -lpthread

all: make_testset pkc_compare

pkc_core.o: ../io/pkc_core.c ../io/pkc_core.h
	$(CC) $(CFLAGS) -c ../io/pkc_core.c -o $@

make_testset: make_testset.cxx
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

pkc_compare: pkc_compare.cxx pkc_core.o
	$(CXX) $(CXXFLAGS) $< pkc_core.o -o $@ $(LIBS)

clean:
	rm -f pkc_core.o make_testset pkc_compare
