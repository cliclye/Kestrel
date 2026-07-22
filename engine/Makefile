# Windhover engine — modular build (CPU MoE + dense KPK).
# Primary binary: ./windhover-engine
#
# Layout:
#   io/        safetensors + tokenizer + json headers
#   memory/    hard RAM budget
#   runtime/   forward / generate / CLI (ported GLM math)
#   attn/ moe/ model/ tensor/  — module boundaries (headers + future splits)

DETECT_CC := $(if $(filter default,$(origin CC)),clang,$(CC))
UNAME_S   := $(shell uname -s 2>/dev/null)

ROOT := $(abspath .)
INCLUDES := -I$(ROOT) -I$(ROOT)/io -I$(ROOT)/memory -I$(ROOT)/include -I$(ROOT)/runtime

ifeq ($(UNAME_S),Darwin)
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(and $(OMPDIR),$(wildcard $(OMPDIR)/include/omp.h),$(wildcard $(OMPDIR)/lib/libomp.*)),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) $(INCLUDES) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
ifneq ($(ARCH),)
CFLAGS += -mcpu=$(ARCH)
endif
# Optional SME2 int8 prefill GEMM (G5 gate). Build: make ARCH=native SME=1
# Runtime still defaults to NEON; set WH_SME_RUNTIME=1 to enable (packs on the fly).
ifeq ($(SME),1)
CFLAGS += -DWH_SME -march=armv8.7-a+sme2
endif
LDFLAGS = -lm $(OMPL)
else
CC      ?= gcc
CFLAGS  = -O3 -fopenmp $(INCLUDES) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
ifneq ($(ARCH),)
CFLAGS += -march=$(ARCH)
endif
LDFLAGS = -lm -lpthread
endif

.PHONY: all clean test-oracle

all: windhover-engine

memory/budget.o: memory/budget.c memory/budget.h
	$(CC) $(CFLAGS) -c memory/budget.c -o memory/budget.o

runtime/engine.o: runtime/engine.c memory/budget.h runtime/dense.h runtime/windhover.h
	$(CC) $(CFLAGS) -c runtime/engine.c -o runtime/engine.o

runtime/dense.o: runtime/dense.c runtime/dense.h io/tok.h io/st.h
	$(CC) $(CFLAGS) -c runtime/dense.c -o runtime/dense.o

runtime/windhover.o: runtime/windhover.c runtime/windhover.h runtime/model_desc.h runtime/au.h io/tok.h io/st.h memory/budget.h
	$(CC) $(CFLAGS) -c runtime/windhover.c -o runtime/windhover.o

runtime/au.o: runtime/au.c runtime/au.h
	$(CC) $(CFLAGS) -c runtime/au.c -o runtime/au.o

windhover-engine: runtime/engine.o runtime/dense.o runtime/windhover.o runtime/au.o memory/budget.o
	$(CC) $(CFLAGS) -o $@ runtime/engine.o runtime/dense.o runtime/windhover.o runtime/au.o memory/budget.o $(LDFLAGS)
	@# Compat symlink for older docs/scripts
	ln -sfn windhover-engine kestrel-engine

# SNAP path relative to engine/ — fixtures live in fixtures/
test-oracle: windhover-engine
	SNAP=./fixtures/glm_tiny TF=1 ./windhover-engine 64 16 16

clean:
	rm -f windhover-engine kestrel-engine runtime/engine.o runtime/dense.o runtime/windhover.o runtime/au.o memory/budget.o
