# Windhover engine — modular build (CPU MoE + dense KPK).
# Primary binary: ./windhover-engine[.exe]
#
# Layout:
#   io/        safetensors + tokenizer + json headers
#   memory/    hard RAM budget
#   runtime/   forward / generate / CLI (ported GLM math)
#   attn/ moe/ model/ tensor/  — module boundaries (headers + future splits)
#
# Platforms: Darwin (clang+libomp), Linux (clang preferred), Windows (MinGW/Clang).
# Windows changes are additive — Darwin/Linux hot paths stay byte-identical.

DETECT_CC := $(if $(filter default,$(origin CC)),clang,$(CC))
UNAME_S   := $(shell uname -s 2>/dev/null)

# Host is Windows when building under cmd/MSYS or MinGW (uname may be MINGW*/MSYS*).
IS_WIN :=
ifeq ($(OS),Windows_NT)
IS_WIN := 1
endif
ifneq (,$(findstring MINGW,$(UNAME_S)))
IS_WIN := 1
endif
ifneq (,$(findstring MSYS,$(UNAME_S)))
IS_WIN := 1
endif
ifneq (,$(findstring CYGWIN,$(UNAME_S)))
IS_WIN := 1
endif

ROOT := $(abspath .)
INCLUDES := -I$(ROOT) -I$(ROOT)/io -I$(ROOT)/memory -I$(ROOT)/include -I$(ROOT)/runtime
EXE :=

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

else ifneq ($(IS_WIN),)
# --- Windows 11 (MinGW-w64 / Clang) x64 or aarch64 ---
# Prefer clang when present for IEEE _Float16 (same rationale as Linux).
ifeq ($(origin CC),default)
WHICH_CLANG := $(shell where clang 2>NUL || which clang 2>/dev/null)
ifneq ($(WHICH_CLANG),)
CC = clang
else
CC = gcc
endif
endif
# ARCH: x86-64-v3 (AVX2 MoE) on amd64; armv8-a on ARM64 Windows.
# MSYS2 often omits PROCESSOR_ARCHITECTURE — also probe uname / compiler triple.
ifndef ARCH
UNAME_M := $(shell uname -m 2>/dev/null)
CC_MACHINE := $(shell $(CC) -dumpmachine 2>/dev/null)
ifeq ($(PROCESSOR_ARCHITECTURE),ARM64)
ARCH := armv8-a
else ifeq ($(PROCESSOR_ARCHITEW6432),ARM64)
ARCH := armv8-a
else ifneq (,$(filter aarch64 arm64,$(UNAME_M)))
ARCH := armv8-a
else ifneq (,$(findstring aarch64,$(CC_MACHINE)))
ARCH := armv8-a
else ifneq (,$(findstring arm64,$(CC_MACHINE)))
ARCH := armv8-a
else
ARCH := x86-64-v3
endif
endif
CFLAGS  = -D_FILE_OFFSET_BITS=64 -O3 -fopenmp $(INCLUDES) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
# -march for x86 and armv8-* ISAs; -mcpu only for named CPU cores (e.g. native).
# Important: "armv8-a" contains "arm" — do not send it to -mcpu= (Clang on WinARM rejects it).
ifneq (,$(findstring armv8,$(ARCH)))
CFLAGS += -march=$(ARCH)
else ifneq (,$(findstring aarch64,$(ARCH)))
CFLAGS += -march=$(ARCH)
else ifneq (,$(filter native,$(ARCH)))
CFLAGS += -mcpu=$(ARCH)
else ifneq (,$(findstring cortex,$(ARCH)))
CFLAGS += -mcpu=$(ARCH)
else
CFLAGS += -march=$(ARCH)
endif
LDFLAGS = -lm -fopenmp -static -lpthread -lwinpthread -lpsapi
EXE     = .exe

else
# Prefer clang on Linux: GCC may lack convenient _Float16/__fp16 by-value support
# depending on target; clang 15+ handles Windhover's IEEE fp16 path cleanly.
CC      ?= clang
HAVE_OMP := $(shell echo '\#include <omp.h>' | $(CC) -fopenmp -E - >/dev/null 2>&1 && echo yes)
ifeq ($(HAVE_OMP),yes)
OMPC    = -fopenmp
OMPL    = -fopenmp
else
OMPC    =
OMPL    =
$(warning OpenMP headers not found — building without -fopenmp)
endif
CFLAGS  = -O3 $(OMPC) $(INCLUDES) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
ifneq ($(ARCH),)
CFLAGS += -march=$(ARCH)
endif
LDFLAGS = -lm -lpthread $(OMPL)
endif

.PHONY: all clean test-oracle

all: windhover-engine$(EXE)

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

windhover-engine$(EXE): runtime/engine.o runtime/dense.o runtime/windhover.o runtime/au.o memory/budget.o
	$(CC) $(CFLAGS) -o $@ runtime/engine.o runtime/dense.o runtime/windhover.o runtime/au.o memory/budget.o $(LDFLAGS)
ifeq ($(IS_WIN),)
	@# Compat symlink for older docs/scripts (Unix only)
	ln -sfn windhover-engine kestrel-engine
else
	@# Windows: copy for legacy name
	-copy /Y windhover-engine.exe kestrel-engine.exe >NUL 2>&1 || cp -f windhover-engine.exe kestrel-engine.exe 2>/dev/null || true
endif

# SNAP path relative to engine/ — fixtures live in fixtures/
# Use POSIX env assignment so MSYS2 / Git Bash `sh` works (cmd `set` breaks .\exe).
# Engine returns non-zero on TF mismatch; also assert the 32/32 line so a silent
# "always 0" regression cannot greenwash CI.
test-oracle: windhover-engine$(EXE)
	@out=$$(SNAP=./fixtures/glm_tiny TF=1 ./windhover-engine$(EXE) 64 16 16 2>&1); status=$$?; \
	echo "$$out"; \
	if [ $$status -ne 0 ]; then echo "oracle binary exited $$status" >&2; exit $$status; fi; \
	echo "$$out" | grep -q '32/32' || { echo "oracle failed: expected 32/32" >&2; exit 1; }

clean:
	rm -f windhover-engine windhover-engine.exe kestrel-engine kestrel-engine.exe \
	  runtime/engine.o runtime/dense.o runtime/windhover.o runtime/au.o memory/budget.o
