# MVE-1: Monologue Voice Engine Makefile
# C23 implementation with no external dependencies beyond libc, libm, pthreads

CC = gcc
CFLAGS_BASE = -std=c2x -Wall -Wextra -Wpedantic -fstrict-aliasing -Wconversion -Wshadow -Wformat=2 -Wnull-dereference -Wstack-usage=8192
CFLAGS_DEBUG = $(CFLAGS_BASE) -g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all
CFLAGS_RELEASE = $(CFLAGS_BASE) -O3 -DNDEBUG -march=native -flto -ffast-math

LDFLAGS = -lm -lpthread
LDFLAGS_DEBUG = -lm -lpthread -lasan -lubsan

SRCDIR = src
INCDIR = include
TESTDIR = tests
OBJDIR = build
BINDIR = bin
TESTOBJDIR = build/tests

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SOURCES))
TARGET = $(BINDIR)/mve

TEST_SOURCES = $(wildcard $(TESTDIR)/*.c)
TEST_OBJECTS = $(patsubst $(TESTDIR)/%.c,$(TESTOBJDIR)/%.o,$(TEST_SOURCES))
TEST_TARGETS = $(patsubst $(TESTDIR)/%.c,$(BINDIR)/%,$(TEST_SOURCES))

.PHONY: all debug release clean test dirs tests check valgrind sanitize coverage

all: release

dirs:
	@mkdir -p $(OBJDIR) $(BINDIR) $(TESTOBJDIR)

debug: CFLAGS = $(CFLAGS_DEBUG)
debug: LDFLAGS_FINAL = $(LDFLAGS_DEBUG)
debug: dirs $(TARGET)
	@echo "Debug build complete (with ASan/UBSan): $(TARGET)"

release: CFLAGS = $(CFLAGS_RELEASE)
release: LDFLAGS_FINAL = $(LDFLAGS)
release: dirs $(TARGET)
	@echo "Release build complete: $(TARGET)"

$(TARGET): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS_FINAL)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# Test targets - each test is standalone with only necessary libs
$(TESTOBJDIR)/%.o: $(TESTDIR)/%.c
	@mkdir -p $(TESTOBJDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# Individual test rules (explicit dependencies for each test)
$(BINDIR)/test_dsp: $(TESTOBJDIR)/test_dsp.o build/dsp.o build/tensor.o
	@mkdir -p $(BINDIR)
	$(CC) $^ -o $@ $(LDFLAGS_FINAL) -lm

$(BINDIR)/test_fft_simple: $(TESTOBJDIR)/test_fft_simple.o build/dsp.o
	@mkdir -p $(BINDIR)
	$(CC) $^ -o $@ $(LDFLAGS_FINAL) -lm

$(BINDIR)/test_fft_n8: $(TESTOBJDIR)/test_fft_n8.o build/dsp.o
	@mkdir -p $(BINDIR)
	$(CC) $^ -o $@ $(LDFLAGS_FINAL) -lm

$(BINDIR)/test_fft_roundtrip2: $(TESTOBJDIR)/test_fft_roundtrip2.o build/dsp.o
	@mkdir -p $(BINDIR)
	$(CC) $^ -o $@ $(LDFLAGS_FINAL) -lm

# Algorithm tests
$(BINDIR)/test_algos: $(TESTOBJDIR)/test_algos.o build/dsp.o build/tensor.o
	@mkdir -p $(BINDIR)
	$(CC) $^ -o $@ $(LDFLAGS_FINAL) -lm

TEST_TARGETS = $(BINDIR)/test_dsp $(BINDIR)/test_fft_simple $(BINDIR)/test_fft_n8 $(BINDIR)/test_fft_roundtrip2 $(BINDIR)/test_algos

tests: $(TEST_TARGETS)
	@echo "All test binaries built successfully"

check: tests
	@echo "Running all unit tests..."
	@failed=0; \
	for test in $(TEST_TARGETS); do \
		echo "Running $$test..."; \
		if ./$$test; then \
			echo "  ✓ PASSED"; \
		else \
			echo "  ✗ FAILED"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	if [ $$failed -eq 0 ]; then \
		echo "All tests passed!"; \
	else \
		echo "$$failed test(s) failed"; \
		exit 1; \
	fi

valgrind: debug $(TEST_TARGETS)
	@echo "Running tests with Valgrind..."
	@which valgrind > /dev/null || (echo "Valgrind not installed"; exit 1)
	@for test in $(TEST_TARGETS); do \
		echo "Valgrinding $$test..."; \
		valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=1 ./$$test || exit 1; \
	done
	@echo "Valgrind checks passed!"

sanitize: debug $(TEST_TARGETS)
	@echo "Running tests with sanitizers (already enabled in debug build)..."
	@export LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/12/libasan.so; \
	for test in $(TEST_TARGETS); do \
		echo "Running $$test with ASan/UBSan..."; \
		ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 ./$$test || exit 1; \
	done
	@echo "Sanitizer checks passed!"

coverage: CFLAGS += -fprofile-arcs -ftest-coverage
coverage: LDFLAGS_FINAL = $(LDFLAGS) -lgcov
coverage: clean tests
	@echo "Running tests for coverage..."
	@for test in $(TEST_TARGETS); do \
		./$$test || true; \
	done
	@echo "Coverage data generated. Run 'gcov' to analyze."

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(TESTOBJDIR) *.gcda *.gcno *.gcov

install: release
	cp $(TARGET) /usr/local/bin/mve
	chmod +x /usr/local/bin/mve

uninstall:
	rm -f /usr/local/bin/mve

docs:
	@echo "Generating API documentation..."
	@which doxygen > /dev/null && doxygen Doxyfile || echo "Doxygen not available, skipping documentation generation"

analyze:
	@echo "Running static analysis..."
	@which clang-tidy > /dev/null && clang-tidy $(SOURCES) -- -I$(INCDIR) || echo "clang-tidy not available"

format:
	@echo "Formatting code..."
	@which clang-format > /dev/null && clang-format -i $(SOURCES) $(INCDIR)/*.h $(TESTDIR)/*.c || echo "clang-format not available"

bench: release
	@echo "Build ready for benchmarking"
	@$(TARGET) bench models/default.mve 2>/dev/null || echo "No model available for benchmarking"

ci: clean sanitize check
	@echo "CI pipeline completed successfully"

help:
	@echo "MVE-1 Build System Targets:"
	@echo "  make          - Build release version (default)"
	@echo "  make debug    - Build with debug symbols and sanitizers"
	@echo "  make release  - Build optimized release version"
	@echo "  make tests    - Build all test binaries"
	@echo "  make check    - Run all unit tests"
	@echo "  make sanitize - Run tests with AddressSanitizer/UBSan"
	@echo "  make valgrind - Run tests with Valgrind (memory checking)"
	@echo "  make coverage - Generate code coverage data"
	@echo "  make clean    - Remove all build artifacts"
	@echo "  make install  - Install to /usr/local/bin"
	@echo "  make uninstall- Remove from /usr/local/bin"
	@echo "  make analyze  - Run static analysis (clang-tidy)"
	@echo "  make format   - Format code (clang-format)"
	@echo "  make docs     - Generate API documentation (doxygen)"
	@echo "  make bench    - Run benchmarks"
	@echo "  make ci       - Run full CI pipeline (clean, sanitize, check)"
	@echo "  make help     - Show this help message"
