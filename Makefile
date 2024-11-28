# Htrain Makefile. Build and test Htrain.

## This is the makefile if you are running cabal-install 1.24 or later.

## Locations of binaries used when running tests, or generating the images to go along with our README.md.
# The location of GHC, used to compile .hs examples.
GHC=ghc
GHCVERSION=$(shell ${GHC} --version | sed "s/.*version //")
ARCHITECTURE=$(shell uname -m | sed "s/i[3-6]86/i386/" )
# new-style location root. must NOT have trailing slash
BUILDROOT=dist-newstyle/build/${ARCHITECTURE}-linux/ghc-${GHCVERSION}/htrain-0.0.1
EXEBUILDROOT=${BUILDROOT}/x/
TESTBUILDROOT=${BUILDROOT}/t/
BENCHBUILDROOT=${BUILDROOT}/b/

exebin = ${EXEBUILDROOT}/$(1)/build/$(1)/$(1)
exedir = ${EXEBUILDROOT}/$(1)

# The location of the created HTrain binary, for running shell based test cases.
HTRAIN=trainLLM
HTRAINDIR=$(call exedir,${HTRAIN})
HTRAINBIN=$(call exebin,${HTRAIN})
# The location of the created test binary, for running haskell test cases.
#TESTSUITE=${TESTBUILDROOT}/test-hslice/build/test-hslice/test-hslice
#BROKENTESTSUITE=${TESTBUILDROOT}/test-broken/build/test-broken/test-broken
#STATTESTSUITE=${TESTBUILDROOT}/test-stat/build/test-stat/test-stat


## Options used when calling ImplicitCAD. for testing, and for image generation.
# Enable multiple CPU usage.
# Use the parallel garbage collector.
# spit out some performance statistics.
RTSOPTS=+RTS -N -qg -t

# Uncomment for profiling support. Note that you will need to recompile all of the libraries, as well.
#PROFILING= --enable-profiling

## FIXME: escape this right
# Uncomment for valgrind on the examples.
#VALGRIND=valgrind --tool=cachegrind --cachegrind-out-file=$$each.cachegrind.`date +%s`

LIBDIR=Text
LIBFILES=$(shell find ${LIBDIR} -name '*.hs')
LIBBUILD=$(shell find ${LIBDIR} -name '*.hi')
LIBTARGET=${BUILDROOT}/build/${LIBDIR}/Htrain.o

FILETARGETS=the-verdict.txt gpt2-vocab.json

EXECBUILDDIRS=$(HTRAINDIR)
EXECTARGETS=$(HTRAINBIN)  # $(TESTSUITE) $(BROKENTESTSUITE) $(STATTESTSUITE)
TARGETS=$(EXECTARGETS) $(LIBTARGET)

# Mark the below fake targets as unreal, so make will not get choked up if a file with one of these names is created.
.PHONY: build install clean distclean nukeclean docs dist examples tests

# Empty out the default suffix list, to make debugging output cleaner.
.SUFFIXES:

# Allow for us to (ab)use $$* in dependencies of rules.
.SECONDEXPANSION:

# Disable make's default builtin rules, to make debugging output cleaner.
MAKEFLAGS += --no-builtin-rules

# Build binaries.
build: $(TARGETS) $(FILETARGETS)

# Install.
install: build
	cabal install -j

# Cleanup from using the rules in this file.
clean:
	rm -f Setup
	rm -f $(TARGETS)
	rm -f $(LIBBUILD)
	rm -rf ${EXECBUILDDIRS}
	rm -f ${BUILDROOT}/build/libHS*
	rm -f ${BUILDROOT}/cache/registration

# Clean up before making a release.
distclean: clean Setup
	./Setup clean 
	rm -f Setup Setup.hi Setup.o
	rm -rf dist-newstyle
	rm -f `find ./ -name "*~"`
	rm -f `find ./ -name "\#*\#"`
	rm -rf ${FILETARGETS}

# Destroy the current user's cabal/ghc environment.
nukeclean: distclean
	rm -rf ~/.cabal/ ~/.ghc/

# Generate documentation.
docs: $(TARGETS)
	cabal haddock --enable-documentation

# Upload to hackage?
dist: $(TARGETS)
	./Setup sdist

# Generate examples.
#examples: $(HTRAINBIN)
#	cd Examples && for each in `find ./ -name '*stl' -type f | sort`; do { echo $$each ; ../$(HTRAINBIN) $$each $(RTSOPTS); } done

# Generate images from the examples, so we can upload the images to our website.
#images: examples
#	cd Examples && for each in `find ./ -name '*.stl' -type f | sort`; do { filename=$(basename "$$each"); filename="$${filename%.*}"; if [ -e $$filename.transform ] ; then echo ${stl2ps} $$each $$filename.ps `cat $$filename.transform`; else ${stl2ps} $$each $$filename.ps; fi; ${convert} $$filename.ps $$filename.png; } done

# tests.
#tests: $(HTRAINBIN)
#	cd tests && for each in `find ./ -name '*.stl' -type f | sort`; do { echo $$each ; ../$(HTRAINBIN) slice -l $$each $(RTSOPTS); md5sum out.gcode; } done
#	$(TESTSUITE)

#format:
#	./layout/ormolu.sh -f

# The library.
$(LIBTARGET): $(LIBFILES)
	cabal v2-build ${PROFILING} htrain

# The test suite, since it's source is stored in a different location than the other binaries we build:
${TESTBUILDROOT}/test-htrain/build/test-htrain/test-htrain: Setup ${BUILDROOT}/setup-config $(LIBTARGET) $(LIBFILES)
	cabal v2-build test-htrain

# The test suite, since it's source is stored in a different location than the other binaries we build:
#${TESTBUILDROOT}/test-broken/build/test-broken/test-broken: Setup ${BUILDROOT}/setup-config $(LIBTARGET) $(LIBFILES)
#	cabal v2-build test-broken


# The test suite, since it's source is stored in a different location than the other binaries we build:
#${TESTBUILDROOT}/test-stat/build/test-stat/test-stat: Setup ${BUILDROOT}/setup-config $(LIBTARGET) $(LIBFILES)
#	cabal v2-build test-stat

# Build a binary target with cabal.
${EXEBUILDROOT}/%: programs/$$(word 1,$$(subst /, ,%)).hs Setup ${BUILDROOT}/setup-config $(LIBTARGET) $(LIBFILES)
	cabal v2-build -j ${PROFILING} $(word 1,$(subst /, ,$*))
	touch $@

# Build a benchmark target with cabal.
#${BENCHBUILDROOT}/%: programs/$$(word 1,$$(subst /, ,%)).hs Setup ${BUILDROOT}/setup-config $(LIBTARGET) $(LIBFILES)
#	cabal v2-build $(word 1,$(subst /, ,$*))

dist-newstyle/cache/config: htrain.cabal
	cabal v2-update
	touch $@

# Prepare to build.
dist-newstyle/cache/plan.json: htrain.cabal
	cabal v2-install -j --only-dependencies --upgrade-dependencies --overwrite-policy=always $(PROFILING)
	cabal v2-configure -j --enable-tests --enable-benchmarks $(PROFILING)

# The setup command, used to perform administrative tasks (haddock, upload to hackage, clean, etc...).
Setup: Setup.*hs dist-newstyle/cache/config dist-newstyle/cache/plan.json $(LIBTARGET)
	$(GHC) -O2 -Wall --make Setup -package Cabal
	touch $@

the-verdict.txt:
	curl https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt -o $@

gpt2-vocab.json:
	curl https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -o $@
