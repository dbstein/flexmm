# flexmm: Flexible python/numba framework for implementing FMM methods

A reasonably performant Python/Numba framework for implementing Fast-Multipole Methods. Currently, only 2D is supported, and I have implementations for the "Kernel-Independent" FMM and the "Black-Box" FMM.  These live on top of the same codebase, and only minor modifications are required to specify S2M, M2M, M2L, L2L, and L2T operators. Thanks to some particular structural choices, M2L translation operators only need to be specified for boxes of the same size. I imagine that kernel-dependent FMMs could be built in the same way.  SVD acceleration of M2L operators is done automatically for all methods without any extra infrastructure required.

Both the KI and BB FMM methods only require a function to evalute the kernel.  Examples should make clear how this works. Unfortunately, the calling sequence is a little irritating, in order to get everything into the right namespace in a clean-ish way.  This certainly could be made better.

## Timings

Some timings for the Laplace KIFMM (with 60 equivalent source points, good to about 13 digits). All values are for a FMM with 1,000,000 source points and 1,000,000 target points, on my 4 core macbook pro, and given in thousands of points/sec/core. Comparisons are made to FMMLIB2D, with the PREC variable set to 4.

### Random source points on a circle, target points in a grid:

|                 | flexmm | fmmlib2d |
|-----------------|--------|----------|
| form multipoles | 558    | 469      |
| source eval     | 496    | 908      |
| target eval     | 2218   | 780      |

### Random clusters of source points, target points in a grid:

|                 | flexmm | fmmlib2d |
|-----------------|--------|----------|
| form multipoles | 858    | 348      |
| source eval     | 442    | 166      |
| target eval     | 2291   | 831      |

### Uniformly random source points, target points in a grid:

|                 | flexmm | fmmlib2d |
|-----------------|--------|----------|
| form multipoles | 947    | 422      |
| source eval     | 417    | 203      |
| target eval     | 414    | 156      |

Right now, some functions don't get inlined.  Numba has an update in the works that supports specifying which functions to always inline (at the level of the Numba IR). I suspect that once this is an option, these timings will improve somewhat (at the cost of increased compile times). This should be particularly noticabel if you are running with a kernel that is complicated enough that LLVM decides not to inline it through its automated heuristics.

## To do

1. An interface to allow other kinds of sources (e.g. dipoles)
2. An interface to allow the user to compute other results (e.g. gradients)
3. Reuse of precomputations
4. Vector valued functions
5. 3D
6. Lots of housecleaning

A 3D implementation should actually be quite easy, because of the simplified FMM structure. Note- I have now made a first pass at generating the 3D code; along with a template for the 3D BBFMM.  It runs (slowly), but has terrible errors- some optimization and bug fixing will be required.
