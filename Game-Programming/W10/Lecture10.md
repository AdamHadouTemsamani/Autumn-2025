# Lecture 10 (Performance)

## Measurements and Data

### Importance of measuring

There is no single "fastest" algorithm (e.g., Quick Sort vs. Bubble Sort) in a vacuum. Performance depends on data quantity, implementation, and data regularity

In most answer we don't know the answer to a question "what is the fastest sorting algorithm", so the only way to figure out is measuring. 

Time Scales: 
* Modern CPUs run at billions of cycles per second (e.g., 3.4 GHz). A single cycle takes ~0.29 nanoseconds.

### Types of measuring

**Sampling (Profilers)**
* Method: The profiler periodically interrupts the CPU to check which function is currently running.
  * Pros: No code modification needed (no insturmnetation), unbiased, runs closer to final product.
  * Cons: Shows symptoms (where time is spent) rather than root causes

**High Frequency Clock**
* Using OS/APIs to time specific blocks of code.
  * Pros: Precise enough, measure very specific part of the code. 
  * Cons: require prior setup and insturmentation

**Cycle counter**
* Reading the CPU's internal instruction cycle counter (e.g., rdtsc, perf).
  * Pros: Highest resolution possible (not necessarily precision)
  * Cons: Very difficult to interpet. 

## Optimization types

### Asymptotical 

Describes how algorithms scales "towards infinity".
* However, we don't have infinite elements most of the time
* Usually algorithms that are asyptotically better are more complex.

### Avoid Useless work

Don't just optimize code; remove it. Eliminate "superfluous" work.
* SUch as reuse of code, over-aggressive chekcs, too many levels of abstractions

### Cache coherence

Organizing data to leverage the hardware cache. CPUs fetch memory in "lines"; accessing data sequentially is much faster than random access.
* Data-Oriented Design

### Cache results

Storing the result of an expensive calculation to reuse late.

Major risk: Cache Invalidation, reading data from a cache is not valid anymore (has to be refetched).

### Parallelization

Multithreading: Spreading distinct tasks across multiple CPU cores.

SIMD (Single Instruction, Multiple Data): Using special CPU registers (SSE, AVX) to perform math on multiple numbers (e.g., 4 vectors) in a single instruction cycle.


### Specialization

Generic code is often slow because it handles every edge case. Specialized code (handling only your specific game's case) allows for "fast paths"

### Domain Knowledge Approximation

In games, "believable" > "correct." You can cheat if the player doesn't notice
* objects that are far away from the camera
can be rendered at lower resolutions
* Letting AI make decisions based on world data from 2 frames ago to avoid waiting for the current frame's update

