# Lecture 3

**Understand the concept of multiple coordinate systems and why is that useful**
Instead of using raw pixels, games use abstract coordinate frames to define positions relative to specific reference points (frames of reference).

Object Space: Relative to the texture itself (Origin: Top-Left, Up: Negative Y).

World Space: Standard math coordinates for game logic (Origin: Bottom-Left, Up: Positive Y).

Camera Space: Relative to the view (Origin: Center, Up: Positive Y)

Screen Space: Relative to the monitor pixels (Origin: Top-Left, Up: Negative Y).

Why is it?
It decouples game code from resources, ensuring changes in monitor or window size do not break the game.

Allows designers to reason about space in a way that suits the logic.

Enables rendering of objects that exist outside the current screen boundaries.

Allows for reversible operations like translation, rotation, and scaling to move between these frames.

**Learn how to develop header-only style libraries**
Classic Build (Separate Compilation): The standard C++ approach.
* Structure: Uses Header files (.h) for declarations (data types, function declarations) shared among files, and Source files (.cpp) for executable implementation.
* Each source file is compiled into an object file (.obj) and then linked to the executable.
* Very slow compilation on large codebases, and complex depedency tracking

Unity Build: An alternative approach often used to speed up
* Combines all code into a single compilation unit (a single file that includes all other code)
* Offers very fast compilation and requires no complex build system, but code can become difficult to manage ("spiral") if not tended to.

