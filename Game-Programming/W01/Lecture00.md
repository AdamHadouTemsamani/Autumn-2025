# Lecture 0

## Time and Loops 

The core of any game engine is a system of multiple loops
* Make sure every part of the engine is exected at the **correct time**
* And it is notified of any relevant interaction with external entities (player, os, etc.)

## Main Loop 

It consists of:

* init(): handles initialization
* process_events(): handles user input, OS interrupts, …
* update(): executes one simulation
step for all systems and game
elements
* render(): prepares data to send to
the renderer/GPU

### Simple 

```
// Simple loop
int main()
{
    init();
    while(!quit) // game loop
    {
        process_events();
        update();
        render();
    }
    return 0;
}
```

Effective, but too naive.
* Tries to execute everything as soon as possible
* Game will go faster or slower, depending on how much computation needs to be done.

### Fixed


```
// Fixed loop
int main()
{
    init();
    while(!quit) // game loop
    {
        float start = get_current_time();
        
        process_events();
        update();
        render();

        float end = get_current_time();
        float elapsed = end - start;
        
        sleep(MS_PER_FRAME - elapsed);
    }
    return 0;
}
```

* This partially addresses the variable computation length problem
* We need to specify target speed in milliseonds per frame
* After a single step is done, waits until the target time is elapsed
* The game cannot go too fast, but can still go too slow

### Fluid

```
int main()
{
    init();
    float start = get_current_time();
    float delta = 0;
 
    while(!quit) // game loop
    {
        process_events();
        update(delta);
        render();
 
        float end = get_current_time();
        delta = end - start;
        start = end;
    }
    return 0;
}
```

* Goes back running as fast as possible, but now also notifies the subsystem how much time passed since last computation
* **Works well in many cases**, but:
  * Slightly more complex
  * Subsystems and game code needs to be written to take advantage of this
  * Extreme variations in FPS will still create problem when dealing with user input (and won't look good)
  * Extremely power-hungry

---

### Discussion

**Fluid Loop Nuances**:
* Extreme variations in FPS are problematic especially when they are too low, as this affect user input
  * Must wait for update() and render() to finish before it can run again
* Input lag