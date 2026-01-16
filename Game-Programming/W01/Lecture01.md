# Lecture 1 (Sprite Based Rendering)

## Types of delay

### Busy wait

```
SDL_Time walltime_busywait = walltime_work_end;
while(walltime_busywait - walltime_frame_beg < target_framerate_ns)
    SDL_GetCurrentTime(&walltime_busywait);
```

* Precise but costly.
* Continously checks the current time in a loop until the target frame rate is reached. 


### Simple Delay

```
// NOTE: `SDL_Delay` gets milliseconds, but our timer gives us
nanoseconds! We need to convert it manually
SDL_Delay((target_framerate_ns - time_elapsed_work) / 1000000);
```

* Simple, but too imprecise.
* Requires manual conversion as SDL_Delay uses milliseconds while requires nanoseconds.

### Delay NS

```
SDL_DelayNS(target_framerate_ns - time_elapsed_work);
```

* Higher resolution than simple delay, however still too imprecise.
* Uses SDL_DelayNS directly.

### Delay precise

```
SDL_DelayPrecise(target_framerate_ns - time_elapsed_work);
```

* Precise, as it attempts to wait as close to the requested time as possible
* May return later than expected due to OS scheduling
* Uses SDL_DelayPrevise, which may utilize busy waiting internally if necessary.

### Custom Delay

```
SDL_DelayNS(target_framerate_ns - time_elapsed_work - 1000000);
SDL_Time walltime_busywait = walltime_work_end;

while(walltime_busywait - walltime_frame_beg < target_framerate_ns)
    SDL_GetCurrentTime(&walltime_busywait);
```

* Balance between precision and cost.
* Uses a sleeping delay (SDL_DelayNS) with an arbitrary safety margin (1 millisecond), then "busy waits" (loops) what is left.

### Discussion

The custom delay is considered the best approach:
* Balances between precision and cost.
* Busy wait is very precise but too costly
* Simple/NS delay saves ppower, but is too imprecise, becuase the OS schedule can often make it over-sleep.
* Custom delay, sleep for most of the time, and then busy-wait for the tiny remainder.
  * High precision of a busy wait, with the low power consumption of a sleep
    * Sleeping is cheap, and busy wait (looping) is too expensive. 

 ## Sprite Based Rendering

 ### Questions of the lecture

 