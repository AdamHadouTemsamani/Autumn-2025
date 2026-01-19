# Lecture 00

Describe the basics structure of an interactive application:
* The engine relies on multiple loops to execute parts at the correct time and handle external interactions.
* Main Loop Stages:
  * init(): Handles initialization.
  * process_events(): Manages user input and OS interrupts.
  * update(): Executes simulation steps for systems.
  * render(): Prepares data for the GPU.
* Simple: Executes as fast as possible; game will go faster or slower depending on computation load
* Fixed: Waits for a target time (milliseconds per frame) to elapse; prevents running too fast but not too slow.
* Fluid: Passes a delta (time elapsed) to the update function; handles speed variations better but can suffer from input lag at low FPS.

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

## Lecture 01

**Describe the relationship between textures and sprites**
* Textures: Images uploaded to the GPU used for 2D and 3D graphics to store colors, shadows, or geometry data
* Sprite: A high-level representation containing the info needed to render a texture
  * Includes Texture, Position/Size, Tint, and Pivot.
  * Pivot: The logical center of a sprite (e.g., a joint for rotation rather than the image center).

**Describe the concept of a sprite atlas**
* It is more efficiently packing multiple sprites into a single texture.
  * Arrangements: 
    * Uniform Grid: Trivial to generate/access but wastes space with non-uniform shapes.
    * Packed: Requires algorithms to generate and extract but minimizes wasted space.

**Rendering System (2D Graphics)**
  * Raster Grapihcs: An image produced as an array (raster) of picture picture elemnts stored in a frame buffer 
    * Consists of a grid with number representing Red, Green, Blue
    * Any rectangular shape
  * Modern engines treat 2D graphics as special case of 3D; images are turned into triangles (three edges, three vertices) for rendering.
  * Timing/Delays: A rendering system needs precise timing loops
    * Busy Wait: Precise but costly CPU usage. (for loop until time has passed)
    * Simple/NS Delay: Low power but imprecise due to OS scheduling. (Sleep for x time)
    * Custom Delay: The best approach; sleeps for the bulk of the time, then busy-waits the remainder for precision.

**Use vectors to position objects in a game world**
* Sum ($a+b$): Used for movement (Position + Velocity). 
* Subtraction ($a-b$): Finds the vector pointing from one object to another.
* Scale ($a*f$): Changes speed or size.
* Dot Product: Scalar value used to determine angles/facing directions.
  * 0: Perpendicular, > 0: Same direction. < 0: Opposite direction

**Review some of the non-basic C++ topics used in the course**
  * Memory Management:
    * Code (Text): Contains machine code.
    * Static/Global: Stores global and static variables.
    * Stack: Stores local variables and parameters; fixed size; frames allocated per function call
    * Heap: General-purpose dynamic memory; grows as needed; manually managed (alloc/free).
  * Macros:
    * Preprocessor: Text replacement before compilation.
    * Should be used with caution