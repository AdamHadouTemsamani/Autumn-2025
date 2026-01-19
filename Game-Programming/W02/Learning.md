# Lecture 2

**Understand the impacts of implementing a physics system in your engine**
* Design: Implementing physics leads to a loss of control and predictability but allows for emergent behaviors.
* Content creation becomes more complex as assets (e.g., animations) need to interact correctly with physics.

**List common collision primitives**
* Spheres/Circles: The fastest primitive to test.
* Capsules: Commonly used for character controllers.
* AABB (Axis-Aligned Bounding Box):
* OBB (Oriented Bounding Box) 

**Explain the difference between a Convex and a Concave collider**
* Convex: Simple shapes where any line drawn between two internal points remains fully inside the shape (e.g., Sphere, Box).
* Concave: Complex shapes with indentations (e.g., a cross or a crescent). To detect collisions efficiently, concave objects are often approximated by multiple convex colliders (Compound Shapes) or broken down via Convex Decomposition

**Explain point<->sphere & sphere<->sphere collision detection**
* Point vs Sphere: A collision occurs if the distance between the point and the sphere's center is less than the radius ($|center - p| < radius$).
* Sphere vs Sphere: Checks if the distance between the two centers is less than the sum of their radii. 

**Describe collision queries, such as Ray Casting, Shape Casting & Sensors/Triggers**
Collision Queries: Hypothetical tests used to gather information rather than resolve physical impacts

Ray Casting: Shoots a line into the scene to detect what it hits.

Shape Casting: Sweeps a shape (like a sphere) along a path. This creates a "swept shape" (e.g., a capsule) and is useful for detecting fast-moving objects to prevent them from passing through thin walls (tunneling).

**Describe the difference between broad and narrow collision detection phase**
Broad-phase: Rapidly separates objects into groups to identify potential collisions. It uses spatial partitioning to reject distant pairs, avoiding an $O(n^2)$ check on every objec

Narrow-phase: Performs precise, expensive intersection tests only on the candidate pairs identified by the broad-phase

**Reason about performance considerations of collision detection**
Bounding Volume Hierarchies: Check simple bounding volumes (like AABBs) before checking the complex mesh inside.

Static Colliders: Optimizes performance by ignoring collisions between two immobile objects (e.g., the ground vs. a wall).

Collision Layers: Uses a matrix to define which object types should ignore each other (e.g., debris ignoring debris).

World Partitioning: Algorithms like Quadtrees/Octrees, Grid Partitioning, or Dynamic AABB Trees subdivide space so you only test objects within the same local region