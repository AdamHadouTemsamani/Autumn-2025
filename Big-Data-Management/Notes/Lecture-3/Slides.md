# Lecture 3 (MapReduce & Hadoop)

## Distributed Systems to Big Data Systems

Working with distirbuted systems is totally different from writing software on a single computer

**Problems with distributed systems:**
* Non-deterministic behavior
* Partial failures
* Debugging becomes very hard
* High probability of variace in runtime/throughput performance

### Google Cluster Idea (2003)

* Failures are the norm
* Data is growing: either large files or billions of small files
* Append-only instead of overwritting
  * Random write are paractically inexistent

**Goal**: throughput and not peak performance 
  * Individual, single task vs. Batch 

### Need for new tools
* Distributed File Systems (DFS)
  * Store petabytes of data in a cluster
  * Transparently handle fault-tolerance and replication
* Parallel Processing Platforms 
  * Offer a programming model that allows 

