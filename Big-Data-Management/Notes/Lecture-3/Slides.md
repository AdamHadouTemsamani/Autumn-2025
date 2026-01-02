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
**Distributed File Systems (DFS)**
  * Store petabytes of data in a cluster
  * Transparently handle fault-tolerance and replication
**Parallel Processing Platforms**
  * Offer a programming model that allows developers to easily write distributed applications
  * Alleviate developer from handling concurrency, network communication, and machine failured

## Distirbuted File System (DFS)

Files should be: enormous and rarely updates
  
* Stores very large data files efficinelty and reliable.
* Files are partitioned into ficed-sized chunks (e.g., 64MB)
  * Stored as Linux files
* Chunks are replicated (typically three times)

### Google File System (GFS) architecture
* Single master, multiple chunk servers
* Master (Coordinator)
  * maintains a map of where data lives (which chunk servers holds which specific chunk of a file)
* Chunk Servers (Workers)
  * Store actual data. Files are split into fixed-size chunks
![alt text](images/gfs.png)

Master is a potential single point of failure / also a scalability bottleneck
* Solution: use shadow masters; secondary master servers that provide highly available, read-only access to the file system when the primary master fails

### Hadoop Distributed File System (HDFS)


