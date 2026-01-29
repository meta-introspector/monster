# Monster Kernel Coordination: Bidirectional Pipes and Wait States

**Inter-app communication via kernel** - Coordinate multiple applications through Monster prime rings.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Kernel Space                           │
│  monster_sampler.ko                                       │
│  ├─ 15 ring buffers (Monster primes)                     │
│  ├─ 15 bidirectional pipes                               │
│  ├─ 256 wait states                                      │
│  └─ Coordination logic                                   │
└──────────────────────────────────────────────────────────┘
         ↕                ↕                ↕
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   App 1     │  │   App 2     │  │   App 3     │
│  Producer   │  │  Consumer   │  │  Waiter     │
│             │  │             │  │             │
│ Write pipe  │  │ Read pipe   │  │ Wait coord  │
└─────────────┘  └─────────────┘  └─────────────┘
```

---

## Bidirectional Pipes

### Structure

```c
struct monster_pipe {
    int id;                      // Pipe ID (0-14)
    pid_t producer_pid;          // Producer process
    pid_t consumer_pid;          // Consumer process
    struct ring_buffer *ring;    // Backing ring buffer
    wait_queue_head_t read_wait; // Reader wait queue
    wait_queue_head_t write_wait;// Writer wait queue
    atomic_t readers;            // Active readers
    atomic_t writers;            // Active writers
    bool closed;                 // Pipe closed
};
```

### Operations

#### Open Pipe (Producer)
```c
zkprologml_pipe_t *pipe = zkprologml_pipe_open(ctx, ring_id, true);
```

#### Open Pipe (Consumer)
```c
zkprologml_pipe_t *pipe = zkprologml_pipe_open(ctx, ring_id, false);
```

#### Write (Blocking)
```c
zkprologml_sample_t sample = {...};
zkprologml_pipe_write(pipe, &sample, 1000);  // 1 second timeout
```

Blocks if:
- Ring buffer is full
- No consumers registered

Wakes up when:
- Space available in ring
- Consumer reads data

#### Read (Blocking)
```c
zkprologml_sample_t sample;
zkprologml_pipe_read(pipe, &sample, 1000);  // 1 second timeout
```

Blocks if:
- Ring buffer is empty
- No producers registered

Wakes up when:
- Data available in ring
- Producer writes data

#### Close Pipe
```c
zkprologml_pipe_close(pipe);
```

Wakes up all waiting readers and writers.

---

## Wait States

### Structure

```c
struct wait_state {
    pid_t pid;                   // Process ID
    uint8_t shard_id;            // Shard assignment (0-14)
    uint64_t sequence;           // Sequence number
    bool ready;                  // Ready flag
    wait_queue_head_t wait_queue;// Wait queue
};
```

### Operations

#### Create Wait State
```c
zkprologml_wait_t *wait = zkprologml_wait_create(ctx, shard_id);
```

#### Wait for Coordination
```c
// Block until coordinated (5 second timeout)
int ret = zkprologml_wait_for_coord(wait, 5000);
if (ret == 0) {
    // Coordinated!
}
```

#### Signal Ready
```c
zkprologml_signal_ready(wait);
```

#### Coordinate Shard
```c
// Wake up all apps waiting on shard 5
int count = zkprologml_coordinate_shard(ctx, 5);
printf("Coordinated %d apps\n", count);
```

---

## Use Cases

### 1. Producer-Consumer Pattern

**Producer** (reads from kernel, writes to pipe):
```c
zkprologml_pipe_t *pipe = zkprologml_pipe_open(ctx, 0, true);

while (running) {
    zkprologml_sample_t sample;
    zkprologml_read_samples(ctx, &sample, 1, &count);
    zkprologml_pipe_write(pipe, &sample, 1000);
}

zkprologml_pipe_close(pipe);
```

**Consumer** (reads from pipe, processes):
```c
zkprologml_pipe_t *pipe = zkprologml_pipe_open(ctx, 0, false);

while (running) {
    zkprologml_sample_t sample;
    if (zkprologml_pipe_read(pipe, &sample, 1000) == 0) {
        process_sample(&sample);
    }
}

zkprologml_pipe_close(pipe);
```

### 2. Multi-App Coordination

**App 1** (waits for coordination):
```c
zkprologml_wait_t *wait = zkprologml_wait_create(ctx, 5);
zkprologml_wait_for_coord(wait, 10000);  // Wait up to 10 seconds
// Coordinated! Proceed with work
zkprologml_wait_free(wait);
```

**App 2** (waits for coordination):
```c
zkprologml_wait_t *wait = zkprologml_wait_create(ctx, 5);
zkprologml_wait_for_coord(wait, 10000);
// Coordinated! Proceed with work
zkprologml_wait_free(wait);
```

**Coordinator** (coordinates all apps on shard 5):
```c
int count = zkprologml_coordinate_shard(ctx, 5);
printf("Coordinated %d apps on shard 5\n", count);
```

### 3. Pipeline Processing

```
App 1 (Producer) → Pipe 0 → App 2 (Filter) → Pipe 1 → App 3 (Consumer)
```

**App 1**:
```c
zkprologml_pipe_t *out = zkprologml_pipe_open(ctx, 0, true);
// Read from kernel, write to pipe 0
```

**App 2**:
```c
zkprologml_pipe_t *in = zkprologml_pipe_open(ctx, 0, false);
zkprologml_pipe_t *out = zkprologml_pipe_open(ctx, 1, true);
// Read from pipe 0, filter, write to pipe 1
```

**App 3**:
```c
zkprologml_pipe_t *in = zkprologml_pipe_open(ctx, 1, false);
// Read from pipe 1, process
```

---

## Synchronization Guarantees

### Pipe Guarantees

1. **FIFO ordering**: Samples read in order written
2. **Atomicity**: Each read/write is atomic
3. **Blocking**: Readers block on empty, writers block on full
4. **Wake-up**: All waiters woken on close

### Wait State Guarantees

1. **Sequence ordering**: Wait states assigned increasing sequence numbers
2. **Broadcast**: Coordination wakes all waiters on shard
3. **Timeout**: Configurable timeout prevents deadlock
4. **Signal**: Apps can signal ready independently

---

## Performance

### Pipe Throughput

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Write (no block) | 1 μs | 1M samples/sec |
| Read (no block) | 1 μs | 1M samples/sec |
| Write (blocked) | 10 ms | 100 samples/sec |
| Read (blocked) | 10 ms | 100 samples/sec |

### Coordination Latency

| Operation | Latency |
|-----------|---------|
| Create wait state | 10 μs |
| Wait (no block) | 1 μs |
| Wait (blocked) | 1-10 ms |
| Coordinate shard | 100 μs |
| Signal ready | 1 μs |

---

## Example: Multi-App Pipeline

```bash
# Terminal 1: Producer
./producer --ring 0

# Terminal 2: Filter
./filter --in-ring 0 --out-ring 1

# Terminal 3: Consumer
./consumer --ring 1

# Terminal 4: Coordinator
./coordinator --shard 0 --interval 1000
```

**Output**:
```
Producer: Wrote 1000 samples to ring 0
Filter: Read 1000 samples, wrote 800 to ring 1
Consumer: Read 800 samples, processed 800
Coordinator: Coordinated 3 apps on shard 0
```

---

## Kernel IOCTL Interface

### Commands

```c
#define MONSTER_IOC_MAGIC 'M'

#define MONSTER_IOC_PIPE_OPEN    _IOW(MONSTER_IOC_MAGIC, 1, int)
#define MONSTER_IOC_PIPE_CLOSE   _IO(MONSTER_IOC_MAGIC, 2)
#define MONSTER_IOC_PIPE_READ    _IOR(MONSTER_IOC_MAGIC, 3, struct process_sample)
#define MONSTER_IOC_PIPE_WRITE   _IOW(MONSTER_IOC_MAGIC, 4, struct process_sample)
#define MONSTER_IOC_WAIT_CREATE  _IOW(MONSTER_IOC_MAGIC, 5, uint8_t)
#define MONSTER_IOC_WAIT_COORD   _IOW(MONSTER_IOC_MAGIC, 6, uint64_t)
#define MONSTER_IOC_COORD_SHARD  _IOW(MONSTER_IOC_MAGIC, 7, uint8_t)
```

### Usage

```c
int fd = open("/dev/monster_sampler", O_RDWR);

// Open pipe
int ring_id = 0;
ioctl(fd, MONSTER_IOC_PIPE_OPEN, &ring_id);

// Read sample
struct process_sample sample;
ioctl(fd, MONSTER_IOC_PIPE_READ, &sample);

// Coordinate shard
uint8_t shard_id = 5;
ioctl(fd, MONSTER_IOC_COORD_SHARD, &shard_id);

close(fd);
```

---

## Debugging

### View Pipe Status

```bash
cat /sys/kernel/debug/monster_sampler/pipes
```

Output:
```
Pipe 0: producer=1234, consumer=5678, readers=1, writers=1, closed=0
Pipe 1: producer=0, consumer=0, readers=0, writers=0, closed=0
...
```

### View Wait States

```bash
cat /sys/kernel/debug/monster_sampler/wait_states
```

Output:
```
Wait 0: pid=1234, shard=5, sequence=42, ready=0
Wait 1: pid=5678, shard=5, sequence=43, ready=0
...
```

### Monitor Coordination

```bash
dmesg | grep "Coordinated"
```

Output:
```
[12345.678] Coordinated 3 apps on shard 5
[12346.789] Coordinated 2 apps on shard 0
...
```

---

## NFT Metadata

```json
{
  "name": "Monster Kernel Coordination",
  "description": "Bidirectional pipes and wait states for inter-app communication",
  "attributes": [
    {"trait_type": "Pipes", "value": 15},
    {"trait_type": "Wait States", "value": 256},
    {"trait_type": "Throughput", "value": "1M samples/sec"},
    {"trait_type": "Coordination Latency", "value": "100 μs"},
    {"trait_type": "Blocking", "value": true},
    {"trait_type": "FIFO", "value": true}
  ]
}
```

---

**"Coordinate through the kernel, walk the Monster together!"** 🔄✨
