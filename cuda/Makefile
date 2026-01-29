NVCC = nvcc
CFLAGS = -O3 -arch=sm_75
LDFLAGS = -lcudart

all: monster_walk_gpu

monster_walk_gpu: monster_walk_gpu.cu
	$(NVCC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f monster_walk_gpu

run: monster_walk_gpu
	./monster_walk_gpu

.PHONY: all clean run
