#include "cuda_space_model.h"
#include "helper_cuda.h"
#include "space_model.h"
#include "stdio.h"
#include "stdlib.h"

// input - body nodes
__constant__ volatile float *mass_const;
__constant__ volatile float *point_x_const, *point_y_const;
__constant__ volatile float *speed_x_const, *speed_y_const;
__constant__ volatile float *acc_x_const, *acc_y_const;

__constant__ int capacity_const;
__device__ volatile int object_len;
__device__ volatile float length_const;

// cell nodes
__constant__ volatile int *node_cnt_const, *node_child_const;
__constant__ volatile float *node_mass_const;
__constant__ volatile float *node_point_x_const, *node_point_y_const;

// sorting related
__constant__ volatile int *inorder_const, *sort_const;

// calculation related
__constant__ float dt_const;

#define NUM_BLOCK 12
#define LOCK -2
#define NULL_POINTER -1

__device__ __inline__ int getQuadrant(float root_x, float root_y, float x,
                                      float y) {
  int idx = 0;

  if (root_x < x) {
    idx += 1;
  }

  if (root_y < y) {
    idx += 2;
  }

  return idx;
}


__device__ __inline__ void shuffleNonNullPointer(int quad_idx, int nonnull_idx,
                                                 int child_idx, int cell_idx) {
  if (quad_idx != nonnull_idx) {
    node_child_const[cell_idx * 4 + nonnull_idx] = child_idx;
    node_child_const[cell_idx * 4 + quad_idx] = -1;
  }

  return;
}



// compute 1 / sqrt of the displacement
__device__ __inline__ float getDistance(float x, float y) {
  return rsqrtf(x * x + y * y + SOFT_CONST);
}



__device__ void update_acc(float mass, float r_inv, float dr_x, float dr_y, float *acc_x, float *acc_y) {
  float F = mass * G_CONST * pow(r_inv, 3.0f);
  *acc_x += dr_x * F;
  *acc_y += dr_y * F;
}



// kenerl1: Init root
__global__ void kernel1Init() {
  int root_idx = 0;

  for (int i = 0; i < 4; i++) {
    node_child_const[4 * root_idx + i] = NULL_POINTER;
  }

  node_mass_const[root_idx] = -1.f;     // -1.f represents no mass computed yet
  node_point_x_const[root_idx] = WINDOW_W / 2; 
  node_point_y_const[root_idx] = WINDOW_H / 2;
  object_len = 0;
  length_const = WINDOW_H / 2 + BORDER;
  inorder_const[root_idx] = 0;

#if __CUDA_ARCH__ >= 200
  // if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
  //   printf("Hello from kernel 1\n");
  // }
#endif
}



// kernel2: Build hierarchical decomposition by inserting each body into
// quadtree
__global__ void kernel2BuildTree() {
#if __CUDA_ARCH__ >= 200
// if (blockIdx.x * blockDim.x + threadIdx.x == 0)
//   printf("Hello from kernel 2\n");

  // caches the root’s data in the register file
  register float length = length_const;
  register float root_point_x = node_point_x_const[0];
  register float root_point_y = node_point_y_const[0];
  register int idx = blockIdx.x * blockDim.x + threadIdx.x;
  register int step = blockDim.x * gridDim.x;

  register float body_point_x = 0.f; 
  register float body_point_y = 0.f; 
  register float cell_point_x = 0.f; 
  register float cell_point_y = 0.f; 

  register int new_node_idx = 0;
  register int insert_idx = 0;   
  register int quad_idx = 0;     
  register float curr_length;    
  register int child;

  register int target = 0;       
  register int new_cell_idx;    
  register bool first_insert = true;
  register int quad_iter = 0;


  while (idx < capacity_const) {
    body_point_x = point_x_const[idx];
    body_point_y = point_y_const[idx];

    // reset for each iter
    curr_length = length;
    insert_idx = 0;
    first_insert = true;

    // find the cell
    while (true) {
      quad_idx = getQuadrant(node_point_x_const[insert_idx],
                             node_point_y_const[insert_idx], body_point_x,
                             body_point_y);
      child = node_child_const[insert_idx * 4 + quad_idx];

      if (child < capacity_const)
        break;
      
      curr_length *= 0.5;
      insert_idx = child-capacity_const;
    }


    if (child != LOCK) {
      target = insert_idx * 4 + quad_idx;
      if (child == atomicCAS((int *)&node_child_const[target], child, LOCK)) {
        if (child == NULL_POINTER) {
          node_child_const[target] = idx; // insert body and release lock

        } else {                          // colided with another body
          do {
            new_node_idx = atomicAdd((int *)&object_len, 1) + 1; // atomically get the next unused cell
            
            if (first_insert) {
              new_cell_idx = new_node_idx;
            }

            // finder center coordinate of the new cell
            curr_length *= 0.5f;
            cell_point_x = node_point_x_const[insert_idx] -
                          pow((double)-1.f, (double)((quad_idx) & 1)) * curr_length;
            cell_point_y = node_point_y_const[insert_idx] -
                            pow((double)-1.f, (double)((quad_idx >> 1) & 1)) * curr_length;

            // init new cell
            node_point_x_const[new_node_idx] = cell_point_x;
            node_point_y_const[new_node_idx] = cell_point_y;
            node_mass_const[new_node_idx] = -1.0f;
            inorder_const[new_node_idx] = -1;

            for (quad_iter = 0; quad_iter < 4; quad_iter++) {
              node_child_const[new_node_idx * 4 + quad_iter] = NULL_POINTER;
            }

            // insert new cell if not the first insert if not the first insert
            // do not insert the first new cell to avoid releasing lock too early
            if (!first_insert) {
              node_child_const[insert_idx * 4 + quad_idx] = new_node_idx + capacity_const;
            } else {
              first_insert = false;
            }

            // update collided body to the new cell
            quad_idx = getQuadrant(cell_point_x, cell_point_y,
                                   point_x_const[child], point_y_const[child]);
            node_child_const[new_node_idx * 4 + quad_idx] = child;

            // check for further collisions
            insert_idx = new_node_idx;
            quad_idx = getQuadrant(cell_point_x, cell_point_y, body_point_x,
                                   body_point_y);
            child = node_child_const[insert_idx * 4 + quad_idx];
          } while (child >= 0);

          // insert new body
          node_child_const[insert_idx * 4 + quad_idx] = idx;

          // make sure newcell subtree is visible
          __threadfence();

          // insert new_cell and release lock
          node_child_const[target] = new_cell_idx + capacity_const;
        }
        idx += step;
      }
    }

    // wait for other warps to finish insertion
    __syncthreads();
  }
#endif
}

// kernel3: Summarize body information in each internal octree node
__global__ void kernel3Summarize() {
#if __CUDA_ARCH__ >= 200
  // if (blockIdx.x * blockDim.x + threadIdx.x == 0)
  //   printf("Hello from kernel 3\n");

  register int idx = object_len - blockIdx.x * blockDim.x - threadIdx.x;
  register int step = blockDim.x * gridDim.x;

  register int quad_iter;
  register int nonnull_idx; // keep track of non-null pointer
  register int cnt;
  register int child_idx;
  register int missing = 0;

  register float mass;
  register float cumulative_mass;
  register float sum_x; // sum of product of mass * point_x
  register float sum_y; // sum of product of mass * point_y

  __shared__ volatile int cache[256 * 4]; // NUM_THREAD * 4

  while (idx >= 0) {
    // initialize default settings for each cell
    nonnull_idx = 0;
    sum_x = 0.f;
    sum_y = 0.f;

    if (missing == 0) {
      cumulative_mass = 0.f;
      cnt = 0;

      // initialize center of gravity
      for (quad_iter = 0; quad_iter < 4; quad_iter++) {
        child_idx = node_child_const[idx * 4 + quad_iter];

        // iterate over existing children ONLY
        if (child_idx >= 0) {
          // reset mass for each child
          mass = 0.f;

          // move the existing children to the front of child array in each
          // cell and move all the nulls to the end
          shuffleNonNullPointer(quad_iter, nonnull_idx, child_idx, idx);

          if (child_idx < capacity_const) { // body
            mass = mass_const[child_idx];
          } else { // cell
            mass = node_mass_const[child_idx - capacity_const];
          }

          if (mass >= 0.f) { // child is ready
            // add its contribution to center of gravity
            cumulative_mass += mass;

            if (child_idx < capacity_const) { // body
              sum_x += point_x_const[child_idx] * mass;
              sum_y += point_y_const[child_idx] * mass;
              cnt++;
            } else { // cell
              sum_x += node_point_x_const[child_idx - capacity_const] * mass;
              sum_y += node_point_y_const[child_idx - capacity_const] * mass;
              cnt += node_cnt_const[child_idx - capacity_const];
            }
          } else {
            // cache child index
            cache[missing * 256 + threadIdx.x] = child_idx;
            missing++;
          }
          nonnull_idx++;
        }
      }
    } 

    if (missing != 0) {
      do {
        child_idx = cache[(missing - 1) * 256 + threadIdx.x];
        mass = node_mass_const[child_idx - capacity_const];

        // check if child in cache is ready
        // if not,  break out of the loop
        // "thread divergence deliberately forces the thread to wait for
        // a while before trying again to throttle polling requests."
        if (mass >= 0.f) {
          // remove from cache and add its contribution to center of gravity
          missing--;
          cumulative_mass += mass;

          sum_x += node_point_x_const[child_idx - capacity_const] * mass;
          sum_y += node_point_y_const[child_idx - capacity_const] * mass;
          cnt += node_cnt_const[child_idx - capacity_const];
        }
      } while (mass >= 0.f && missing != 0);
    }

    if (missing == 0) {
      // store center of gravity
      node_point_x_const[idx] = sum_x / cumulative_mass;
      node_point_y_const[idx] = sum_y / cumulative_mass;

      // store cumulative count
      node_cnt_const[idx] = cnt;

      __threadfence(); // make sure center of gravity is visible

      // store cumulative mass
      node_mass_const[idx] = cumulative_mass;

      __threadfence(); // make sure to sync before next iteration

      idx -= step;
    }
  }
#endif
}

// kernel4: Approximately sort the bodies by spatial distance
__global__ void kernel4Sort() {
#if __CUDA_ARCH__ >= 200
  // if (blockIdx.x * blockDim.x + threadIdx.x == 0)
  //   printf("Hello from kernel 4 \n");

  register int idx = blockIdx.x * blockDim.x + threadIdx.x;
  register int step = blockDim.x * gridDim.x;

  register int child_idx;
  register int quad_iter; // traverse 4 child
  register int inorder_rank;

  // top-down traversal of cell nodes
  while (idx <= object_len) {
    inorder_rank = inorder_const[idx];
    
    // check if rank has been assigned
    if (inorder_rank >= 0) {
      for (quad_iter = 0; quad_iter < 4; quad_iter++) {
        child_idx = node_child_const[idx * 4 + quad_iter];

        if (child_idx >= capacity_const) { // cell
          child_idx -= capacity_const;
          inorder_const[child_idx] = inorder_rank;
          inorder_rank += node_cnt_const[child_idx];
        } else if (child_idx >= 0) { // body
          sort_const[inorder_rank] = child_idx;
          inorder_rank++;
          
        }
        
      }
      idx += step;
    }
    __threadfence();
  }
  __syncthreads();
#endif
}



__device__ float2 kernel5HelperComputeForNode(int node_idx) {
  register int stack[2048];
  register int depth = 0;

  // push root node onto the stack
  stack[depth++] = length_const;
  stack[depth++] = capacity_const;

  // cache
  register int curr_idx = 0;
  register int curr_length = 0;
  register float r_inv = 0.f;
  register float dr_x = 0.f, dr_y = 0.f;
  register float acc_x = 0.f, acc_y = 0.f;
  register float mass;
  register float point_x = point_x_const[node_idx];
  register float point_y = point_y_const[node_idx];

  while (depth > 0) {
    curr_idx = stack[--depth];
    curr_length = stack[--depth];

    if (curr_idx >= 0 && curr_idx < capacity_const) { // body node
      if (curr_idx != node_idx) {
        dr_x = point_x_const[curr_idx] - point_x;
        dr_y = point_y_const[curr_idx] - point_y;
        mass = mass_const[curr_idx];

        r_inv = getDistance(dr_x, dr_y);
        update_acc(mass, r_inv, dr_x, dr_y, &acc_x, &acc_y);
      }

    } else { // cell node
      curr_idx -= capacity_const;
      dr_x = node_point_x_const[curr_idx] - point_x;
      dr_y = node_point_y_const[curr_idx] - point_y;
      mass = node_mass_const[curr_idx];

      // if the cell distance is sufficiently far way
      if (curr_length * r_inv < SD_TRESHOLD) {
        r_inv = getDistance(dr_x, dr_y);
        update_acc(mass, r_inv, dr_x, dr_y, &acc_x, &acc_y);
      } else {
        for (int quad_iter = 0; quad_iter < 4; quad_iter++) {
          // add the length and child_idx of children that are not null
          if (node_child_const[4 * curr_idx + quad_iter] != NULL_POINTER) {
            stack[depth++] = curr_length * 0.5;
            stack[depth++] = node_child_const[4 * curr_idx + quad_iter];
          } else {
            break;  // early return
          }
        }
      }
    }
  }
  __syncthreads();

  return make_float2(acc_x, acc_y);
}

// kernel5: Compute forces acting on each body with help of quadtree
__global__ void kernel5Compute() {
#if __CUDA_ARCH__ >= 200
  // if (blockIdx.x * blockDim.x + threadIdx.x == 0)
  //   printf("Hello from kernel 5\n");

  register int idx;
  register int step = blockDim.x * gridDim.x;

  register int node_idx;
  register float2 acc;

  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < capacity_const; idx += step) {
    node_idx = sort_const[idx];

    // precompute and cache info
    acc = kernel5HelperComputeForNode(node_idx);

    acc_x_const[node_idx] = acc.x;
    acc_y_const[node_idx] = acc.y;
  }
#endif
}

// kernel 6: Update body positions and velocities
__global__ void kernel6Update() {
#if __CUDA_ARCH__ >= 200
  // if (blockIdx.x * blockDim.x + threadIdx.x == 0)
  //   printf("Hello from kernel 6\n");
  
  register int idx;
  register int step = blockDim.x * gridDim.x;

  register float delta_speed_x, delta_speed_y;
  register float speed_x, speed_y;

  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < capacity_const; idx += step) {
    delta_speed_x = acc_x_const[idx] * dt_const;
    delta_speed_y = acc_x_const[idx] * dt_const;

    speed_x = speed_x_const[idx] + delta_speed_x;
    speed_y = speed_y_const[idx] + delta_speed_y;

    speed_x_const[idx] = speed_x;
    speed_y_const[idx] = speed_y;

    point_x_const[idx] += speed_x * dt_const;
    point_y_const[idx] += speed_y * dt_const;

  }

#endif
}

void main_update(SpaceModel *m, SimulationConfig config, GS_FLOAT dt) {

  float *mass, *point_x, *point_y, *speed_x, *speed_y; // host vars
  float *mass_dev, *point_x_dev, *point_y_dev, *speed_x_dev, *speed_y_dev,
      *acc_x, *acc_y; // device vars

  size_t capacity;
  int cell_capacity;
  float max_split;

  int len;
  int *len_dev;

  len = m->objects->len;
  capacity =  len;

  // calculate the max # of potential splits in a galaxy to estiamte size of cell array
  max_split = log2f(config.view_bounds.size.x / config.galaxy_size) * config.objects_n;
  cell_capacity = config.galaxies_n * round(max_split);

  cudaMalloc((void **)&len_dev, sizeof(int));
  cudaMemcpy(len_dev, &len, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(object_len, &len_dev, sizeof(int));

  mass = (float *)malloc(sizeof(float) * capacity);
  point_x = (float *)malloc(sizeof(float) * capacity);
  point_y = (float *)malloc(sizeof(float) * capacity);
  speed_x = (float *)malloc(sizeof(float) * capacity);
  speed_y = (float *)malloc(sizeof(float) * capacity);

  // Flattening out the Object struct into multiple arrays
  // One array per field
  for (int i = 0; i < capacity; i++) {
    mass[i] = m->objects->objects[i].mass;
    point_x[i] = m->objects->objects[i].position.x;
    point_y[i] = m->objects->objects[i].position.y;
    speed_x[i] = m->objects->objects[i].speed.x;
    speed_y[i] = m->objects->objects[i].speed.y;
  }

  cudaMalloc(&mass_dev, sizeof(float) * (capacity + 1));
  cudaMalloc(&point_x_dev, sizeof(float) * (capacity + 1));
  cudaMalloc(&point_y_dev, sizeof(float) * (capacity + 1));
  cudaMalloc(&speed_x_dev, sizeof(float) * (capacity + 1));
  cudaMalloc(&speed_y_dev, sizeof(float) * (capacity + 1));
  cudaMalloc(&acc_x, sizeof(float) * (capacity + 1));
  cudaMalloc(&acc_y, sizeof(float) * (capacity + 1));

  cudaMemcpy(mass_dev, mass, sizeof(float) * capacity, cudaMemcpyHostToDevice);
  cudaMemcpy(point_x_dev, point_x, sizeof(float) * capacity,
             cudaMemcpyHostToDevice);
  cudaMemcpy(point_y_dev, point_y, sizeof(float) * capacity,
             cudaMemcpyHostToDevice);
  cudaMemcpy(speed_x_dev, speed_x, sizeof(float) * capacity,
             cudaMemcpyHostToDevice);
  cudaMemcpy(speed_y_dev, speed_y, sizeof(float) * capacity,
             cudaMemcpyHostToDevice);

  // Copy device memory to constant memory
  cudaMemcpyToSymbol(mass_const, &mass_dev, sizeof(void *));
  cudaMemcpyToSymbol(point_x_const, &point_x_dev, sizeof(void *));
  cudaMemcpyToSymbol(point_y_const, &point_y_dev, sizeof(void *));
  cudaMemcpyToSymbol(speed_x_const, &speed_x_dev, sizeof(void *));
  cudaMemcpyToSymbol(speed_y_const, &speed_y_dev, sizeof(void *));
  cudaMemcpyToSymbol(acc_x_const, &acc_x, sizeof(void *));
  cudaMemcpyToSymbol(acc_y_const, &acc_y, sizeof(void *));

  // cell vars
  int *node_cnt, *node_child;
  float *node_mass, *node_point_x, *node_point_y;

  cudaMalloc(&node_cnt, sizeof(int) * (cell_capacity + 1));
  cudaMalloc(&node_child, sizeof(int) * (cell_capacity + 1) * 4);
  cudaMalloc(&node_mass, sizeof(float) * (cell_capacity + 1));
  cudaMalloc(&node_point_x, sizeof(float) * (cell_capacity + 1));
  cudaMalloc(&node_point_y, sizeof(float) * (cell_capacity + 1));

  // initialize all counts to 0
  cudaMemset(node_cnt, 0, sizeof(int) * (cell_capacity + 1));

  cudaMemcpyToSymbol(node_cnt_const, &node_cnt, sizeof(void *));
  cudaMemcpyToSymbol(node_child_const, &node_child, sizeof(void *));
  cudaMemcpyToSymbol(node_mass_const, &node_mass, sizeof(void *));
  cudaMemcpyToSymbol(node_point_x_const, &node_point_x, sizeof(void *));
  cudaMemcpyToSymbol(node_point_y_const, &node_point_y, sizeof(void *));

  // for sorting
  int *inorder_rank, *sort;
  cudaMalloc((void **)&inorder_rank, sizeof(int) * (cell_capacity + 1));
  cudaMalloc((void **)&sort, sizeof(int) * (capacity + 1));

  cudaMemcpyToSymbol(inorder_const, &inorder_rank, sizeof(void *));
  cudaMemcpyToSymbol(sort_const, &sort, sizeof(void *));

  cudaMemcpyToSymbol(capacity_const, &capacity, sizeof(size_t));
  cudaMemcpyToSymbol(dt_const, &dt, sizeof(float));

  // alternative to kernel1
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);
  cudaEventRecord (start, 0);
  kernel1Init<<<NUM_BLOCK * 8, 256>>>();
  cudaEventRecord (stop, 0);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&time, start, stop);
  cudaEventDestroy (start);
  cudaEventDestroy (stop);
  // printf("Kernel 1 time: %3.4f \n", time);

  // kernel2: Build hierarchical decomposition by inserting each body into
  // quadtree
  cudaEventCreate (&start);
  cudaEventCreate (&stop);
  cudaEventRecord (start, 0);

  kernel2BuildTree<<<NUM_BLOCK * 8, 256>>>();

  cudaEventRecord (stop, 0);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&time, start, stop);
  cudaEventDestroy (start);
  cudaEventDestroy (stop);
  // printf("Kernel 2 time: %3.4f \n", time);

  // kernel3: Summarize body information in each internal octree node
  cudaEventCreate (&start);
  cudaEventCreate (&stop);
  cudaEventRecord (start, 0);

  kernel3Summarize<<<NUM_BLOCK * 8, 256>>>();

  cudaEventRecord (stop, 0);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&time, start, stop);
  cudaEventDestroy (start);
  cudaEventDestroy (stop);
  // printf("Kernel 3 time: %3.4f \n", time);
  
  // kernel4: Approximately sort the bodies by spatial distance
  cudaEventCreate (&start);
  cudaEventCreate (&stop);
  cudaEventRecord (start, 0);

  kernel4Sort<<<NUM_BLOCK * 8, 256>>>();

  cudaEventRecord (stop, 0);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&time, start, stop);
  cudaEventDestroy (start);
  cudaEventDestroy (stop);
  // printf("Kernel 4 time: %3.4f \n", time);

  // kernel5: Compute forces acting on each body with help of octree
  cudaEventCreate (&start);
  cudaEventCreate (&stop);
  cudaEventRecord (start, 0);

  kernel5Compute<<<NUM_BLOCK * 8, 256>>>();

  cudaEventRecord (stop, 0);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&time, start, stop);
  cudaEventDestroy (start);
  cudaEventDestroy (stop);
  // printf("Kernel 5 time: %3.4f \n", time);

  // kernel 6: Update body positions and velocities
  cudaEventCreate (&start);
  cudaEventCreate (&stop);
  cudaEventRecord (start, 0);

  kernel6Update<<<NUM_BLOCK * 8, 256>>>();

  cudaEventRecord (stop, 0);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&time, start, stop);
  cudaEventDestroy (start);
  cudaEventDestroy (stop);
  // printf("Kernel 6 time: %3.4f \n", time);

  // cudaError_t error = cudaPeekAtLastError();
  // if (error != cudaSuccess) {
  //   printLastCudaError(cudaGetErrorString(error));
  //   exit(-1);
  // }

  // GPU to CPU
  cudaMemcpy(point_x, point_x_dev, sizeof(float) * capacity,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(point_y, point_y_dev, sizeof(float) * capacity,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(speed_x, speed_x_dev, sizeof(float) * capacity,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(speed_y, speed_y_dev, sizeof(float) * capacity,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < capacity; i++) {
    m->objects->objects[i].position.x = point_x[i];
    m->objects->objects[i].position.y = point_y[i];
    m->objects->objects[i].speed.x = speed_x[i];
    m->objects->objects[i].speed.y = speed_y[i];
  }

  // remove out of bounds bodies
  spacemodel_remove_objects_outside_bounds(m);
  
  free(mass);
  free(point_x);
  free(point_y);
  free(speed_x);
  free(speed_y);

  cudaFree(len_dev);
  cudaFree(mass_dev);
  cudaFree(point_x_dev);
  cudaFree(point_y_dev);
  cudaFree(speed_x_dev);
  cudaFree(speed_y_dev);
  cudaFree(inorder_rank);
  cudaFree(sort);
  cudaFree(acc_x);
  cudaFree(acc_y);
  cudaFree(node_cnt);
  cudaFree(node_child);
  cudaFree(node_mass);
  cudaFree(node_point_x);
  cudaFree(node_point_y);
}

void gpu_timer_start(cudaEvent_t start, cudaEvent_t stop) {
  cudaEventCreate (&start);
  cudaEventCreate (&stop);
  cudaEventRecord (start, 0);
}

float gpu_timer_stop(cudaEvent_t start, cudaEvent_t stop) {
  float time;
  cudaEventRecord (stop, 0);
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&time, start, stop);
  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  return time;
}
