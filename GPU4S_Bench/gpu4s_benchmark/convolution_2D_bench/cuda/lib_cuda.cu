#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//#define BLOCK_SIZE 32

int areArraysIdentical(float *arr1, float *arr2, unsigned int N) 
{
    for (unsigned int i = 0; i < N; i++) 
    {
        if (fabs(arr1[i] - arr2[i]) > 1e-4)
        {
            return 0; // Arrays are not identical
        }
    }
    return 1; // Arrays are identical
}

float* findIdenticalArray(float *arr1, float *arr2, float *arr3, int N) 
{
    if (areArraysIdentical(arr1, arr2, N) || areArraysIdentical(arr2, arr3, N)) 
    {
        return arr2; // Return the identical array (arr2 in this case)
    } 
    else if (areArraysIdentical(arr1, arr3, N)) 
    {
        return arr1;
    }
    return NULL; // No identical arrays found
}


__global__ void
covolution_kernel(const bench_t *A, bench_t *B1, bench_t *B2, bench_t *B3, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size)
{
    int size = n;

    unsigned int originalBased = ceil(float(n)/BLOCK_SIZE);
    int x = (blockIdx.x % originalBased) * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int kernel_rad = kernel_size / 2;

    bench_t sum = 0;

    if (x < size && y < size)
    {
        for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3 
            {
                for(int j = -kernel_rad; j <= kernel_rad; ++j){
                    // get value
                    bench_t value = 0;
                    
                    if (i + x < 0 || j + y < 0)
                    {
                        value = 0;
                        //printf("ENTRO %d %d\n", i + x , j + y);
                    }
                    else if ( i + x > size - 1 || j + y > size - 1)
                    {
                        value = 0;
                        //printf("ENTRO UPPER%d %d\n", i + x , j + y);
                    }
                    else
                    {
                        value = A[(x + i)*size+(y + j)];
                    }
                    //printf("ACHIVED position  %d %d value %f\n", (x + i) , (y + j), value);
                    sum += value * kernel[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
                }
            }
    int block_index = blockIdx.x; // Use blockIdx.x within the grid dimension
    
    if (block_index < originalBased) 
    {
        B1[x * size + y] = sum;
    } 
    else if (block_index < (originalBased * 2)) 
    {
        B2[x * size + y] = sum;
    } 
    else 
    {
        B3[x * size + y] = sum;
    }

    }
}

void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}

void init(GraficObject *device_object, int platform ,int device, char* device_name){
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	//printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    device_object->start = new cudaEvent_t;
    device_object->stop = new cudaEvent_t;
    device_object->start_memory_copy_device = new cudaEvent_t;
    device_object->stop_memory_copy_device = new cudaEvent_t;
    device_object->start_memory_copy_host = new cudaEvent_t;
    device_object->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(device_object->start);
    cudaEventCreate(device_object->stop);
    cudaEventCreate(device_object->start_memory_copy_device);
    cudaEventCreate(device_object->stop_memory_copy_device);
    cudaEventCreate(device_object->start_memory_copy_host);
    cudaEventCreate(device_object->stop_memory_copy_host);
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix){
   
   // Allocate the device input vector A
	cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&device_object->d_A, size_a_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device input vector B
    err = cudaMalloc((void **)&device_object->d_B, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->d_B_redundant_1, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->d_B_redundant_2, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device output vector C
    err = cudaMalloc((void **)&device_object->kernel, size_c_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* kernel, unsigned int size_a, unsigned int size_b){
    cudaEventRecord(*device_object->start_memory_copy_device);
	cudaError_t err = cudaMemcpy(device_object->d_A, h_A, sizeof(bench_t) * size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMemcpy(device_object->kernel, kernel, sizeof(bench_t) * size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector kernel from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }



    cudaEventRecord(*device_object->stop_memory_copy_device);
    
}
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(3*ceil(float(n)/BLOCK_SIZE), ceil(float(m)/BLOCK_SIZE));
    cudaEventRecord(*device_object->start);
    covolution_kernel<<<dimGrid, dimBlock>>>(device_object->d_A, device_object->d_B, device_object->d_B_redundant_1, device_object->d_B_redundant_2, device_object->kernel, n, m, w, kernel_size);
    cudaEventRecord(*device_object->stop);
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_B_result, bench_t* h_B_result_redundant_1, bench_t* h_B_result_redundant_2, int size)
{
    cudaEventRecord(*device_object->start_memory_copy_host);
    cudaMemcpy(h_B_result            , device_object->d_B            , size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_result_redundant_1, device_object->d_B_redundant_1, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_result_redundant_2, device_object->d_B_redundant_2, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(*device_object->stop_memory_copy_host);

    float *majorityVotingResult = findIdenticalArray(h_B_result, h_B_result_redundant_1, h_B_result_redundant_2, size);
    if (NULL == majorityVotingResult)
    {
        fprintf(stderr, "None of arrays are identical!\n");
    }
    else
    {
        h_B_result = majorityVotingResult;
    }
}

float get_elapsed_time(GraficObject *device_object, bool csv_format,bool csv_format_timestamp, long int current_time){
    cudaEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    cudaEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time
    cudaEventElapsedTime(&milliseconds, *device_object->start, *device_object->stop);
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    
    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", milliseconds_h_d,milliseconds,milliseconds_d_h, current_time);
    }
    else if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,milliseconds_d_h);
    }else{
         printf("Elapsed time Host->Device: %.10f milliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel: %.10f milliseconds\n", milliseconds);
         printf("Elapsed time Device->Host: %.10f milliseconds\n", milliseconds_d_h);
    }
    return milliseconds;
}

void clean(GraficObject *device_object){
    cudaError_t err = cudaSuccess;
    err = cudaFree(device_object->d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->d_B_redundant_1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->d_B_redundant_2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }


    err = cudaFree(device_object->kernel);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }


    // delete events
    delete device_object->start;
    delete device_object->stop;
    delete device_object->start_memory_copy_device;
    delete device_object->stop_memory_copy_device;
    delete device_object->start_memory_copy_host;
    delete device_object->stop_memory_copy_host;
}
