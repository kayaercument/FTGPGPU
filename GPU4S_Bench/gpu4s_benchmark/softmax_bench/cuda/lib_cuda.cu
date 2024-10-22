#include "../benchmark_library.h"
#include "math.h"

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
            printf("arr1[%d]: %f, arr2[%d]: %f\n", i, arr1[i], i, arr2[i]);
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
softmax_kernel(const bench_t *A, bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        #ifdef INT
        B[i*size+j] = exp(A[i*size+j]);
        #elif FLOAT
        B[i*size+j] = expf(A[i*size+j]);
        #else
        B[i*size+j] = exp(A[i*size+j]);
        #endif
        atomicAdd(sum_d_B, B[i*size+j]);
    }
}
__global__ void
softmax_finish_kernel(bench_t *B, bench_t *sum_d_B,const int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        B[i*size+j] = (B[i*size+j]/(*sum_d_B));
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


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix){

   // Allocate the device input vector A
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&device_object->d_A, size_a_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc d_A (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc((void **)&device_object->d_A_redundant_1, size_a_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc d_A_redundant_1 (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc((void **)&device_object->d_A_redundant_2, size_a_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc d_A_redundant_2 (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    // Allocate the device input vector B
    err = cudaMalloc((void **)&device_object->d_B, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc d_B (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc((void **)&device_object->d_B_redundant_1, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc d_B_redundant_1 (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc((void **)&device_object->d_B_redundant_2, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc d_B_redundant_2 (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }
    // Allocate the device input sum_d_B
    err = cudaMalloc((void **)&device_object->sum_d_B, sizeof(bench_t));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc sum_d_B (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc((void **)&device_object->sum_d_B_redundant_1, sizeof(bench_t));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc sum_d_B_redundant_1 (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc((void **)&device_object->sum_d_B_redundant_2, sizeof(bench_t));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc sum_d_B_redundant_2 (error code %s)!\n", cudaGetErrorString(err));
        return false;
    }


    return true;

}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a){
    cudaEventRecord(*device_object->start_memory_copy_device);
    cudaError_t err = cudaMemcpy(device_object->d_A, h_A, sizeof(bench_t) * size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(device_object->d_A_redundant_1, h_A, sizeof(bench_t) * size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A (redundant_1)from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(device_object->d_A_redundant_2, h_A, sizeof(bench_t) * size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A (redundant_2) from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    cudaEventRecord(*device_object->stop_memory_copy_device);   
}
void execute_kernel(GraficObject *device_object, 
                    bench_t* h_B_result, bench_t* h_B_result_redundant_1, bench_t* h_B_result_redundant_2, 
                    bench_t* sum_h_B_result, bench_t* sum_h_B_result_redundant_1, bench_t* sum_h_B_result_redundant_2,
                    unsigned int n, unsigned int m,unsigned int w)
{
    cudaError_t err;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x), ceil(float(m)/dimBlock.y));
    cudaEventRecord(*device_object->start);
    softmax_kernel<<<dimGrid, dimBlock>>>(device_object->d_A            , device_object->d_B            , device_object->sum_d_B            , n);
    softmax_kernel<<<dimGrid, dimBlock>>>(device_object->d_A_redundant_1, device_object->d_B_redundant_1, device_object->sum_d_B_redundant_1, n);
    softmax_kernel<<<dimGrid, dimBlock>>>(device_object->d_A_redundant_2, device_object->d_B_redundant_2, device_object->sum_d_B_redundant_2, n);
    cudaEventRecord(*device_object->stop);

    // =================================================================================================================
    // Comparison will be done on CPU.
    // Get all redundant instances from d_B result from device to host
    // Apply majorityVoting, then copy 3 different instances from host to device to be used on further calculations
    err = cudaMemcpy(h_B_result            , device_object->d_B            , n * sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy d_B from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(h_B_result_redundant_1, device_object->d_B_redundant_1, n * sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy d_B_redundant_1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(h_B_result_redundant_2, device_object->d_B_redundant_2, n * sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy d_B_redundant_2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    
    float *majorityVotingResult = findIdenticalArray(h_B_result, h_B_result_redundant_1, h_B_result_redundant_2, n);
    if (NULL == majorityVotingResult)
    {
        fprintf(stderr, "h_B comparison: None of arrays are identical!\n");
    }
    else
    {
        h_B_result = majorityVotingResult;
    }
    // TODO: instead of copying all 3 GPU instances again, copy just "broken" one which is detected during majorityVoting step.
    err = cudaMemcpy(device_object->d_B            , h_B_result, n * sizeof(bench_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(device_object->d_B_redundant_1, h_B_result, n * sizeof(bench_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_B_redundant_1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(device_object->d_B_redundant_2, h_B_result, n * sizeof(bench_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_B_redundant_2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // =================================================================================================================
    // Second step: comparison of second output

    // Comparison will be done on CPU.
    // Get all redundant instances from sum_d_B result from device to host
    // Apply majorityVoting, then copy 3 different instances from host to device to be used on further calculations

    err = cudaMemcpy(sum_h_B_result            , device_object->sum_d_B            , sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy sum_d_B from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(sum_h_B_result_redundant_1, device_object->sum_d_B_redundant_1, sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy sum_d_B_redundant_1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(sum_h_B_result_redundant_2, device_object->sum_d_B_redundant_2, sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy sum_d_B_redundant_2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }


    // Only 1 item we have
    majorityVotingResult = findIdenticalArray(sum_h_B_result, sum_h_B_result_redundant_1, sum_h_B_result_redundant_2, 1);
    if (NULL == majorityVotingResult)
    {
        fprintf(stderr, "sum_h_B comparison: None of arrays are identical!\n");
    }
    else
    {
        device_object->sum_d_B = majorityVotingResult;
    }
    
    err = cudaMemcpy(device_object->sum_d_B            , sum_h_B_result, sizeof(bench_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy sum_d_B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(device_object->sum_d_B_redundant_1, sum_h_B_result, sizeof(bench_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy sum_d_B_redundant_1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(device_object->sum_d_B_redundant_2, sum_h_B_result, sizeof(bench_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy sum_d_B_redundant_2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }


    softmax_finish_kernel<<<dimGrid, dimBlock>>>(device_object->d_B            , device_object->sum_d_B            , n);
    softmax_finish_kernel<<<dimGrid, dimBlock>>>(device_object->d_B_redundant_1, device_object->sum_d_B_redundant_1, n);
    softmax_finish_kernel<<<dimGrid, dimBlock>>>(device_object->d_B_redundant_2, device_object->sum_d_B_redundant_2, n);
    
    // Overall result is d_B and its redundant instances

    // Comparison will be done on CPU.
    // Get all redundant instances from d_B result from device to host
    // Apply majorityVoting, then copy 3 different instances from host to device to be used on further calculations
    err = cudaMemcpy(h_B_result            , device_object->d_B            , n * sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_B from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(h_B_result_redundant_1, device_object->d_B_redundant_1, n * sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_B_redundant_1 from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMemcpy(h_B_result_redundant_2, device_object->d_B_redundant_2, n * sizeof(bench_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy h_B_redundant_2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }


    
    majorityVotingResult = findIdenticalArray(h_B_result, h_B_result_redundant_1, h_B_result_redundant_2, n);
    if (NULL == majorityVotingResult)
    {
        fprintf(stderr, "softmax_finish_kernel result: None of arrays are identical!\n");
    }
    else
    {
        device_object->d_B = majorityVotingResult;
    }
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){
    cudaEventRecord(*device_object->start_memory_copy_host);
    cudaMemcpy(h_C, device_object->d_B, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(*device_object->stop_memory_copy_host);
    }

float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time){
    cudaEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    cudaEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time
    cudaEventElapsedTime(&milliseconds, *device_object->start, *device_object->stop);
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", milliseconds_h_d,milliseconds,milliseconds_d_h,current_time);
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
        fprintf(stderr, "Failed to free device vector B redundant_1 (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->d_B_redundant_2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B redundant_2 (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->sum_d_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device  sum_d_B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->sum_d_B_redundant_1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device  sum_d_B_redundant_1 (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->sum_d_B_redundant_2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device  sum_d_B_redundant_2 (error code %s)!\n", cudaGetErrorString(err));
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
