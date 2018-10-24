#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "CL\cl.h"

char* file_to_string(char* filename) {
    FILE* f = fopen(filename, "rb");
	if (f) {
		fseek(f, 0L, SEEK_END);
		unsigned int length = ftell(f);
		fseek(f, 0L, SEEK_SET);
		char* string = (char*)malloc(sizeof(char)*(length+1));
		if (string)
			fread(string, 1, length, f);
		else return NULL;
		fclose(f);
		string[length] = '\0';
		 return string;
	}
	else return NULL;
}

void handle_error(cl_int error_code, char* s) {
	if(CL_SUCCESS != error_code) {
		printf(s, error_code);
		exit(EXIT_FAILURE);
	}
}

void handle_program_build_errors(cl_int error_code, cl_program program, cl_device_id device) {
	if (CL_SUCCESS != error_code) {
		char* build_log;
		size_t n_bytes;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &n_bytes);
	    build_log = (char*)malloc(sizeof(char)*n_bytes);
		if (build_log) {
			cl_int new_error_code = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(char)*n_bytes, build_log, &n_bytes);
			printf("%s", build_log);
			free(build_log);
			handle_error(new_error_code, "clGetProgramBuildInfo(...) failed with error code %d.\n");
		}
		else printf("Out of memory.\n");
	}
}

int main(int argc, char** argv) {
	char*            kernel_source = file_to_string("CalculatePi.cl");
	cl_int           error_code;
	cl_platform_id   platform;
	char             platform_name[128];
	cl_device_id     device;
	char             device_name[128];
	cl_context       context;
	cl_program       program;
	cl_kernel        kernel;
	cl_command_queue command_queue;
	size_t           workgroup_size;
	cl_int*          output;
	cl_mem           output_on_device;
	cl_uint          N; //Number of samples per kernel.
	cl_uint          approximate_number_of_total_samples = 1000000000;
	cl_uint          total_samples;

	if (NULL == kernel_source) {
		printf("Could not read kernel source file \"CalculatePi.cl\".\n");
		return EXIT_FAILURE;
	}

	error_code = clGetPlatformIDs(1, &platform, NULL);
	handle_error(error_code, "clGetPlatformIDs(...) failed with error code %d.\n");

	error_code = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
	handle_error(error_code, "clGetPlatformInfo(...) failed with error code %d.\n");
	printf("Using platform: %s.\n", platform_name);

	error_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	handle_error(error_code, "clGetDeviceIDs(...) failed with error code %d.\n");

	error_code = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
	handle_error(error_code, "clGetDeviceInfo(...) failed with error code %d.\n");
	printf("Using OpenCL device: %s.\n", device_name);

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &error_code);
	handle_error(error_code, "clCreateContext(...) failed with error code %d.\n");

	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, &error_code);
	handle_error(error_code, "clCreateProgramWithSource(...) failed with error code %d.\n");

	error_code = clBuildProgram(program, 1, (const cl_device_id*)&device, NULL, NULL, NULL);
	handle_program_build_errors(error_code, program, device);
	
	/*char* what = (char*)malloc(sizeof(char) * 10000); size_t number;
	clGetProgramInfo(program, CL_PROGRAM_SOURCE, 10000, (void*)what, &number);
	printf("%d %d %s", number, strlen(kernel_source), what);*/
	
	const char* kernel_name = "count_points";
	kernel = clCreateKernel(program, kernel_name, &error_code);
	handle_error(error_code, "clCreateKernel(...) failed with error code %d.\n");

	error_code = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
	handle_error(error_code, "clGetKernelWorkGroupInfo(...) failed with error code %d.\n");

	N = approximate_number_of_total_samples / workgroup_size;
	total_samples = N * workgroup_size;

	output = (cl_int*)malloc(sizeof(cl_int)*workgroup_size);
	if (!output) {
		printf("Out of memory.\n");
		exit(EXIT_FAILURE);
	}
	
	output_on_device = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int)*workgroup_size, NULL, &error_code);
	handle_error(error_code, "clCreateBuffer(...) failed with error code %d.\n");
	
	error_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &output_on_device);
	handle_error(error_code, "clSetKernelArg(kernel, 0, ...) failed with error code %d.\n");

	error_code = clSetKernelArg(kernel, 1, sizeof(cl_uint), &N);
	handle_error(error_code, "clSetKernelArg(kernel, 1, ...) failed with error code %d.\n");

	command_queue = clCreateCommandQueueWithProperties(context, device, NULL, &error_code);
	handle_error(error_code, "clCreateCommandQueueWithProperties(...) failed with error code %d.\n");

	struct timespec start, finish;
	timespec_get(&start, TIME_UTC);

	size_t local_item_size[] = { 1 };
	error_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, 0, (const size_t*)&workgroup_size, local_item_size, 0, NULL, NULL);
	handle_error(error_code, "clEnqueueNDRangeKernel(...) failed with error code %d.\n");

	error_code = clFinish(command_queue);
	handle_error(error_code, "clFinish(...) failed with error code %d.\n");

	error_code = clEnqueueReadBuffer(command_queue, output_on_device, CL_TRUE, 0, workgroup_size * sizeof(cl_int), output, 0, NULL, NULL);
	handle_error(error_code, "clEnqueueReadBuffer(...) failed with error code %d.\n");

	//output = (cl_int*)clEnqueueMapBuffer(command_queue, output_on_device, CL_TRUE, CL_MAP_READ, 0, workgroup_size*sizeof(cl_int), NULL, NULL, NULL, NULL);

	error_code = clFinish(command_queue);
	handle_error(error_code, "clFinish(...) failed with error code %d.\n");

	timespec_get(&finish, TIME_UTC);
	printf("Computation and reading of buffer finished in %fs.\n", (double)(finish.tv_sec - start.tv_sec) + (double)(finish.tv_nsec - start.tv_nsec)/1000000000.0);

	cl_int sum = 0;
	for (int i = 0; i != workgroup_size; ++i)
		sum += output[i];
	double fraction_in_circle = ((double)sum) / ((double)total_samples),
		pi = 3*sqrt(3)/2 + fraction_in_circle * 6.0 * (1.0 - sqrt(3)/2);

	printf("pi approximation:\t%f\npi:\t\t\t%f", pi, 3.141592265359);

	error_code = clReleaseMemObject(output_on_device);
	handle_error(error_code, "clReleaseMemObject(...) failed with error code %d.\n");
	
	free(output);
	
	error_code = clReleaseCommandQueue(command_queue);
	handle_error(error_code, "clReleaseCommandQueue(...) failed with error code %d.\n");
	
	error_code = clReleaseKernel(kernel);
	handle_error(error_code, "clReleaseKernel(...) failed with error code %d.\n");
	
	error_code = clReleaseProgram(program);
	handle_error(error_code, "clReleaseProgram(...) failed with error code %d.\n");
	
	error_code = clReleaseContext(context);
	handle_error(error_code, "clReleaseContext(...) failed with error code %d.\n");
	
	free(kernel_source);
}