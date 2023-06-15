#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>

using namespace std;

// 矩阵大小
int N = 10000;
// 在矩阵较小时，传统算法效率更高，会优先调用传统算法，阈值设为1024
int Limit = 1024;
int BLOCK_SIZE = 32;

// 生成随机矩阵
void generateMatrix(int *M){
    for(int i = 0 ; i < N*N ; i++){
        M[i] = rand()%100;
    }
}


// 矩阵分块
void split(int *X11, int *X12, int *X21, int *X22, int *X, int n,
           const sycl::nd_item<3> &item_ct1) {
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
        if(i < n && j < n) {
		X11[i * n + j] = X[i * 2 * n + j];
		X12[i * n + j] = X[i * 2 * n + j + n];
		X21[i * n + j] = X[(i + n) * 2 * n + j];
		X22[i * n + j] = X[(i + n) * 2 * n + j + n];
	}
}


// 矩阵A B相加，结果放入C
void add(int *A, int *B, int *C, int n, const sycl::nd_item<3> &item_ct1) {
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
        if(i < n && j < n) {
		C[i * n + j] = A[i * n + j] + B[i * n + j];
	}
}


// 求和
void sub(int *A, int *B, int *C, int n, const sycl::nd_item<3> &item_ct1) {
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
        if(i < n && j < n) {
		C[i * n + j] = A[i * n + j] - B[i * n + j];
	}
}


// 相乘
void mul(int *A, int *B, int *C, int n, const sycl::nd_item<3> &item_ct1) {
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
        if(i < n && j < n) {
		C[i * n + j] = 0;
		for(int k = 0; k < n; k++) {
			C[i * n + j] += A[i * n + k] * B[k * n + j];
		}
	}
}


// 合并矩阵
void merge(int *C11, int *C12, int *C21, int *C22, int *C, int n,
           const sycl::nd_item<3> &item_ct1) {
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);
        if(i < n && j < n) {
		C[i * 2 * n + j] = C11[i * n + j];
		C[i * 2 * n + j + n] = C12[i * n + j];
		C[(i + n) *2 * n + j] = C21[i * n + j];
		C[(i + n) * 2 * n + j + n] = C22[i * n + j];
	}
}


// Sterassen主方法
void sterassen(int *A, int *B, int *C, int n) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        sycl::range<3> block(1, BLOCK_SIZE, BLOCK_SIZE);

        int *A_gpu, *B_gpu, *C_gpu;
        A_gpu = (int *)sycl::malloc_device(sizeof(int) * n * n, q_ct1);
        B_gpu = (int *)sycl::malloc_device(sizeof(int) * n * n, q_ct1);
        C_gpu = (int *)sycl::malloc_device(sizeof(int) * n * n, q_ct1);
        q_ct1.memcpy(A_gpu, A, sizeof(int) * n * n).wait();
        q_ct1.memcpy(B_gpu, B, sizeof(int) * n * n).wait();

        // 阈值以内时，调用传统矩阵乘法
	if (n <= Limit){
                sycl::range<3> grid(1, (size_t)ceil((int)n / (int)block[1]),
                                    (size_t)ceil((int)n / (int)block[2]));
                /*
		DPCT1049:21: The work-group size passed to the
                 * SYCL kernel may exceed the limit. To get the device limit,
                 * query info::device::max_work_group_size. Adjust the
                 * work-group size if needed.
		*/
                q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                                   [=](sycl::nd_item<3> item_ct1) {
                                           mul(A_gpu, B_gpu, C_gpu, n,
                                               item_ct1);
                                   });
                dev_ct1.queues_wait_and_throw();
                q_ct1.memcpy(C, C_gpu, sizeof(int) * n * n).wait();
                sycl::free(A_gpu, q_ct1);
                sycl::free(B_gpu, q_ct1);
                sycl::free(C_gpu, q_ct1);
                return;
	}
	
	// 数据分配
	int *A11, *A12, *A21, *A22, *B11, *B12, *B21, *B22, *C11, *C12, *C21, *C22, *M1, *M2 ,*M3 , *M4 , *M5 , *M6 , *M7 , *T1 , *T2;
	int m = n / 2 ;
        A11 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        A12 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        A21 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        A22 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        B11 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        B12 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        B21 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        B22 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        C11 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        C12 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        C21 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        C22 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        M1 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        M2 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        M3 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        M4 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        M5 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        M6 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        M7 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        T1 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        T2 = (int *)sycl::malloc_device(sizeof(int) * m * m, q_ct1);
        sycl::range<3> grid(1, (size_t)ceil((int)m / (int)block[1]),
                            (size_t)ceil((int)m / (int)block[2]));
        // 矩阵分块
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   split(A11, A12, A21, A22, A_gpu, m,
                                         item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        //split the matrix B to 4 parts
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   split(B11, B12, B21, B22, B_gpu, m,
                                         item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();

        //M1
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(B12, B22, T1, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        sterassen(A11 , T1 ,M1 , m);
        dev_ct1.queues_wait_and_throw();
        
        //M2
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(A11, A12, T1, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        sterassen(T1 , B22 , M2 , m);
        dev_ct1.queues_wait_and_throw();

        //M3
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(A21, A22, T1, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        sterassen(T1 , B11, M3 , m);
        dev_ct1.queues_wait_and_throw();
        
        //M4
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   sub(B21, B11, T1, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        sterassen(A22 , T1 , M4 , m );
        dev_ct1.queues_wait_and_throw();
        
        //M5
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(A11, A22, T1, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();

        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(B11, B22, T2, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        sterassen(T1 , T2 , M5 , m);
        dev_ct1.queues_wait_and_throw();
        
        //M6
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   sub(A12, A22, T1, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();

        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(B21, B22, T2, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        sterassen(T1 , T2 , M6 , m);
        dev_ct1.queues_wait_and_throw();
        
        //M7
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   sub(A11, A21, T1, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(B11, B12, T2, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        sterassen(T1 , T2 , M7 , m);
        dev_ct1.queues_wait_and_throw();

        //C11
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(M5, M4, T1, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();

        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   sub(T1, M2, T2, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(T2, M6, C11, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();

        //C12
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(M1, M2, C12, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();

        //C21
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(M3, M4, C21, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();

        //C22
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   add(M5, M1, T1, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();

        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   sub(T1, M3, T2, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();
        
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   sub(T2, M7, C22, m, item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();

        //合并 C11 , C12 , C21 , C22
        q_ct1.parallel_for(sycl::nd_range<3>(grid * block, block),
                           [=](sycl::nd_item<3> item_ct1) {
                                   merge(C11, C12, C21, C22, C_gpu, m,
                                         item_ct1);
                           });
        dev_ct1.queues_wait_and_throw();

        q_ct1.memcpy(C, C_gpu, sizeof(int) * n * n).wait();

        //释放内存
        sycl::free(A11, q_ct1);
        sycl::free(A12, q_ct1);
        sycl::free(A21, q_ct1);
        sycl::free(A22, q_ct1);
        sycl::free(B11, q_ct1);
        sycl::free(B12, q_ct1);
        sycl::free(B21, q_ct1);
        sycl::free(B22, q_ct1);
        sycl::free(T1, q_ct1);
        sycl::free(T2, q_ct1);
        sycl::free(M1, q_ct1);
        sycl::free(M2, q_ct1);
        sycl::free(M3, q_ct1);
        sycl::free(M4, q_ct1);
        sycl::free(M5, q_ct1);
        sycl::free(M6, q_ct1);
        sycl::free(M7, q_ct1);
        sycl::free(A_gpu, q_ct1);
        sycl::free(B_gpu, q_ct1);
        sycl::free(C_gpu, q_ct1);
}

int main(){

    size_t bytes = N * N * sizeof(int);
    // 分配内存
    int *h_A;
    int *h_B;
    int *h_C;
    h_A = (int *)malloc(bytes);
    h_B = (int *)malloc(bytes);
    h_C = (int *)malloc(bytes);

    // 生成随机矩阵
    generateMatrix(h_A);
    generateMatrix(h_B);

    //运算并计时
    auto t1 = std::chrono::high_resolution_clock::now();
	sterassen(h_A,h_B,h_C,N);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto ms_int = duration_cast<chrono::milliseconds>(t2 - t1);
    cout<<"time cost : "<<ms_int.count()<<endl;

}
