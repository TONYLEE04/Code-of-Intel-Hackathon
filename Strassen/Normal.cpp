#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <chrono>

int N = 10000;

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


// 方法：生成测试数据
void generateMatrix(float *M){
    for(int i = 0 ; i < N*N ; i++){
        M[i] = rand()%100;
    }
}


//GPU load矩阵相乘
void matrixMul(float* d_A,float* d_B,float* d_C , int n,
               const sycl::nd_item<3> &item_ct1){

    int rowIdx = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                 item_ct1.get_local_id(1);
    int colIdx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                 item_ct1.get_local_id(2);

    for (int k = 0; k < n; k++) {
        d_C[rowIdx * n + colIdx] += d_A[rowIdx * n + k] * d_B[k * n + colIdx];
    }
}

int main() try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    // 确定矩阵大小
    size_t bytes = N * N * sizeof(float);
    // 申请内存空间
    float *h_A;
    float *h_B;
    float *h_C;
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    h_C = (float *)malloc(bytes);

    // 生成随机测试矩阵
    generateMatrix(h_A);
    generateMatrix(h_B);

    // 提交数据到显存
    float *d_A;
    float *d_B;
    float *d_C;
    d_A = (float *)sycl::malloc_device(bytes, q_ct1);
    d_B = (float *)sycl::malloc_device(bytes, q_ct1);
    dpct::err0 e = (d_C = (float *)sycl::malloc_device(bytes, q_ct1), 0);

    q_ct1.memcpy(d_A, h_A, bytes).wait();
    q_ct1.memcpy(d_B, h_B, bytes).wait();

    // 分块
    sycl::range<3> gridSize(1, 1, 1);
    sycl::range<3> blockSize(1, 1, 1);
    gridSize[2] = N / 32; blockSize[2] = 32;
    gridSize[1] = N / 32; blockSize[1] = 32;

    // 计算 并且计时
    auto t1 = high_resolution_clock::now();
    q_ct1.submit([&](sycl::handler &cgh) {
        auto N_ct3 = N;

        cgh.parallel_for(sycl::nd_range<3>(gridSize * blockSize, blockSize),
                         [=](sycl::nd_item<3> item_ct1) {
                             matrixMul(d_A, d_B, d_C, N_ct3, item_ct1);
                         });
    });

    q_ct1.memcpy(h_C, d_C, bytes).wait();

    auto t2 = high_resolution_clock::now();

    // 计算耗时
    auto ms_int = duration_cast<chrono::milliseconds>(t2 - t1);
    cout<<"time cost : "<<ms_int.count()<<endl;
    //print the output matrix
    //for(int i = 0; i < N; i++){
    //for(int j = 0; j < N; j++){
    //    cout << h_C[i * N + j] << " ";
    //}
    //cout << endl;
    //}



    //释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    sycl::free(d_A, q_ct1);
    sycl::free(d_B, q_ct1);
    sycl::free(d_C, q_ct1);

    return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
