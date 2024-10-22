#!/bin/bash

CWD=`pwd`
echo "Current working directory: $CWD"

export FI_APP_DIR=/home/yagiz/nvbit_release/tools/nvbitfi
export TEST_APPS_DIR=$FI_APP_DIR/test-apps
rm -rf $FI_APP_DIR/logs
rm $TEST_APPS_DIR/convolution_2D_bench/golden* 
rm $TEST_APPS_DIR/convolution_2D_bench/std*
rm $TEST_APPS_DIR/convolution_2D_bench/convolution_2D_cuda_float_256
cp $CWD/convolution_2D_cuda_float_256 $TEST_APPS_DIR/convolution_2D_bench/convolution_2D_cuda_float_256


rm $TEST_APPS_DIR/max_pooling_bench/golden* 
rm $TEST_APPS_DIR/max_pooling_bench/std*
rm $TEST_APPS_DIR/max_pooling_bench/max_pooling_cuda_float_256
cp $CWD/max_pooling_cuda_float_256 $TEST_APPS_DIR/max_pooling_bench/max_pooling_cuda_float_256


rm $TEST_APPS_DIR/wavelet_transform/golden* 
rm $TEST_APPS_DIR/wavelet_transform/std*
rm $TEST_APPS_DIR/wavelet_transform/wavelet_transform_cuda_float_256
cp $CWD/wavelet_transform_cuda_float_256 $TEST_APPS_DIR/wavelet_transform/wavelet_transform_cuda_float_256


rm $TEST_APPS_DIR/relu_bench/golden* 
rm $TEST_APPS_DIR/relu_bench/std*
rm $TEST_APPS_DIR/relu_bench/relu_cuda_float_256
cp $CWD/relu_cuda_float_256 $TEST_APPS_DIR/relu_bench/relu_cuda_float_256


rm $TEST_APPS_DIR/matrix_multiplication_bench/golden* 
rm $TEST_APPS_DIR/matrix_multiplication_bench/std*
rm $TEST_APPS_DIR/matrix_multiplication_bench/matrix_multiplication_cuda_float_256
cp $CWD/matrix_multiplication_cuda_float_256 $TEST_APPS_DIR/matrix_multiplication_bench/matrix_multiplication_cuda_float_256

cd $FI_APP_DIR
sh test.sh


