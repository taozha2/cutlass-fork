#!/bin/bash

KERNEL="PvcGemmBF16BF16FP32_RRR_999"
#KERNEL="PvcGemmBF16BF16FP32_RRR_3"
#KERNEL="PvcGemmBF16BF16FP32_RCR_6"
#KERNEL="PvcGemmBF16BF16FP32_RRR_10"
#KERNEL="PvcGemmBF16BF16FP32_RRR_11"
WORKDIR=`date +%Y%m%d%I%M`"_"$KERNEL

SHAPE_LIST_FILE=./$WORKDIR/gemm_shapes.csv
TEST_REPORT=./$WORKDIR/gemm_shapes_report.csv
RUN_LOG=./$WORKDIR/gemm_shapes_run.log
INPUT_IN=./$WORKDIR/input.in
BENCHMARK=./build/benchmarks/gemm/cutlass_benchmarks_gemm_sycl
BM_NAME=" --bm_name=bf16_bf16_fp32"
DEBUG="Yes"
#DEBUG="No"

L="1"
M=""
K=""
N=""

# Env
. /opt/intel/oneapi/setvars.sh

export CC=icx
export CXX=icpx

#export LD_LIBRARY_PATH=$LIBRARY_PATH

#export ZE_AFFINITY_MASK=0
#export CUTLASS_ENABLE_SYCL=ON
# ON is GPU time
#export CUTLASS_SYCL_PROFILING_ENABLED=ON
# OFF is wall time
#export CUTLASS_SYCL_PROFILING_ENABLED=OFF
#export DPCPP_SYCL_TARGET=intel_gpu_bmg_g21
#export CUTLASS_ENABLE_BENCHMARKS=ON

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export CMAKE_BUILD_TYPE=Release
export IGC_VISAOptions="-perfmodel"
export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"

debug_print() {
    if [ $DEBUG = "Yes" ]; then
        echo "$@"
    fi
}

echo "mkdir $WORKDIR"
rm -rf $WORKDIR
mkdir $WORKDIR

echo "Convert gemm_shaps.csv shapes to benchmark input.in format."
cp gemm_shapes.csv $WORKDIR
rm -rf $INPUT_IN
shapelist=`cat $SHAPE_LIST_FILE`
for shape in $shapelist
do
    debug_print "shape is: $shape"

    # Not shape data
    echo $shape | grep "M,"  >/dev/zero   
    if [ $? = 0 ]; then
        continue
    fi

    echo $shape | grep "Total"  >/dev/zero   
    if [ $? = 0 ]; then
        continue
    fi

    M=`echo $shape |awk -F , '{print $1}'`
    K=`echo $shape |awk -F , '{print $2}'`
    N=`echo $shape |awk -F , '{print $3}'`

    debug_print "M: $M, K: $K, N: $N"
   
    # skip ; lines 
    if [ ${M:0:1} = ";" ]; then
        debug_print "Commented line, $shape"
        continue
    fi

# Generate tmp_input.in
# PvcGemmBF16BF16FP32_RRR_1 --bm_name=bf16_bf16_fp32 --l=1 --m=4096 --k=4096 --n=4096 
    SHAPE_INPUT="$KERNEL $BM_NAME --l=$L --m=$M --k=$K --n=$N"
    debug_print $SHAPE_INPUT
    
    echo $SHAPE_INPUT >>$INPUT_IN
done

# Remove ^M from the file.
dos2unix $INPUT_IN

# Use Benchmark run input.in
echo $BENCHMARK --config_file=$INPUT_IN 
$BENCHMARK --config_file=$INPUT_IN |tee $RUN_LOG

echo "Update gemm_shaps_report.csv"
# cp gemm_shapes_run.log $WORKDIR
# Read gemm_shaps.csv
echo "M,K,N,Result,Tflops,HBM">$TEST_REPORT
#read logs in $RUN_LOG
while IFS= read -r runlog; do
    debug_print "runlog is: $runlog"
    #Pass report:
    #PvcGemmBF16BF16FP2_RRR_1/bf16_bf16_fp32/1x2x768x1/manual_time              0.050 ms        0.051 ms        13931 alpha=1 avg_runtime_ms=0.0503974 avg_tflops=60.9556u avg_throughput=0.0915921 best_bandwidth=0.12271 best_runtime_ms=0.037617 b    est_tflop=81.6652u beta=0 k=768 l=1 m=1 n=2 total_runtime_ms=702.086 layoutA=RowMajor layoutB=RowMajor layoutC=RowMajor"
    #Fail report:
    #PvcGemmBF16BF16FP32_RRR_1/bf16_bf16_fp32/2x768x1x1/manual_time         ERROR OCCURRED: 'Disposition Failed.'

    echo $runlog | grep "$KERNEL" >/dev/zero
    #Not report data
    if [ $? != 0 ]; then
        continue
    fi

    M=`echo $runlog |awk -F " " '{print $1}' |awk -F "/" '{print $3}'|awk -F "x" '{print $1}'`
    N=`echo $runlog |awk -F " " '{print $1}' |awk -F "/" '{print $3}'|awk -F "x" '{print $2}'`
    K=`echo $runlog |awk -F " " '{print $1}' |awk -F "/" '{print $3}'|awk -F "x" '{print $3}'`
    L=`echo $runlog |awk -F " " '{print $1}' |awk -F "/" '{print $3}'|awk -F "x" '{print $4}'`

    debug_print "M: $M, K: $K, N: $N, L: $L"

    echo $runlog | grep "avg_tflops">/dev/zero  
    if [ $? = 0 ]; then
        RESULT="Pass"
        avg_tflops=`echo $runlog |awk -F " " '{print $9}' |awk -F "=" '{print $2}'`
        avg_throughput=`echo $runlog |awk -F " " '{print $10}' |awk -F "=" '{print $2}'`
    else
        RESULT="Fail"
        avg_tflops=""
    fi
    REPORT="$M,$K,$N,$RESULT,$avg_tflops, $avg_throughput"
    debug_print $REPORT
    
    echo $REPORT >>$TEST_REPORT
done < $RUN_LOG

Pass=`cat $TEST_REPORT |grep Pass |wc -l`
Fail=`cat $TEST_REPORT |grep Fail |wc -l`
Total=`expr $Pass + $Fail`

echo "Total: $Total, Pass: $Pass, Fail: $Fail" 

echo "Total: $Total, Pass: $Pass, Fail: $Fail"  >>$TEST_REPORT
