script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf ${script_dir}/.git/hooks/pre-commit
rm -rf ${script_dir}/build && mkdir ${script_dir}/build && cd ${script_dir}/build

clear
unset do_validation
unset epilogue
unset IGC_VISAOptions
unset SYCL_PROGRAM_COMPILE_OPTIONS
unset sycl_compiler_path
unset gpu_driver_path
unset enable_prefetch
unset divide_b
unset disable_gemm

#do_validation=" -DDO_VALIDATION "
#do_validation=" -DDO_VALIDATION -DCOPY_DEBUG"

#disable_gemm=" -DDISABLE_GEMM "
enable_prefetch=" -DENABLE_PREFETCH "
#epilogue+=" -DEPILOGUE_SOFTMAX "
divide_b=" -DDIV_B=1 -DR_I_CNT=1 "

# https://github.com/intel/llvm/releases/tag/nightly-2024-05-16
# https://ubit-gfx.intel.com/build/19168301/artifacts
sycl_compiler_path=/opt/cutlass/compiler/1008/
gpu_driver_path=/opt/cutlass/gpu_driver/gfx-driver-ci-comp_igc-27004/extract/

# AOT compile
output=intel_gpu_pvc
# jit compile
#output=spir64



export ZE_AFFINITY_MASK=0
export CPATH=$sycl_compiler_path:$sycl_compiler_path/include/:$sycl_compiler_path/include/sycl/
export LIBRARY_PATH=$gpu_driver_path/usr/lib/x86_64-linux-gnu/:$sycl_compiler_path/lib/
export LD_LIBRARY_PATH=$LIBRARY_PATH
export IGC_EnableVISANoSchedule=1
export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=${script_dir}/build/mm_dumps
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export OCL_ICD_VENDORS=$HOME
#export IGC_VISAOptions="-newspillcost "

#export IGC_VectorAliasBBThreshold=1500
#export IGC_VISAOptions="-perfmodel"

#export SYCL_PROGRAM_COMPILE_OPTIONS=" -vc-codegen -vc-disable-indvars-opt -doubleGRF -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' "

#target=./test/unit/cute/intel_xe/cutlass_test_unit_cute_intel_xe
#target=./examples/sycl/pvc/flash_attention_v2/pvc_flash_attention
#target=./examples/sycl/pvc/pvc_gemm
#target=./examples/sycl/pvc/pvc_gemm_with_epilogue_relu
#target=./examples/sycl/pvc/pvc_gemm_with_epilogue_gelu
target=./examples/sycl/pvc/pvc_gemm_with_epilogue_lincombdeeltact
#target=./examples/sycl/pvc/pvc_gemm_with_per_row_bias
#target=./examples/sycl/pvc/pvc_collective_builder
#target=./examples/sycl/pvc/pvc_gemm_streamk

clear

cmake .. -G Ninja -DCMAKE_CUDA_HOST_COMPILER=${sycl_compiler_path}/bin/clang++ \
-DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$output -DCMAKE_CXX_COMPILER=${sycl_compiler_path}/bin/clang++ \
-DCMAKE_CXX_FLAGS=" -ftarget-register-alloc-mode=pvc:large -DPREFETCH_DEFAULT -DSYCL_INTEL_TARGET ${disable_gemm} ${epilogue} ${do_validation} ${enable_prefetch} ${divide_b} -gline-tables-only " \
&& ninja -v $target && $target
# --batch=1 --num_heads=1 --iterations=1 --m=1024 --n=2048 --k=4096

# -gline-tables-only

# unitrace command
#unitrace -k -i 20 --chrome-kernel-logging -o cutlass_pvc_gemm.csv $target

#print kernel calls
#unitrace --device-timing --kernel-submission --device-timeline --chrome-kernel-logging --chrome-device-logging --chrome-no-thread-on-device --chrome-no-engine-on-device -i 20 $target -o cutlass.csv
#unitrace -k --chrome-kernel-logging --chrome-device-logging --chrome-no-thread-on-device --chrome-no-engine-on-device -i 20 $target -o cutlass.csv

#check stalls
#unitrace --chrome-kernel-logging --stall-sampling -i 20 -o cutlass_pvc_gemm.csv $target

#unitrace --metric-list
#unitrace -k -g <group> -i 20 --chrome-kernel-logging $target -o

#csv_file=cutlass_pvc_gemm.metrics.3584319.csv
#python3 ~/workspace/cutlass/unitrace/tools/unitrace/scripts/analyzeperfmetrics.py -l $csv_file
#python3 ~/workspace/cutlass/unitrace/tools/unitrace/scripts/analyzeperfmetrics.py -m "XVE_STALL[%],XVE_THREADS_OCCUPANCY_ALL[%],XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION[%],XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION[%],XVE_INST_EXECUTED_SEND_ALL_UTILIZATION[%],XVE_INST_EXECUTED_CONTROL_ALL_UTILIZATION[%],XVE_INST_EXECUTED_XMX_ALL_UTILIZATION[%]" -y "Occupancy, Stalls and Function Unit Utilizations" -m "AvgGpuSliceFrequencyMHz[MHz]" -y "Frequency" -m "L3_BYTE_READ[bytes],L3_BYTE_WRITE[bytes],GPU_MEMORY_BYTE_READ[bytes],GPU_MEMORY_BYTE_WRITE[bytes]" -y "L3 and Memory" -m "XVE_ACTIVE[%],XVE_STALL[%]" -y "Active and Stalls" -b "L3_BYTE_READ[bytes],L3_BYTE_WRITE[bytes],GPU_MEMORY_BYTE_READ[bytes],GPU_MEMORY_BYTE_WRITE[bytes]" -t "Hardware Metrics" ${csv_file}
#python3 ~/workspace/cutlass/unitrace/tools/unitrace/scripts/analyzeperfmetrics.py -s $IGC_DumpToCustomDir -t "XVE Stalls by Instruction" $csv_file -o ${csv_file}.pdf
#python3 ~/workspace/cutlass/unitrace/tools/unitrace/scripts/analyzeperfmetrics.py -k "main::{lambda(auto:1)#3}" -s $IGC_DumpToCustomDir $csv_file -o ${csv_file}.pdf



