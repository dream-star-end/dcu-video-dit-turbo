#pragma once
#include <iostream>
#include <cstring>
#include <hip/hip_runtime.h>


__attribute__((weak)) int getArch() {
    hipDeviceProp_t props;
    auto hipResult = hipGetDeviceProperties(&props, 0);
    std::string gcn_arch_name(props.gcnArchName);
    gcn_arch_name = gcn_arch_name.substr(3, 3);
    if (gcn_arch_name == "92a") gcn_arch_name = "930";
    int gcn_arch = std::stoi(gcn_arch_name);
    return gcn_arch;
}


enum class FAFUNC {
    FORWARD,
    BACKWARD,
    KVCACHE
};

// 内部静态变量的懒汉实现 //
template<typename Kernel_traits, FAFUNC Func, bool MLS_Enabled=false>
class DeviceProperties {
public:
    int gcn_arch;
    int cu_count;
    size_t lds_size;
    // 获取单实例对象
    static DeviceProperties& GetInstance() {
        static DeviceProperties instance; // 内部静态变量实现单例
        return instance;
    }

private:
    // 禁止外部构造
    DeviceProperties()  { // 可以在这里给内部变量赋初始值
        hipDeviceProp_t props;
        auto hipResult = hipGetDeviceProperties(&props, 0);
        this->cu_count = props.multiProcessorCount;
        this->gcn_arch = getArch();

        const char* fa_debug = std::getenv("FA_DEBUG");
        bool do_fa_debug = fa_debug != nullptr;

        if constexpr (Func == FAFUNC::FORWARD) {
            const size_t least_required_size = ((Kernel_traits::kHeadDim == 192) && (Kernel_traits::kHeadDimV == 192)) ? (21 * 1024) : Kernel_traits::STAGES * Kernel_traits::kNWarps * sizeof(typename Kernel_traits::Element) * 32 * 32;
            const bool run_new_mls = gcn_arch >= 938 and MLS_Enabled;
            const size_t q_smem_size = run_new_mls ? least_required_size: Kernel_traits::q_smem_size;
            const size_t k_smem_size = run_new_mls ? least_required_size: Kernel_traits::k_smem_size * 2;
            const size_t v_smem_size = run_new_mls ? least_required_size: Kernel_traits::v_smem_size * 2;
            if (gcn_arch == 928 or gcn_arch == 936 or gcn_arch == 938 or gcn_arch == 946) {
                this->lds_size = run_new_mls ? std::max(q_smem_size, std::max(v_smem_size, k_smem_size)): std::max(q_smem_size, v_smem_size + k_smem_size);
            }
            else if (gcn_arch == 930) {
                this->lds_size = 32 * 1024;
            }
            if (do_fa_debug and std::strcmp(fa_debug, "2")) {
                printf("gcn_arch: %d\nq_smem_size: %ld\nk_smem_size: %ld\nv_smem_size: %ld\nshared memory size: %ld\ncu count: %d\n", this->gcn_arch, q_smem_size, k_smem_size, v_smem_size, this->lds_size, this->cu_count);
            }
        } else if constexpr (Func == FAFUNC::BACKWARD) {
            this->lds_size = 32 * 1024;
            if(this->gcn_arch >= 936 && Kernel_traits::kHeadDim <= 128){
                if(this->gcn_arch == 936 || this->gcn_arch == 938) {
                    this->lds_size = 21 * 1024;
                } else {
                    this->lds_size = 16 * 1024;
                }
            }
            if(Kernel_traits::kHeadDim == 256) {
                this->lds_size = 64 * 1024;
            }
        } else if constexpr (Func == FAFUNC::KVCACHE) {
            /*尚未实现, 因为 kvcache 存在 reuse, lds 大小取决于 reuse 大小*/
        }

        // 指定 CU 数目, 会影响负载均衡的效果
        const char* fa_enforce_cu_count = std::getenv("FA_ENFORCE_CU");
        if (fa_enforce_cu_count not_eq nullptr) {
           int tmp = std::atoi(fa_enforce_cu_count);
           if (tmp > 0) {
               this->cu_count = tmp;
               if (do_fa_debug) printf("cu count is enfored to be %d!\n", this->cu_count);
           }
        }
        // 指定 lds 大小, 会影响 SIMD 占用率
        const char* fa_enforce_lds_size = std::getenv("FA_ENFORCE_LDS_SIZE");
        if (fa_enforce_lds_size not_eq nullptr) {
           int tmp = std::atoi(fa_enforce_lds_size);
           if (tmp > 0) {
               this->lds_size = tmp * 1024;
               if (do_fa_debug) printf("lds size is enfored to be %ld KB!\n", this->lds_size);
           }
        }
    }

    // 禁止外部析构
    ~DeviceProperties() {
    }

    // 禁止外部拷贝构造
    DeviceProperties(const DeviceProperties &single) = delete;

    // 禁止外部赋值操作
    DeviceProperties& operator=(const DeviceProperties &single) = delete;
};