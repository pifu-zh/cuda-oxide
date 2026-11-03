/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
/* Host driver for the panic transport probe. SPDX-License-Identifier: Apache-2.0 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(x) do{hipError_t _e=(x);if(_e){printf("ERR %s@%d\n",hipGetErrorString(_e),__LINE__);return 1;}}while(0)
static void* img_load(const char* p){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);void*b=malloc(n);if(fread(b,1,n,f)!=(size_t)n)exit(1);fclose(f);return b;}
int main(){
  CK(hipInit(0));
  hipModule_t mod; CK(hipModuleLoadData(&mod, img_load("/home/zhuo/tmp-prof/panic_probe/probe.hsaco")));
  hipFunction_t f1,f2;
  CK(hipModuleGetFunction(&f1,mod,"probe_const"));
  CK(hipModuleGetFunction(&f2,mod,"probe_global"));
  printf("=== launch probe_const\n");
  CK(hipModuleLaunchKernel(f1,1,1,1,1,1,1,0,0,NULL,NULL));
  CK(hipDeviceSynchronize());
  printf("=== launch probe_global\n");
  CK(hipModuleLaunchKernel(f2,1,1,1,1,1,1,0,0,NULL,NULL));
  CK(hipDeviceSynchronize());
  printf("=== done\n");
  return 0;
}
