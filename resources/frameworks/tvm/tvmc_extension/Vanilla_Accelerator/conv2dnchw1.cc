
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


#ifdef __cplusplus
extern "C"
#endif

    /*!
     * \brief Conv2D function for mock-accelerator examples. Limited to same-padded Conv2D with
     * stride (1,1) and datatype float. \param ifmap Pointer to input feature map data of size
     * iw*ih*ic*sizeof(float). \param weights Pointer to weight data of size
     * kh*kw*ic**oc*sizeof(float). \param result Pointer to output feature map data of size
     * iw*ih*oc*sizeof(float). \param oc Number of channels of output feature map. \param iw Width
     * of input feature map, ifmap. \param ih Height of input feature map, ifmap. \param ic Number
     * of channels of input feature map. \param kh Height of convolution kernels. \param kw Width of
     * convolution kernels.
     *
     * \return error code
     *
     */



int32_t vanilla_accelerator_conv2dnchw(float* ifmap, float* weights, float* result, int32_t oc, int32_t iw, int32_t ih, int32_t ic,
                        int32_t kh, int32_t kw) {

  //VanillaAccelerator base_adr: 0x70001000

  *(uint32_t**)0x70001000 = (uint32_t*)ifmap;
  *(uint32_t**)0x70001004 = (uint32_t*)weights;
  *(uint32_t**)0x70001008 = (uint32_t*)result;
  

  *(uint32_t*)0x7000100c = oc;
  *(uint32_t*)0x70001010 = iw;
  *(uint32_t*)0x70001014 = ih;
  *(uint32_t*)0x70001018 = ic;
  *(uint32_t*)0x7000101c = kh;
  *(uint32_t*)0x70001020 = kw;


  //issue start signal
  //printf("issue start ...\n");
  *(uint32_t*)0x70001024 = 0x00000001;

  return 0;
}