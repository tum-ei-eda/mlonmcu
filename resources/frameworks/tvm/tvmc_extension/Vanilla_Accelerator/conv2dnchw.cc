
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

typedef struct regs 
        {
            uint32_t ifmap;   
            uint32_t weights; 
            uint32_t result;  
            uint32_t oc;      
            uint32_t iw;      
            uint32_t ih;      
            uint32_t ic;      
            uint32_t kh;      
            uint32_t kw;  
            uint32_t control;
        } regs_t;

int vanilla_accelerator_conv2dnchw(float* ifmap, float* weights, float* result, int oc, int iw, int ih, int ic,
                        int kh, int kw) {
  
  //VanillaAccelerator base_adr: 0x70001000
  regs_t *p_regs  = (regs_t *)0x70001000;  // set the base address of the peripheral, that would come form some hw ip header. 
	p_regs->ifmap   = (uint32_t) ifmap; 
	p_regs->weights = (uint32_t) weights;
	p_regs->result  = (uint32_t) result;  
	
	p_regs->oc = oc;      
	p_regs->iw = iw;      
	p_regs->ih = ih;      
	p_regs->ic = ic;      
	p_regs->kh = kh;      
	p_regs->kw = kw; 
	p_regs->control = 1;  // last command, to start the operation
  
  
  

  return 0;
}
