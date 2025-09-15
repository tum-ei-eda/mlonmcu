

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


#ifdef __cplusplus
extern "C"
#endif

    /*!
     * \brief Conv2D function for mock-accelerator examples. Limited to same-padded Conv2D with
     * stride (1,1) and datatype int8, as well as a bias addition. \param ifmap Pointer to input feature map data of size
     * iw*ih*ic*sizeof(int8). \param weights Pointer to weight data of size
     * kh*kw*ic**oc*sizeof(int8). \param bias_data Pointer to bias data of size oc*sizeof(int32).
     * \param result Pointer to output feature map data of size
     * iw*ih*oc*sizeof(int32). \param oc Number of channels of output feature map. \param iw Width
     * of input feature map, ifmap. \param ih Height of input feature map, ifmap. \param ic Number
     * of channels of input feature map. \param kh Height of convolution kernels. \param kw Width of
     * convolution kernels.\param i_zp and k_zp zero point parameters of 
     * input feature map and kernel.
     *
     * \return error code
     *
     */
typedef struct regs 
        {
            uint32_t ifmap;   
            uint32_t weights;
            uint32_t bias; 
            uint32_t result;  
            int32_t oc;      
            int32_t iw;      
            int32_t ih;      
            int32_t ic;      
            int32_t kh;      
            int32_t kw; 
            int32_t i_zp;
            int32_t k_zp;
            uint32_t control;
        } regs_t;

int q_vanilla_accelerator_conv2dnchw(int8_t* q_vanilla_accelerator_0_i0, int8_t* q_vanilla_accelerator_0_i1, int32_t* bias_data, int32_t* compute,
                                      int32_t oc, int32_t iw, int32_t ih, int32_t ic, int32_t kh, int32_t kw, int32_t i_zp, int32_t k_zp) {


  // QVanillaAccelerator (without timing) base_adr = 0x70000000, 
  regs_t *p_regs = (regs_t *)0x70000000;

  p_regs->ifmap   = (uint32_t) q_vanilla_accelerator_0_i0; 
	p_regs->weights = (uint32_t) q_vanilla_accelerator_0_i1;
	p_regs->bias    = (uint32_t) bias_data; 
	p_regs->result  = (uint32_t) compute;  
	
	p_regs->oc = oc;      
	p_regs->iw = iw;      
	p_regs->ih = ih;      
	p_regs->ic = ic;      
	p_regs->kh = kh;      
	p_regs->kw = kw; 
	p_regs->i_zp = i_zp;
	p_regs->k_zp = k_zp;
	p_regs->control = 1;  //issue start signal
 
  return 0;
}
