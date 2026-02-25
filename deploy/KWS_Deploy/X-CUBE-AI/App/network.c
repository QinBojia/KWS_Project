/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-02-16T03:46:29+0800
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x3d9a7886399bd4629cfdabacb28c575e"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-02-16T03:46:29+0800"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  mfcc_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 806, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  _stem_stem_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6448, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6448, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_dw_dw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6448, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_dw_dw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6448, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_pw_pw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6448, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_pw_pw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6448, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_dw_dw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6448, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_dw_dw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6448, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_pw_pw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12896, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_pw_pw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12896, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_dw_dw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3472, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_dw_dw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3472, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_pw_pw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6944, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_pw_pw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6944, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_dw_dw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6944, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_dw_dw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6944, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_pw_pw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6944, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_pw_pw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6944, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_4_dw_dw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_4_dw_dw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_4_pw_pw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_4_pw_pw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_dw_dw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_dw_dw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_pw_pw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_pw_pw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_6_dw_dw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_6_dw_dw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_6_pw_pw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_6_pw_pw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_7_dw_dw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_7_dw_dw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_7_pw_pw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_7_pw_pw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_8_dw_dw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_8_dw_dw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_8_pw_pw_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_8_pw_pw_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  logits_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 8, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _stem_stem_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 72, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _stem_stem_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_dw_dw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 72, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_dw_dw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_pw_pw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_pw_pw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_dw_dw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 72, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_dw_dw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_pw_pw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_pw_pw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_dw_dw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_dw_dw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_pw_pw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 512, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_pw_pw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_dw_dw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_dw_dw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_pw_pw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_pw_pw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_4_dw_dw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 288, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_4_dw_dw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_4_pw_pw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2048, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_4_pw_pw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_dw_dw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_dw_dw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_pw_pw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_pw_pw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_6_dw_dw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 576, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_6_dw_dw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_6_pw_pw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_6_pw_pw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_7_dw_dw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_7_dw_dw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_7_pw_pw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_7_pw_pw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_8_dw_dw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1152, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_8_dw_dw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_8_pw_pw_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16384, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_8_pw_pw_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  logits_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1024, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  logits_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  _stem_stem_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 9, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_pw_pw_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_pw_pw_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_pw_pw_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_pw_pw_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_4_pw_pw_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_5_pw_pw_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_6_pw_pw_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_7_pw_pw_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_8_pw_pw_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_dw_dw_0_Conv_output_0_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_features_features_0_dw_dw_0_Conv_output_0_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_dw_dw_0_Conv_output_0_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 13, 62), AI_STRIDE_INIT(4, 4, 4, 32, 416),
  1, &_features_features_0_dw_dw_0_Conv_output_0_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_dw_dw_0_Conv_output_0_weights, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 8), AI_STRIDE_INIT(4, 1, 8, 8, 8),
  1, &_features_features_0_dw_dw_0_Conv_output_0_weights_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_dw_dw_2_Relu_output_0_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 13, 62), AI_STRIDE_INIT(4, 4, 4, 32, 416),
  1, &_features_features_0_dw_dw_2_Relu_output_0_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_pw_pw_0_Conv_output_0_bias, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_features_features_0_pw_pw_0_Conv_output_0_bias_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_pw_pw_0_Conv_output_0_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 13, 62), AI_STRIDE_INIT(4, 4, 4, 32, 416),
  1, &_features_features_0_pw_pw_0_Conv_output_0_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_pw_pw_0_Conv_output_0_scratch0, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_features_features_0_pw_pw_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_pw_pw_0_Conv_output_0_weights, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 8), AI_STRIDE_INIT(4, 4, 32, 256, 256),
  1, &_features_features_0_pw_pw_0_Conv_output_0_weights_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_pw_pw_2_Relu_output_0_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 13, 62), AI_STRIDE_INIT(4, 4, 4, 32, 416),
  1, &_features_features_0_pw_pw_2_Relu_output_0_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_dw_dw_0_Conv_output_0_bias, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_features_features_1_dw_dw_0_Conv_output_0_bias_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_dw_dw_0_Conv_output_0_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 13, 62), AI_STRIDE_INIT(4, 4, 4, 32, 416),
  1, &_features_features_1_dw_dw_0_Conv_output_0_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_dw_dw_0_Conv_output_0_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 8), AI_STRIDE_INIT(4, 1, 8, 8, 8),
  1, &_features_features_1_dw_dw_0_Conv_output_0_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_dw_dw_2_Relu_output_0_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 13, 62), AI_STRIDE_INIT(4, 4, 4, 32, 416),
  1, &_features_features_1_dw_dw_2_Relu_output_0_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_pw_pw_0_Conv_output_0_bias, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_features_features_1_pw_pw_0_Conv_output_0_bias_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_pw_pw_0_Conv_output_0_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 13, 62), AI_STRIDE_INIT(4, 4, 4, 64, 832),
  1, &_features_features_1_pw_pw_0_Conv_output_0_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_pw_pw_0_Conv_output_0_scratch0, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_features_features_1_pw_pw_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_pw_pw_0_Conv_output_0_weights, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 1, 16), AI_STRIDE_INIT(4, 4, 32, 512, 512),
  1, &_features_features_1_pw_pw_0_Conv_output_0_weights_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_pw_pw_2_Relu_output_0_output, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 13, 62), AI_STRIDE_INIT(4, 4, 4, 64, 832),
  1, &_features_features_1_pw_pw_2_Relu_output_0_output_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_dw_dw_0_Conv_output_0_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_features_features_2_dw_dw_0_Conv_output_0_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_dw_dw_0_Conv_output_0_output, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 7, 31), AI_STRIDE_INIT(4, 4, 4, 64, 448),
  1, &_features_features_2_dw_dw_0_Conv_output_0_output_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_dw_dw_0_Conv_output_0_weights, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 16), AI_STRIDE_INIT(4, 1, 16, 16, 16),
  1, &_features_features_2_dw_dw_0_Conv_output_0_weights_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_dw_dw_2_Relu_output_0_output, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 7, 31), AI_STRIDE_INIT(4, 4, 4, 64, 448),
  1, &_features_features_2_dw_dw_2_Relu_output_0_output_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_pw_pw_0_Conv_output_0_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_features_features_2_pw_pw_0_Conv_output_0_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_pw_pw_0_Conv_output_0_output, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 7, 31), AI_STRIDE_INIT(4, 4, 4, 128, 896),
  1, &_features_features_2_pw_pw_0_Conv_output_0_output_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_pw_pw_0_Conv_output_0_scratch0, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_features_features_2_pw_pw_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_pw_pw_0_Conv_output_0_weights, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 4, 64, 2048, 2048),
  1, &_features_features_2_pw_pw_0_Conv_output_0_weights_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_pw_pw_2_Relu_output_0_output, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 7, 31), AI_STRIDE_INIT(4, 4, 4, 128, 896),
  1, &_features_features_2_pw_pw_2_Relu_output_0_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_dw_dw_0_Conv_output_0_bias, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_features_features_3_dw_dw_0_Conv_output_0_bias_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_dw_dw_0_Conv_output_0_output, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 7, 31), AI_STRIDE_INIT(4, 4, 4, 128, 896),
  1, &_features_features_3_dw_dw_0_Conv_output_0_output_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_dw_dw_0_Conv_output_0_weights, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 32), AI_STRIDE_INIT(4, 1, 32, 32, 32),
  1, &_features_features_3_dw_dw_0_Conv_output_0_weights_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_dw_dw_2_Relu_output_0_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 7, 31), AI_STRIDE_INIT(4, 4, 4, 128, 896),
  1, &_features_features_3_dw_dw_2_Relu_output_0_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_pw_pw_0_Conv_output_0_bias, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_features_features_3_pw_pw_0_Conv_output_0_bias_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_pw_pw_0_Conv_output_0_output, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 7, 31), AI_STRIDE_INIT(4, 4, 4, 128, 896),
  1, &_features_features_3_pw_pw_0_Conv_output_0_output_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_pw_pw_0_Conv_output_0_scratch0, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_features_features_3_pw_pw_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_pw_pw_0_Conv_output_0_weights, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 32), AI_STRIDE_INIT(4, 4, 128, 4096, 4096),
  1, &_features_features_3_pw_pw_0_Conv_output_0_weights_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_pw_pw_2_Relu_output_0_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 7, 31), AI_STRIDE_INIT(4, 4, 4, 128, 896),
  1, &_features_features_3_pw_pw_2_Relu_output_0_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_4_dw_dw_0_Conv_output_0_bias, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_features_features_4_dw_dw_0_Conv_output_0_bias_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_4_dw_dw_0_Conv_output_0_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 16), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &_features_features_4_dw_dw_0_Conv_output_0_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_4_dw_dw_0_Conv_output_0_weights, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 32), AI_STRIDE_INIT(4, 1, 32, 32, 32),
  1, &_features_features_4_dw_dw_0_Conv_output_0_weights_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_4_dw_dw_2_Relu_output_0_output, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 4, 16), AI_STRIDE_INIT(4, 4, 4, 128, 512),
  1, &_features_features_4_dw_dw_2_Relu_output_0_output_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_4_pw_pw_0_Conv_output_0_bias, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_features_features_4_pw_pw_0_Conv_output_0_bias_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_4_pw_pw_0_Conv_output_0_output, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 16), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &_features_features_4_pw_pw_0_Conv_output_0_output_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_4_pw_pw_0_Conv_output_0_scratch0, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_features_features_4_pw_pw_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_4_pw_pw_0_Conv_output_0_weights, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 32, 1, 1, 64), AI_STRIDE_INIT(4, 4, 128, 8192, 8192),
  1, &_features_features_4_pw_pw_0_Conv_output_0_weights_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_4_pw_pw_2_Relu_output_0_output, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 16), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &_features_features_4_pw_pw_2_Relu_output_0_output_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_dw_dw_0_Conv_output_0_bias, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_features_features_5_dw_dw_0_Conv_output_0_bias_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_dw_dw_0_Conv_output_0_output, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 16), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &_features_features_5_dw_dw_0_Conv_output_0_output_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_dw_dw_0_Conv_output_0_weights, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &_features_features_5_dw_dw_0_Conv_output_0_weights_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_dw_dw_2_Relu_output_0_output, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 16), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &_features_features_5_dw_dw_2_Relu_output_0_output_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_pw_pw_0_Conv_output_0_bias, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_features_features_5_pw_pw_0_Conv_output_0_bias_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_pw_pw_0_Conv_output_0_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 16), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &_features_features_5_pw_pw_0_Conv_output_0_output_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_pw_pw_0_Conv_output_0_scratch0, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_features_features_5_pw_pw_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_pw_pw_0_Conv_output_0_weights, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_features_features_5_pw_pw_0_Conv_output_0_weights_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_5_pw_pw_2_Relu_output_0_output, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 4, 16), AI_STRIDE_INIT(4, 4, 4, 256, 1024),
  1, &_features_features_5_pw_pw_2_Relu_output_0_output_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_6_dw_dw_0_Conv_output_0_bias, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_features_features_6_dw_dw_0_Conv_output_0_bias_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_6_dw_dw_0_Conv_output_0_output, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 2, 8), AI_STRIDE_INIT(4, 4, 4, 256, 512),
  1, &_features_features_6_dw_dw_0_Conv_output_0_output_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_6_dw_dw_0_Conv_output_0_weights, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &_features_features_6_dw_dw_0_Conv_output_0_weights_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_6_dw_dw_2_Relu_output_0_output, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 2, 8), AI_STRIDE_INIT(4, 4, 4, 256, 512),
  1, &_features_features_6_dw_dw_2_Relu_output_0_output_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_6_pw_pw_0_Conv_output_0_bias, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_features_features_6_pw_pw_0_Conv_output_0_bias_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_6_pw_pw_0_Conv_output_0_output, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 8), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &_features_features_6_pw_pw_0_Conv_output_0_output_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_6_pw_pw_0_Conv_output_0_scratch0, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_features_features_6_pw_pw_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_6_pw_pw_0_Conv_output_0_weights, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 128), AI_STRIDE_INIT(4, 4, 256, 32768, 32768),
  1, &_features_features_6_pw_pw_0_Conv_output_0_weights_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_6_pw_pw_2_Relu_output_0_output, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 8), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &_features_features_6_pw_pw_2_Relu_output_0_output_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_7_dw_dw_0_Conv_output_0_bias, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_features_features_7_dw_dw_0_Conv_output_0_bias_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_7_dw_dw_0_Conv_output_0_output, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 8), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &_features_features_7_dw_dw_0_Conv_output_0_output_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_7_dw_dw_0_Conv_output_0_weights, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &_features_features_7_dw_dw_0_Conv_output_0_weights_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_7_dw_dw_2_Relu_output_0_output, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 8), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &_features_features_7_dw_dw_2_Relu_output_0_output_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_7_pw_pw_0_Conv_output_0_bias, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_features_features_7_pw_pw_0_Conv_output_0_bias_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_7_pw_pw_0_Conv_output_0_output, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 8), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &_features_features_7_pw_pw_0_Conv_output_0_output_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_7_pw_pw_0_Conv_output_0_scratch0, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_features_features_7_pw_pw_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_7_pw_pw_0_Conv_output_0_weights, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 65536),
  1, &_features_features_7_pw_pw_0_Conv_output_0_weights_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_7_pw_pw_2_Relu_output_0_output, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 8), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &_features_features_7_pw_pw_2_Relu_output_0_output_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_8_dw_dw_0_Conv_output_0_bias, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_features_features_8_dw_dw_0_Conv_output_0_bias_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_8_dw_dw_0_Conv_output_0_output, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 8), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &_features_features_8_dw_dw_0_Conv_output_0_output_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_8_dw_dw_0_Conv_output_0_weights, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &_features_features_8_dw_dw_0_Conv_output_0_weights_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_8_dw_dw_2_Relu_output_0_output, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 8), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &_features_features_8_dw_dw_2_Relu_output_0_output_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_8_pw_pw_0_Conv_output_0_bias, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_features_features_8_pw_pw_0_Conv_output_0_bias_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_8_pw_pw_0_Conv_output_0_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 8), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &_features_features_8_pw_pw_0_Conv_output_0_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_8_pw_pw_0_Conv_output_0_scratch0, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_features_features_8_pw_pw_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_8_pw_pw_0_Conv_output_0_weights, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 4, 512, 65536, 65536),
  1, &_features_features_8_pw_pw_0_Conv_output_0_weights_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_8_pw_pw_2_Relu_output_0_output, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 2, 8), AI_STRIDE_INIT(4, 4, 4, 512, 1024),
  1, &_features_features_8_pw_pw_2_Relu_output_0_output_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_output, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_pool_GlobalAveragePool_output_0_output_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  _stem_stem_0_Conv_output_0_bias, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &_stem_stem_0_Conv_output_0_bias_array, NULL)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  _stem_stem_0_Conv_output_0_output, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 13, 62), AI_STRIDE_INIT(4, 4, 4, 32, 416),
  1, &_stem_stem_0_Conv_output_0_output_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  _stem_stem_0_Conv_output_0_scratch0, AI_STATIC,
  84, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 3, 3), AI_STRIDE_INIT(4, 4, 4, 4, 12),
  1, &_stem_stem_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  _stem_stem_0_Conv_output_0_weights, AI_STATIC,
  85, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 3, 8), AI_STRIDE_INIT(4, 4, 4, 32, 96),
  1, &_stem_stem_0_Conv_output_0_weights_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_output, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 13, 62), AI_STRIDE_INIT(4, 4, 4, 32, 416),
  1, &_stem_stem_2_Relu_output_0_output_array, NULL)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  logits_bias, AI_STATIC,
  87, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &logits_bias_array, NULL)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  logits_output, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &logits_output_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  logits_weights, AI_STATIC,
  89, 0x0,
  AI_SHAPE_INIT(4, 128, 8, 1, 1), AI_STRIDE_INIT(4, 4, 512, 4096, 4096),
  1, &logits_weights_array, NULL)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  mfcc_output, AI_STATIC,
  90, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 13, 62), AI_STRIDE_INIT(4, 4, 4, 4, 52),
  1, &mfcc_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &logits_weights, &logits_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  logits_layer, 41,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &logits_chain,
  NULL, &logits_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_8_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_pool_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _pool_GlobalAveragePool_output_0_layer, 39,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &_pool_GlobalAveragePool_output_0_chain,
  NULL, &logits_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 8), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 8), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_8_pw_pw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_8_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_8_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_8_pw_pw_2_Relu_output_0_layer, 38,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_8_pw_pw_2_Relu_output_0_chain,
  NULL, &_pool_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_8_pw_pw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_8_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_8_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_8_pw_pw_0_Conv_output_0_weights, &_features_features_8_pw_pw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_8_pw_pw_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_8_pw_pw_0_Conv_output_0_layer, 37,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_features_features_8_pw_pw_0_Conv_output_0_chain,
  NULL, &_features_features_8_pw_pw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_8_dw_dw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_8_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_8_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_8_dw_dw_2_Relu_output_0_layer, 36,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_8_dw_dw_2_Relu_output_0_chain,
  NULL, &_features_features_8_pw_pw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_8_dw_dw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_7_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_8_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_8_dw_dw_0_Conv_output_0_weights, &_features_features_8_dw_dw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_8_dw_dw_0_Conv_output_0_layer, 35,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &_features_features_8_dw_dw_0_Conv_output_0_chain,
  NULL, &_features_features_8_dw_dw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 128, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_7_pw_pw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_7_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_7_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_7_pw_pw_2_Relu_output_0_layer, 34,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_7_pw_pw_2_Relu_output_0_chain,
  NULL, &_features_features_8_dw_dw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_7_pw_pw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_7_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_7_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_7_pw_pw_0_Conv_output_0_weights, &_features_features_7_pw_pw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_7_pw_pw_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_7_pw_pw_0_Conv_output_0_layer, 33,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_features_features_7_pw_pw_0_Conv_output_0_chain,
  NULL, &_features_features_7_pw_pw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_7_dw_dw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_7_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_7_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_7_dw_dw_2_Relu_output_0_layer, 32,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_7_dw_dw_2_Relu_output_0_chain,
  NULL, &_features_features_7_pw_pw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_7_dw_dw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_6_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_7_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_7_dw_dw_0_Conv_output_0_weights, &_features_features_7_dw_dw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_7_dw_dw_0_Conv_output_0_layer, 31,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &_features_features_7_dw_dw_0_Conv_output_0_chain,
  NULL, &_features_features_7_dw_dw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 128, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_6_pw_pw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_6_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_6_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_6_pw_pw_2_Relu_output_0_layer, 30,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_6_pw_pw_2_Relu_output_0_chain,
  NULL, &_features_features_7_dw_dw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_6_pw_pw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_6_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_6_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_6_pw_pw_0_Conv_output_0_weights, &_features_features_6_pw_pw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_6_pw_pw_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_6_pw_pw_0_Conv_output_0_layer, 29,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_features_features_6_pw_pw_0_Conv_output_0_chain,
  NULL, &_features_features_6_pw_pw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_6_dw_dw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_6_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_6_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_6_dw_dw_2_Relu_output_0_layer, 28,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_6_dw_dw_2_Relu_output_0_chain,
  NULL, &_features_features_6_pw_pw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_6_dw_dw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_6_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_6_dw_dw_0_Conv_output_0_weights, &_features_features_6_dw_dw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_6_dw_dw_0_Conv_output_0_layer, 27,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &_features_features_6_dw_dw_0_Conv_output_0_chain,
  NULL, &_features_features_6_dw_dw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_5_pw_pw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_5_pw_pw_2_Relu_output_0_layer, 26,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_5_pw_pw_2_Relu_output_0_chain,
  NULL, &_features_features_6_dw_dw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_5_pw_pw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_5_pw_pw_0_Conv_output_0_weights, &_features_features_5_pw_pw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_5_pw_pw_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_5_pw_pw_0_Conv_output_0_layer, 25,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_features_features_5_pw_pw_0_Conv_output_0_chain,
  NULL, &_features_features_5_pw_pw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_5_dw_dw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_5_dw_dw_2_Relu_output_0_layer, 24,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_5_dw_dw_2_Relu_output_0_chain,
  NULL, &_features_features_5_pw_pw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_5_dw_dw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_4_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_5_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_5_dw_dw_0_Conv_output_0_weights, &_features_features_5_dw_dw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_5_dw_dw_0_Conv_output_0_layer, 23,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &_features_features_5_dw_dw_0_Conv_output_0_chain,
  NULL, &_features_features_5_dw_dw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 64, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_4_pw_pw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_4_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_4_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_4_pw_pw_2_Relu_output_0_layer, 22,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_4_pw_pw_2_Relu_output_0_chain,
  NULL, &_features_features_5_dw_dw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_4_pw_pw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_4_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_4_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_4_pw_pw_0_Conv_output_0_weights, &_features_features_4_pw_pw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_4_pw_pw_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_4_pw_pw_0_Conv_output_0_layer, 21,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_features_features_4_pw_pw_0_Conv_output_0_chain,
  NULL, &_features_features_4_pw_pw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_4_dw_dw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_4_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_4_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_4_dw_dw_2_Relu_output_0_layer, 20,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_4_dw_dw_2_Relu_output_0_chain,
  NULL, &_features_features_4_pw_pw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_4_dw_dw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_4_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_4_dw_dw_0_Conv_output_0_weights, &_features_features_4_dw_dw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_4_dw_dw_0_Conv_output_0_layer, 19,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &_features_features_4_dw_dw_0_Conv_output_0_chain,
  NULL, &_features_features_4_dw_dw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 32, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_3_pw_pw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_3_pw_pw_2_Relu_output_0_layer, 18,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_3_pw_pw_2_Relu_output_0_chain,
  NULL, &_features_features_4_dw_dw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_3_pw_pw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_3_pw_pw_0_Conv_output_0_weights, &_features_features_3_pw_pw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_3_pw_pw_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_3_pw_pw_0_Conv_output_0_layer, 17,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_features_features_3_pw_pw_0_Conv_output_0_chain,
  NULL, &_features_features_3_pw_pw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_3_dw_dw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_3_dw_dw_2_Relu_output_0_layer, 16,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_3_dw_dw_2_Relu_output_0_chain,
  NULL, &_features_features_3_pw_pw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_3_dw_dw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_3_dw_dw_0_Conv_output_0_weights, &_features_features_3_dw_dw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_3_dw_dw_0_Conv_output_0_layer, 15,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &_features_features_3_dw_dw_0_Conv_output_0_chain,
  NULL, &_features_features_3_dw_dw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_2_pw_pw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_2_pw_pw_2_Relu_output_0_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_2_pw_pw_2_Relu_output_0_chain,
  NULL, &_features_features_3_dw_dw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_2_pw_pw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_2_pw_pw_0_Conv_output_0_weights, &_features_features_2_pw_pw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_2_pw_pw_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_2_pw_pw_0_Conv_output_0_layer, 13,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_features_features_2_pw_pw_0_Conv_output_0_chain,
  NULL, &_features_features_2_pw_pw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_2_dw_dw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_2_dw_dw_2_Relu_output_0_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_2_dw_dw_2_Relu_output_0_chain,
  NULL, &_features_features_2_pw_pw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_2_dw_dw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_2_dw_dw_0_Conv_output_0_weights, &_features_features_2_dw_dw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_2_dw_dw_0_Conv_output_0_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &_features_features_2_dw_dw_0_Conv_output_0_chain,
  NULL, &_features_features_2_dw_dw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 16, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_1_pw_pw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_1_pw_pw_2_Relu_output_0_layer, 10,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_1_pw_pw_2_Relu_output_0_chain,
  NULL, &_features_features_2_dw_dw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_1_pw_pw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_1_pw_pw_0_Conv_output_0_weights, &_features_features_1_pw_pw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_1_pw_pw_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_1_pw_pw_0_Conv_output_0_layer, 9,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_features_features_1_pw_pw_0_Conv_output_0_chain,
  NULL, &_features_features_1_pw_pw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_1_dw_dw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_1_dw_dw_2_Relu_output_0_layer, 8,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_1_dw_dw_2_Relu_output_0_chain,
  NULL, &_features_features_1_pw_pw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_1_dw_dw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_1_dw_dw_0_Conv_output_0_weights, &_features_features_1_dw_dw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_1_dw_dw_0_Conv_output_0_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &_features_features_1_dw_dw_0_Conv_output_0_chain,
  NULL, &_features_features_1_dw_dw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 8, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_0_pw_pw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_pw_pw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_0_pw_pw_2_Relu_output_0_layer, 6,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_0_pw_pw_2_Relu_output_0_chain,
  NULL, &_features_features_1_dw_dw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_0_pw_pw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_pw_pw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_0_pw_pw_0_Conv_output_0_weights, &_features_features_0_pw_pw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_0_pw_pw_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_0_pw_pw_0_Conv_output_0_layer, 5,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_features_features_0_pw_pw_0_Conv_output_0_chain,
  NULL, &_features_features_0_pw_pw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_0_dw_dw_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_dw_dw_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_0_dw_dw_2_Relu_output_0_layer, 4,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_features_features_0_dw_dw_2_Relu_output_0_chain,
  NULL, &_features_features_0_pw_pw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_0_dw_dw_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_stem_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_dw_dw_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_0_dw_dw_0_Conv_output_0_weights, &_features_features_0_dw_dw_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_0_dw_dw_0_Conv_output_0_layer, 3,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_dw_if32of32wf32,
  &_features_features_0_dw_dw_0_Conv_output_0_chain,
  NULL, &_features_features_0_dw_dw_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 8, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_stem_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_stem_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _stem_stem_2_Relu_output_0_layer, 2,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_stem_stem_2_Relu_output_0_chain,
  NULL, &_features_features_0_dw_dw_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _stem_stem_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mfcc_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_stem_stem_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_stem_stem_0_Conv_output_0_weights, &_stem_stem_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_stem_stem_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _stem_stem_0_Conv_output_0_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_stem_stem_0_Conv_output_0_chain,
  NULL, &_stem_stem_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 221376, 1, 1),
    221376, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 53376, 1, 1),
    53376, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &mfcc_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &logits_output),
  &_stem_stem_0_Conv_output_0_layer, 0xa245b3a1, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 221376, 1, 1),
      221376, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 53376, 1, 1),
      53376, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &mfcc_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &logits_output),
  &_stem_stem_0_Conv_output_0_layer, 0xa245b3a1, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    mfcc_output_array.data = AI_PTR(g_network_activations_map[0] + 28896);
    mfcc_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28896);
    _stem_stem_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 32120);
    _stem_stem_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 32120);
    _stem_stem_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3104);
    _stem_stem_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3104);
    _stem_stem_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3104);
    _stem_stem_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3104);
    _features_features_0_dw_dw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2176);
    _features_features_0_dw_dw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2176);
    _features_features_0_dw_dw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 2176);
    _features_features_0_dw_dw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 2176);
    _features_features_0_pw_pw_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 32124);
    _features_features_0_pw_pw_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 32124);
    _features_features_0_pw_pw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1760);
    _features_features_0_pw_pw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1760);
    _features_features_0_pw_pw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 1760);
    _features_features_0_pw_pw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1760);
    _features_features_1_dw_dw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 27552);
    _features_features_1_dw_dw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 27552);
    _features_features_1_dw_dw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 27552);
    _features_features_1_dw_dw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 27552);
    _features_features_1_pw_pw_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 53344);
    _features_features_1_pw_pw_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 53344);
    _features_features_1_pw_pw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 512);
    _features_features_1_pw_pw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 512);
    _features_features_1_pw_pw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 512);
    _features_features_1_pw_pw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 512);
    _features_features_2_dw_dw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_2_dw_dw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_2_dw_dw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 39488);
    _features_features_2_dw_dw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 39488);
    _features_features_2_pw_pw_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_2_pw_pw_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_2_pw_pw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 11712);
    _features_features_2_pw_pw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 11712);
    _features_features_2_pw_pw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 11712);
    _features_features_2_pw_pw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 11712);
    _features_features_3_dw_dw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 9536);
    _features_features_3_dw_dw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9536);
    _features_features_3_dw_dw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 9536);
    _features_features_3_dw_dw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9536);
    _features_features_3_pw_pw_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_3_pw_pw_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_3_pw_pw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8640);
    _features_features_3_pw_pw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8640);
    _features_features_3_pw_pw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8640);
    _features_features_3_pw_pw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8640);
    _features_features_4_dw_dw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_4_dw_dw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_4_dw_dw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8192);
    _features_features_4_dw_dw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8192);
    _features_features_4_pw_pw_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_4_pw_pw_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_4_pw_pw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16384);
    _features_features_4_pw_pw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16384);
    _features_features_4_pw_pw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_4_pw_pw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_5_dw_dw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16384);
    _features_features_5_dw_dw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16384);
    _features_features_5_dw_dw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_5_dw_dw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_5_pw_pw_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 16384);
    _features_features_5_pw_pw_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 16384);
    _features_features_5_pw_pw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16640);
    _features_features_5_pw_pw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16640);
    _features_features_5_pw_pw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_5_pw_pw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_6_dw_dw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16384);
    _features_features_6_dw_dw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16384);
    _features_features_6_dw_dw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_6_dw_dw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_6_pw_pw_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 4096);
    _features_features_6_pw_pw_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 4096);
    _features_features_6_pw_pw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 4352);
    _features_features_6_pw_pw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4352);
    _features_features_6_pw_pw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 12544);
    _features_features_6_pw_pw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 12544);
    _features_features_7_dw_dw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_7_dw_dw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_7_dw_dw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8192);
    _features_features_7_dw_dw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8192);
    _features_features_7_pw_pw_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_7_pw_pw_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_7_pw_pw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 16384);
    _features_features_7_pw_pw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 16384);
    _features_features_7_pw_pw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_7_pw_pw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_8_dw_dw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8192);
    _features_features_8_dw_dw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8192);
    _features_features_8_dw_dw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_8_dw_dw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_8_pw_pw_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 8192);
    _features_features_8_pw_pw_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 8192);
    _features_features_8_pw_pw_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8704);
    _features_features_8_pw_pw_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8704);
    _features_features_8_pw_pw_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _features_features_8_pw_pw_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _pool_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 8192);
    _pool_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 8192);
    logits_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    logits_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _stem_stem_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _stem_stem_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    _stem_stem_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    _stem_stem_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _stem_stem_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 288);
    _stem_stem_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 288);
    _features_features_0_dw_dw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_0_dw_dw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 320);
    _features_features_0_dw_dw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 320);
    _features_features_0_dw_dw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_0_dw_dw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 608);
    _features_features_0_dw_dw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 608);
    _features_features_0_pw_pw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_0_pw_pw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 640);
    _features_features_0_pw_pw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 640);
    _features_features_0_pw_pw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_0_pw_pw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 896);
    _features_features_0_pw_pw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 896);
    _features_features_1_dw_dw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_1_dw_dw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 928);
    _features_features_1_dw_dw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 928);
    _features_features_1_dw_dw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_1_dw_dw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1216);
    _features_features_1_dw_dw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1216);
    _features_features_1_pw_pw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_1_pw_pw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 1248);
    _features_features_1_pw_pw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1248);
    _features_features_1_pw_pw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_1_pw_pw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1760);
    _features_features_1_pw_pw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1760);
    _features_features_2_dw_dw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_2_dw_dw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 1824);
    _features_features_2_dw_dw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1824);
    _features_features_2_dw_dw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_2_dw_dw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 2400);
    _features_features_2_dw_dw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2400);
    _features_features_2_pw_pw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_2_pw_pw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 2464);
    _features_features_2_pw_pw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2464);
    _features_features_2_pw_pw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_2_pw_pw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 4512);
    _features_features_2_pw_pw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4512);
    _features_features_3_dw_dw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_3_dw_dw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 4640);
    _features_features_3_dw_dw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 4640);
    _features_features_3_dw_dw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_3_dw_dw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 5792);
    _features_features_3_dw_dw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 5792);
    _features_features_3_pw_pw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_3_pw_pw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 5920);
    _features_features_3_pw_pw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 5920);
    _features_features_3_pw_pw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_3_pw_pw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 10016);
    _features_features_3_pw_pw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 10016);
    _features_features_4_dw_dw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_4_dw_dw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 10144);
    _features_features_4_dw_dw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 10144);
    _features_features_4_dw_dw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_4_dw_dw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 11296);
    _features_features_4_dw_dw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 11296);
    _features_features_4_pw_pw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_4_pw_pw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 11424);
    _features_features_4_pw_pw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 11424);
    _features_features_4_pw_pw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_4_pw_pw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 19616);
    _features_features_4_pw_pw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 19616);
    _features_features_5_dw_dw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_5_dw_dw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 19872);
    _features_features_5_dw_dw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 19872);
    _features_features_5_dw_dw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_5_dw_dw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 22176);
    _features_features_5_dw_dw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 22176);
    _features_features_5_pw_pw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_5_pw_pw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 22432);
    _features_features_5_pw_pw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 22432);
    _features_features_5_pw_pw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_5_pw_pw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 38816);
    _features_features_5_pw_pw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 38816);
    _features_features_6_dw_dw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_6_dw_dw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 39072);
    _features_features_6_dw_dw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 39072);
    _features_features_6_dw_dw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_6_dw_dw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 41376);
    _features_features_6_dw_dw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 41376);
    _features_features_6_pw_pw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_6_pw_pw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 41632);
    _features_features_6_pw_pw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 41632);
    _features_features_6_pw_pw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_6_pw_pw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 74400);
    _features_features_6_pw_pw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 74400);
    _features_features_7_dw_dw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_7_dw_dw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 74912);
    _features_features_7_dw_dw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 74912);
    _features_features_7_dw_dw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_7_dw_dw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 79520);
    _features_features_7_dw_dw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 79520);
    _features_features_7_pw_pw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_7_pw_pw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 80032);
    _features_features_7_pw_pw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 80032);
    _features_features_7_pw_pw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_7_pw_pw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 145568);
    _features_features_7_pw_pw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 145568);
    _features_features_8_dw_dw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_8_dw_dw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 146080);
    _features_features_8_dw_dw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 146080);
    _features_features_8_dw_dw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_8_dw_dw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 150688);
    _features_features_8_dw_dw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 150688);
    _features_features_8_pw_pw_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_8_pw_pw_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 151200);
    _features_features_8_pw_pw_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 151200);
    _features_features_8_pw_pw_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_8_pw_pw_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 216736);
    _features_features_8_pw_pw_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 216736);
    logits_weights_array.format |= AI_FMT_FLAG_CONST;
    logits_weights_array.data = AI_PTR(g_network_weights_map[0] + 217248);
    logits_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 217248);
    logits_bias_array.format |= AI_FMT_FLAG_CONST;
    logits_bias_array.data = AI_PTR(g_network_weights_map[0] + 221344);
    logits_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 221344);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1998616,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xa245b3a1,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 1998616,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xa245b3a1,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

