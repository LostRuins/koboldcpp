#ifndef __NAME_CONVERSTION_H__
#define __NAME_CONVERSTION_H__

#include <string>

#include "model.h"

bool is_cond_stage_model_name(const std::string& name);
bool is_diffusion_model_name(const std::string& name);
bool is_first_stage_model_name(const std::string& name);

std::string convert_tensor_name(std::string name, SDVersion version);

#endif  // __NAME_CONVERSTION_H__