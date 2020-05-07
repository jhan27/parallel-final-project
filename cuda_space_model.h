#ifndef cuda_space_model_h
#define cuda_space_model_h

#include "quad_tree.h"
#include "space_model.h"
#include "build_config.h"
#include "space_controller.h"

void main_update(SpaceModel *m, SimulationConfig config, GS_FLOAT dt);

#endif