//
//  space_controller.h
//  GravitySim
//
//  Created by Krzysztof Gabis on 24.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#ifndef GravitySim_space_controller_h
#define GravitySim_space_controller_h

#include "space_model.h"

typedef struct {
    RectangleD view_bounds;
    RectangleD model_bounds;
    size_t objects_n;
    size_t galaxies_n;
    GS_FLOAT galaxy_size;
    int num_iter;
} SimulationConfig;

typedef struct {
    SpaceModel *model_seq;
    SpaceModel *model_cuda;
    int num_iter;
} SpaceController;

SpaceController* spacecontroller_init(SimulationConfig config);
void spacecontroller_update_cuda(SpaceController *c, SimulationConfig config, GS_FLOAT dt);
void spacecontroller_update(SpaceController *controller, GS_FLOAT dt);
void spacecontroller_dealloc(SpaceController *controller);

#endif
