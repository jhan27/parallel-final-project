//
//  space_controller.c
//  GravitySim
//
//  Created by Krzysztof Gabis on 24.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#include <stdio.h>
#include "space_controller.h"
#include "build_config.h"
#include "cuda_space_model.h"


SpaceController* spacecontroller_init(SimulationConfig config) {
    SpaceController *controller = (SpaceController*)malloc(sizeof(SpaceController));
    if (!controller) {
        return NULL;
    }
    controller->model_seq = spacemodel_init_galaxies(config.model_bounds, config.view_bounds,
                                                 config.galaxies_n, config.objects_n, config.galaxy_size);
    
    controller->model_cuda = spacemodel_init_galaxies(config.model_bounds, config.view_bounds,
                                                 config.galaxies_n, config.objects_n, config.galaxy_size);
    
    if (!controller->model_seq || !controller->model_cuda) {
        free(controller);
        return NULL;
    }
    
    controller->num_iter = config.num_iter;
    return controller;
}

void spacecontroller_update_cuda(SpaceController *c, SimulationConfig config, GS_FLOAT dt) {
    static GS_FLOAT last_update_time = 0.0;
    dt = CONST_TIME;
    main_update(c->model_cuda, config, dt);
    last_update_time += dt;
    if (last_update_time >= (1.0 / MAX_FPS))
    {
        last_update_time = 0.0;
    }
}

void spacecontroller_update(SpaceController *c, GS_FLOAT dt) {
    static GS_FLOAT last_update_time = 0.0;
    spacemodel_update(c->model_seq, dt);
    last_update_time += dt;
    if (last_update_time >= (1.0 / MAX_FPS)) {
        last_update_time = 0.0;
    }
}

void spacecontroller_dealloc(SpaceController *controller) {
    spacemodel_dealloc(controller->model_seq);
    spacemodel_dealloc(controller->model_cuda);

    free(controller);
}
