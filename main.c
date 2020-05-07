//
//  main.c
//  GravitySim
//
//  Created by Krzysztof Gabis on 23.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#define GL_SILENCE_DEPRECATION

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "basic_types.h"
#include "build_config.h"
#include "space_controller.h" 

#define WINDOW_TITLE "GravitySim"
#define SUCCESS 0
#define FAILURE 1

bool main_loop(SpaceController *controller, SimulationConfig config, GS_FLOAT start, bool seq);
void print_usage(const char *program_name);
SimulationConfig get_config(int argc, const char *argv[]);

int main(int argc, const char * argv[]) {
    bool loop = true;
    srand(0);

    SimulationConfig config = get_config(argc, argv);
    SpaceController *controller = spacecontroller_init(config);

    if (!controller) {
        return FAILURE;
    }

    clock_t start = clock();
    while (loop) {
        loop = main_loop(controller, config, start, true);
    }

    loop = true;
    start = clock();
    while (loop) {
      loop = main_loop(controller, config, start, false);
    }

    spacecontroller_dealloc(controller);
    return SUCCESS;
}

bool main_loop(SpaceController *controller, SimulationConfig config, GS_FLOAT start, bool seq) {
    clock_t old_time = clock();
    clock_t current_time;
    clock_t dt;
    
    int cnt = 0;
    while (1) {
        current_time = clock();
        dt = current_time - old_time;

        if(controller->model_cuda->objects->len == 0 || 
           controller->model_seq->objects->len == 0  ||
           cnt == controller->num_iter) {
            clock_t end = clock();
            double duration = (double) (end - start) / CLOCKS_PER_SEC * 1000;

            if (seq) {
              printf("\n---------- sequential ----------\n");
            } else {
              printf("\n------------- cuda ------------- \n");
            }

            printf("Average runtime per each simulation iteration: %.4f ms\n\n", duration / cnt);

            return false;
        }

        if (seq) {
           spacecontroller_update(controller, dt);
        } else {
           spacecontroller_update_cuda(controller, config, dt);
        }
        old_time = current_time;
        cnt++;
    }
    return true;
}

void print_usage(const char *program_name) {
    printf("Usage:%s number_of_galaxies objects_per_galaxy galaxy_size\n", program_name);
    printf("Using default config.\n");
}

SimulationConfig get_config(int argc, const char *argv[]) {
    SimulationConfig config;
    config.galaxies_n = GALAXY_NUM;
    config.galaxy_size = GALAXY_SIZE;
    config.model_bounds = MODEL_BOUNDS;
    config.view_bounds = WINDOW_BOUNDS;
    config.objects_n = OBJECT_NUM;
    config.num_iter = NUM_ITER;
    if (argc != 4) {
        print_usage(argv[0]);
        return config;
    }
    config.galaxies_n = atoi(argv[1]);
    config.objects_n = atoi(argv[2]);
    config.galaxy_size = atoi(argv[3]);
    return config;
}

