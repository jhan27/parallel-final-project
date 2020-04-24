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
#include <GLFW/glfw3.h>
#include <time.h>
#include "screen.h"
#include "drawing.h"
#include "basic_types.h"
#include "space_controller.h"
#include "build_config.h"

#define WINDOW_TITLE "GravitySim"
#define SUCCESS 0
#define FAILURE 1

GLFWwindow* window;

static int gl_init(int width, int height, const char *title);
static void gl_close(void);
static void error_callback(int error, const char* description);
bool main_loop(SpaceController *controller, GS_FLOAT start);
void print_usage(const char *program_name);
SimulationConfig get_config(int argc, const char *argv[]);

int main(int argc, const char * argv[]) {
    bool loop = true;
    srand((unsigned)time(NULL));
    if (gl_init(WINDOW_W, WINDOW_H, WINDOW_TITLE) != SUCCESS) {
        return FAILURE;
    }

    SimulationConfig config = get_config(argc, argv);
    SpaceController *controller = spacecontroller_init(config);
    if (!controller) {
        return FAILURE;
    }

    GS_FLOAT start = glfwGetTime();
    while (loop) {
        loop = main_loop(controller, start);
    }

    spacecontroller_dealloc(controller);
    gl_close();
    return SUCCESS;
}

bool main_loop(SpaceController *controller, GS_FLOAT start) {
    GS_FLOAT old_time = glfwGetTime();
    GS_FLOAT current_time;
    GS_FLOAT dt;

    int cnt = 0;
    while (1) {
        current_time = glfwGetTime();
        dt = current_time - old_time;
        
        if(glfwGetKey(window, GLFW_KEY_ESCAPE) || cnt == controller->num_iter) {
            GS_FLOAT end = glfwGetTime();
            GS_FLOAT duration = end - start;
            printf("main_loop execution time: %.4f ms\n", duration * 1000);
            printf("number of loops: %d\n", cnt);
            printf("avgearge time per loop: %.4f ms\n", duration / cnt * 1000);
            return false;
        }
        spacecontroller_update(window, controller, dt);
        glfwPollEvents();
        old_time = current_time;
        cnt++;
    }
    return true;
}

static int gl_init(int width, int height, const char *title) {
    int status;
    status = glfwInit();
    if (status != GL_TRUE) {
        return FAILURE;
    }
    glfwSetErrorCallback(error_callback);
    window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window) {
        return FAILURE;
    }
    glfwMakeContextCurrent(window);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect_ratio = ((float)height) / width;
    glFrustum(.5, -.5, -.5 * aspect_ratio, .5 * aspect_ratio, 1, 50);
    glMatrixMode(GL_MODELVIEW);
    return SUCCESS;
}

static void gl_close(void) {
    glfwTerminate();
}

void print_usage(const char *program_name) {
    printf("Usage:%s number_of_galaxies objects_per_galaxy galaxy_size\n", program_name);
    printf("Using default config.\n");
}

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
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

