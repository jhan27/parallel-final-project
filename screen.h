//
//  screen.h
//  GravitySim
//
//  Created by Krzysztof Gabis on 23.01.2013.
//  Copyright (c) 2013 Krzysztof Gabis. All rights reserved.
//

#ifndef GravitySim_screen_h
#define GravitySim_screen_h
#define GL_SILENCE_DEPRECATION

#include <stdio.h>
#include <GLFW/glfw3.h>
#include "basic_types.h"

typedef struct {
    int width;
    int height;
    RGBColor *pixels;
} Screen;

Screen *screen_init(int width, int height);
void screen_fill(Screen *screen, RGBColor color);
void screen_display(GLFWwindow* window, Screen *screen);
void screen_dealloc(Screen *screen);

#endif
