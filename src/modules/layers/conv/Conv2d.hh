#pragma once
#include <immintrin.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

#include "src/modules/layers/layer.hh"
#include "src/modules/optimizers/optimizer.hh"

constexpr int ALIGNMENT = 32;

class Conv2d : public ConvLayer
{

};