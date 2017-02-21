//
//  testing_utils.cpp
//  TimeKit
//
//  Created by DB on 10/22/14.
//  Copyright (c) 2014 DB. All rights reserved.
//

#include <math.h>

///used for comparisons ignoring slight floating point errors
static const double EQUALITY_THRESHOLD = .00001;


short int approxEq(double a, double b) {
	return fabs(a - b) < EQUALITY_THRESHOLD;
}

double rnd(double a) {
	return ((double)round(a / EQUALITY_THRESHOLD )) * EQUALITY_THRESHOLD;
}
