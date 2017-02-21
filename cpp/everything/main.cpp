//
//  main.cpp
//  Dig
//
//  Created by DB on 1/20/16.
//  Copyright © 2016 D Blalock. All rights reserved.
//

#include <stdio.h>

// unit tests magic
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
// #include "test/catch.hpp"

int main(int argc, char *const argv[]) {
	return Catch::Session().run(argc, argv);
}
