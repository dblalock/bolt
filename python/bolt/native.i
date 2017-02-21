
%module bolt
%{
#define SWIG_FILE_WITH_INIT
#include <vector>
#include <sys/types.h>
#include "../../cpp/src/include/public.hpp"
%}

%include <config.i>

// ================================================================
// actually have swig parse + wrap the files
// ================================================================
%include "../../cpp/src/include/public.hpp"
