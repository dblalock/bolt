
// #include "../../cpp/src/utils/plot.hpp"

%define %np_vector_typemaps(DTYPE, NPY_DTYPE)

// %fragment("NP_VECTOR_Fragments", "header",  fragment="NumPy_Fragments")
// %{
// 	template <class T>
//   bool ConvertFromNumpyToVector(std::vector<DATA_T>* out, PyObject* in)
//   {

//   }

// %}

namespace std {
	// hmm...apparently telling SWIG to try to optimize this breaks it
	// %typemap(out, fragment="NumPy_Fragments", optimal="1") vector<DTYPE> {
	%typemap(out, fragment="NumPy_Fragments") vector<DTYPE> {
		// create python array of appropriate shape
	 	npy_intp sz = static_cast<npy_intp>($1.size());
	 	npy_intp dims[] = {sz};
	 	PyObject* out_array = PyArray_SimpleNew(1, dims, NPY_DTYPE);

		if (! out_array) {
		    PyErr_SetString(PyExc_ValueError,
		                    "vector wrap: unable to create the output array.");
		    return NULL;
		}

		// ar::plot_array($1.data(), $1.size());

		// copy data from vect into numpy array
		DTYPE* out_data = (DTYPE*) array_data(out_array);
		for (size_t i = 0; i < sz; i++) {
			out_data[i] = static_cast<DTYPE>($1[i]);
			// printf("%.3g -> %.3g\n", (double)$1[i], (double)out_data[i] );
		}

		// ar::plot_array(out_array, sz);

		$result = out_array;
	}

	// %typemap(in, fragment="NumPy_Fragments") vector<DTYPE>
	%typemap(in, fragment="NumPy_Fragments") vector<DTYPE> (vector<DTYPE> temp) {
		// python list
		if (PyList_Check($input)) {
			temp.resize(PyList_Size($input));
			for (size_t i=0; i != PyList_Size($input); ++i) {
		    	temp[i] = PyList_GetItem($input, i);
			}
			$1 = temp;
		numpy array
		} else if (is_array($input)) {
		// if (is_array($input)) {
			if (array_numdims($input) > 1) { // TODO also allow row/column vect
				PyErr_SetString(PyExc_ValueError,
					"Vector conversion requires 1D array.");
				SWIG_fail;
		    }
	        // ensure contiguous array
		    int isNewObject = 0;
		    PyArrayObject* in = obj_to_array_contiguous_allow_conversion($input,
		    	array_type($input), &isNewObject);
		    if (in == NULL) {
				PyErr_SetString(PyExc_ValueError,
					"Could not convert the input into a Python array object.");
				SWIG_fail;
		    }
		    // copy data
			temp.resize(array_size($input, 0));
			DTYPE* data = static_cast<DTYPE*>(array_data(in));
			for (int i = 0; i < array_size($input, 0); i++) {
				temp[i] = data[i];
			}
			$1 = temp;
		// neither list nor array
		} else {
			PyErr_SetString(PyExc_ValueError,
					"Input was not a python list or array");
			SWIG_fail;
		}
	}

	%typemap(in, fragment="NumPy_Fragments") const vector<DTYPE>& (vector<DTYPE> temp) {
		// // python list
		// if (PyList_Check($input)) {
		// 	temp.resize(PyList_Size($input));
		// 	for (size_t i=0; i != PyList_Size($input); ++i) {
		//     	temp[i] = PyList_GetItem($input, i);
		// 	}
		// 	$1 = temp;
		// // numpy array
		// } else if (is_array($input)) {
		if (is_array($input)) {
			if (array_numdims($input) > 1) { // TODO also allow row/column vect
				PyErr_SetString(PyExc_ValueError,
					"Vector conversion requires 1D array.");
				SWIG_fail;
		    }
	        // ensure contiguous array
		    int isNewObject = 0;
		    PyArrayObject* in = obj_to_array_contiguous_allow_conversion($input,
		    	array_type($input), &isNewObject);
		    if (in == NULL) {
				PyErr_SetString(PyExc_ValueError,
					"Could not convert the input into a Python array object.");
				SWIG_fail;
		    }
		    // copy data
			temp.resize(array_size($input, 0));
			DTYPE* data = static_cast<DTYPE*>(array_data(in));
			for (int i = 0; i < array_size($input, 0); i++) {
				temp[i] = data[i];
			}
			$1 = &temp;
		// neither list nor array
		} else {
			PyErr_SetString(PyExc_ValueError,
					"Input was not a python list or array");
			SWIG_fail;
		}
	}

	// %apply(vector<DTYPE>) {(const vector<DTYPE>&)};

	// %typemap(in, fragment="NumPy_Fragments") const vector<DTYPE>& (std::vector<DTYPE > temp) {
	// 	// python list
	// 	if (PyList_Check($input)) {
	// 		temp.resize(PyList_Size($input));
	// 		for (size_t i=0; i != PyList_Size($input); ++i) {
	// 	    	temp[i] = PyList_GetItem($input, i);
	// 		}
	// 		$1 = temp;
	// 	// numpy array
	// 	} else if (is_array(in)) {
	// 		if (array_numdims(in) > 1) { // TODO also allow row/column vect
	// 			PyErr_SetString(PyExc_ValueError,
	// 				"Vector conversion requires 1D array.");
	// 			SWIG_fail;
	// 	    }
	//         // ensure contiguous array
	// 	    int isNewObject = 0;
	// 	    PyArrayObject* temp = obj_to_array_contiguous_allow_conversion(in,
	// 	    	array_type(in), &isNewObject);
	// 	    if (temp == NULL) {
	// 			PyErr_SetString(PyExc_ValueError,
	// 				"Could not convert the input into a Python array object.");
	// 			SWIG_fail;
	// 	    }
	// 	    // copy data
	// 		temp.resize(array_size(in, 0));
	// 		DTYPE* data = static_cast<DTYPE*>(array_data(temp))
	// 		for (int i = 0; i < array_size(in, 0); i++) {
	// 			temp[i] = data[i];
	// 		}
	// 		$1 = temp;
	// 	// neither list nor array
	// 	} else {
	// 		PyErr_SetString(PyExc_ValueError,
	// 				"Input was not a python list or array");
	// 		SWIG_fail;
	// 	}
	// }
}

%enddef
