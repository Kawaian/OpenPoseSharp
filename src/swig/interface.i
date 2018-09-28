#define OP_API

%module OpenPose

%include <typemaps.i>
%include <std_vector.i>
%include <std_pair.i>
%include <std_map.i>
%include <std_string.i>
%include <std_array.i>

/**
    ---------------------------------------------------
		           Include C++ Headers
    ---------------------------------------------------
**/

%{
#include "openpose/core/macros.hpp"
#include "openpose/core/common.hpp"
#include "openpose/core/enumClasses.hpp"
#include "openpose/core/point.hpp"
#include "openpose/core/array.hpp"
#include "openpose/core/rectangle.hpp"

#include "openpose/hand/handExtractorNet.hpp"
#include "openpose/hand/handExtractorCaffe.hpp"

using namespace op;
%}

/**
    ---------------------------------------------------
        Convert C++ references to C# references
    ---------------------------------------------------
**/

// -- BOOL --
%typemap(cstype)    bool & "/* cstype */ out bool"
%typemap(imtype)    bool & "/* imtype */ out bool"
%typemap(csin)      bool & %{out $csinput%}

// -- INT --
%typemap(cstype)    int & "/* cstype */ out int"
%typemap(imtype)    int & "/* imtype */ out int"
%typemap(csin)      int & %{out $csinput%}

// -- DOUBLE --
%typemap(cstype)    double & "/* cstype */ out double"
%typemap(imtype)    double & "/* imtype */ out double"
%typemap(csin)      double & %{out $csinput%}
//%apply double *OUTPUT { double *result };
//%apply double &OUTPUT { double &result };

// -- FLOAT --
//%typemap(cstype)    float & "/* cstype */ out float"
//%typemap(imtype)    float & "/* imtype */ out float"
%typemap(csin)      float & %{out $csinput%}
%apply float *OUTPUT { float *result };
%apply float &OUTPUT { float &result };

// -- FLOAT --
%typemap(cstype)    HeatMapType & "/* cstype */ out HeatMapType"
%typemap(imtype)    HeatMapType & "/* imtype */ out HeatMapType"
%typemap(csin)      HeatMapType & %{out $csinput%}

%include "openpose/core/macros.hpp"
%include "openpose/core/common.hpp"
%include "openpose/core/enumClasses.hpp"
%include "openpose/core/array.hpp"
%include "openpose/core/point.hpp"
%include "openpose/core/rectangle.hpp"

%include "openpose/hand/handExtractorNet.hpp"
%include "openpose/hand/handExtractorCaffe.hpp"

%template(FloatPoint) Point<float>;
%template(IntPoint) Point<int>;

%template(Rectangle) Rectangle<float>;
%template(IntList) std::vector<int>;
%template(HeatMapTypeList) std::vector<HeatMapType>;

%template(FloatArray) Array<float>;
%template(FloatArray2Array) std::array<Array<float>, 2>;

%template(FloatRectangle2Array) std::array<Rectangle<float>, 2>;
%template(FloatRectangle2ArrayList) std::vector<std::array<Rectangle<float>, 2>>;