//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 3.0.12
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------

namespace OpenPoseSharp {

public class HandExtractorCaffe : HandExtractorNet {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;

  internal HandExtractorCaffe(global::System.IntPtr cPtr, bool cMemoryOwn) : base(OpenPosePINVOKE.HandExtractorCaffe_SWIGUpcast(cPtr), cMemoryOwn) {
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(HandExtractorCaffe obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~HandExtractorCaffe() {
    Dispose();
  }

  public override void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          OpenPosePINVOKE.delete_HandExtractorCaffe(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
      base.Dispose();
    }
  }

  public HandExtractorCaffe(IntPoint netInputSize, IntPoint netOutputSize, string modelFolder, int gpuId, ushort numberScales, float rangeScales, HeatMapTypeList heatMapTypes, ScaleMode heatMapScale, bool enableGoogleLogging) : this(OpenPosePINVOKE.new_HandExtractorCaffe__SWIG_0(IntPoint.getCPtr(netInputSize), IntPoint.getCPtr(netOutputSize), modelFolder, gpuId, numberScales, rangeScales, HeatMapTypeList.getCPtr(heatMapTypes), (int)heatMapScale, enableGoogleLogging), true) {
    if (OpenPosePINVOKE.SWIGPendingException.Pending) throw OpenPosePINVOKE.SWIGPendingException.Retrieve();
  }

  public HandExtractorCaffe(IntPoint netInputSize, IntPoint netOutputSize, string modelFolder, int gpuId, ushort numberScales, float rangeScales, HeatMapTypeList heatMapTypes, ScaleMode heatMapScale) : this(OpenPosePINVOKE.new_HandExtractorCaffe__SWIG_1(IntPoint.getCPtr(netInputSize), IntPoint.getCPtr(netOutputSize), modelFolder, gpuId, numberScales, rangeScales, HeatMapTypeList.getCPtr(heatMapTypes), (int)heatMapScale), true) {
    if (OpenPosePINVOKE.SWIGPendingException.Pending) throw OpenPosePINVOKE.SWIGPendingException.Retrieve();
  }

  public HandExtractorCaffe(IntPoint netInputSize, IntPoint netOutputSize, string modelFolder, int gpuId, ushort numberScales, float rangeScales, HeatMapTypeList heatMapTypes) : this(OpenPosePINVOKE.new_HandExtractorCaffe__SWIG_2(IntPoint.getCPtr(netInputSize), IntPoint.getCPtr(netOutputSize), modelFolder, gpuId, numberScales, rangeScales, HeatMapTypeList.getCPtr(heatMapTypes)), true) {
    if (OpenPosePINVOKE.SWIGPendingException.Pending) throw OpenPosePINVOKE.SWIGPendingException.Retrieve();
  }

  public HandExtractorCaffe(IntPoint netInputSize, IntPoint netOutputSize, string modelFolder, int gpuId, ushort numberScales, float rangeScales) : this(OpenPosePINVOKE.new_HandExtractorCaffe__SWIG_3(IntPoint.getCPtr(netInputSize), IntPoint.getCPtr(netOutputSize), modelFolder, gpuId, numberScales, rangeScales), true) {
    if (OpenPosePINVOKE.SWIGPendingException.Pending) throw OpenPosePINVOKE.SWIGPendingException.Retrieve();
  }

  public HandExtractorCaffe(IntPoint netInputSize, IntPoint netOutputSize, string modelFolder, int gpuId, ushort numberScales) : this(OpenPosePINVOKE.new_HandExtractorCaffe__SWIG_4(IntPoint.getCPtr(netInputSize), IntPoint.getCPtr(netOutputSize), modelFolder, gpuId, numberScales), true) {
    if (OpenPosePINVOKE.SWIGPendingException.Pending) throw OpenPosePINVOKE.SWIGPendingException.Retrieve();
  }

  public HandExtractorCaffe(IntPoint netInputSize, IntPoint netOutputSize, string modelFolder, int gpuId) : this(OpenPosePINVOKE.new_HandExtractorCaffe__SWIG_5(IntPoint.getCPtr(netInputSize), IntPoint.getCPtr(netOutputSize), modelFolder, gpuId), true) {
    if (OpenPosePINVOKE.SWIGPendingException.Pending) throw OpenPosePINVOKE.SWIGPendingException.Retrieve();
  }

  public virtual void netInitializationOnThread() {
    OpenPosePINVOKE.HandExtractorCaffe_netInitializationOnThread(swigCPtr);
  }

  public override void forwardPass(FloatRectangle2ArrayList handRectangles, SWIGTYPE_p_cv__Mat cvInputData) {
    OpenPosePINVOKE.HandExtractorCaffe_forwardPass(swigCPtr, FloatRectangle2ArrayList.getCPtr(handRectangles), SWIGTYPE_p_cv__Mat.getCPtr(cvInputData));
    if (OpenPosePINVOKE.SWIGPendingException.Pending) throw OpenPosePINVOKE.SWIGPendingException.Retrieve();
  }

}

}
