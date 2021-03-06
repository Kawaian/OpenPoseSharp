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

public class FloatPoint : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal FloatPoint(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(FloatPoint obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~FloatPoint() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          OpenPosePINVOKE.delete_FloatPoint(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
    }
  }

  public float x {
    set {
      OpenPosePINVOKE.FloatPoint_x_set(swigCPtr, value);
    } 
    get {
      float ret = OpenPosePINVOKE.FloatPoint_x_get(swigCPtr);
      return ret;
    } 
  }

  public float y {
    set {
      OpenPosePINVOKE.FloatPoint_y_set(swigCPtr, value);
    } 
    get {
      float ret = OpenPosePINVOKE.FloatPoint_y_get(swigCPtr);
      return ret;
    } 
  }

  public FloatPoint(float x, float y) : this(OpenPosePINVOKE.new_FloatPoint__SWIG_0(x, y), true) {
  }

  public FloatPoint(float x) : this(OpenPosePINVOKE.new_FloatPoint__SWIG_1(x), true) {
  }

  public FloatPoint() : this(OpenPosePINVOKE.new_FloatPoint__SWIG_2(), true) {
  }

  public FloatPoint(FloatPoint point) : this(OpenPosePINVOKE.new_FloatPoint__SWIG_3(FloatPoint.getCPtr(point)), true) {
    if (OpenPosePINVOKE.SWIGPendingException.Pending) throw OpenPosePINVOKE.SWIGPendingException.Retrieve();
  }

  public float area() {
    float ret = OpenPosePINVOKE.FloatPoint_area(swigCPtr);
    return ret;
  }

  public string toString() {
    string ret = OpenPosePINVOKE.FloatPoint_toString(swigCPtr);
    return ret;
  }

}

}
