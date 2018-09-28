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

public class Rectangle : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal Rectangle(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(Rectangle obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~Rectangle() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          OpenPosePINVOKE.delete_Rectangle(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
    }
  }

  public float x {
    set {
      OpenPosePINVOKE.Rectangle_x_set(swigCPtr, value);
    } 
    get {
      float ret = OpenPosePINVOKE.Rectangle_x_get(swigCPtr);
      return ret;
    } 
  }

  public float y {
    set {
      OpenPosePINVOKE.Rectangle_y_set(swigCPtr, value);
    } 
    get {
      float ret = OpenPosePINVOKE.Rectangle_y_get(swigCPtr);
      return ret;
    } 
  }

  public float width {
    set {
      OpenPosePINVOKE.Rectangle_width_set(swigCPtr, value);
    } 
    get {
      float ret = OpenPosePINVOKE.Rectangle_width_get(swigCPtr);
      return ret;
    } 
  }

  public float height {
    set {
      OpenPosePINVOKE.Rectangle_height_set(swigCPtr, value);
    } 
    get {
      float ret = OpenPosePINVOKE.Rectangle_height_get(swigCPtr);
      return ret;
    } 
  }

  public Rectangle(float x, float y, float width, float height) : this(OpenPosePINVOKE.new_Rectangle__SWIG_0(x, y, width, height), true) {
  }

  public Rectangle(float x, float y, float width) : this(OpenPosePINVOKE.new_Rectangle__SWIG_1(x, y, width), true) {
  }

  public Rectangle(float x, float y) : this(OpenPosePINVOKE.new_Rectangle__SWIG_2(x, y), true) {
  }

  public Rectangle(float x) : this(OpenPosePINVOKE.new_Rectangle__SWIG_3(x), true) {
  }

  public Rectangle() : this(OpenPosePINVOKE.new_Rectangle__SWIG_4(), true) {
  }

  public Rectangle(Rectangle rectangle) : this(OpenPosePINVOKE.new_Rectangle__SWIG_5(Rectangle.getCPtr(rectangle)), true) {
    if (OpenPosePINVOKE.SWIGPendingException.Pending) throw OpenPosePINVOKE.SWIGPendingException.Retrieve();
  }

  public FloatPoint center() {
    FloatPoint ret = new FloatPoint(OpenPosePINVOKE.Rectangle_center(swigCPtr), true);
    return ret;
  }

  public FloatPoint topLeft() {
    FloatPoint ret = new FloatPoint(OpenPosePINVOKE.Rectangle_topLeft(swigCPtr), true);
    return ret;
  }

  public FloatPoint bottomRight() {
    FloatPoint ret = new FloatPoint(OpenPosePINVOKE.Rectangle_bottomRight(swigCPtr), true);
    return ret;
  }

  public float area() {
    float ret = OpenPosePINVOKE.Rectangle_area(swigCPtr);
    return ret;
  }

  public void recenter(float newWidth, float newHeight) {
    OpenPosePINVOKE.Rectangle_recenter(swigCPtr, newWidth, newHeight);
  }

  public string toString() {
    string ret = OpenPosePINVOKE.Rectangle_toString(swigCPtr);
    return ret;
  }

}

}