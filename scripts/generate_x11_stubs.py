#!/usr/bin/env python3
"""
Generate minimal X11 stub headers + a linkable stub .c file
so BAOlab can compile without XQuartz / libX11.

All X11 display functions become no-ops.  Batch-mode commands
(mksynth, mkcmppsf, etc.) work normally; only interactive GUI
features are disabled.
"""
import os
import subprocess
import sys
import textwrap

XLIB_H = textwrap.dedent(r"""
#ifndef _XLIB_H_STUB
#define _XLIB_H_STUB
#define X_H

typedef unsigned long XID, Window, Drawable, Pixmap, Colormap, Atom, Time, KeySym;
typedef unsigned long GC;
typedef int Bool, Status;
typedef unsigned long Font;

typedef struct { int type; unsigned long serial; int send_event; void *display;
  Window window; Window root; Window subwindow; Time time;
  int x, y, x_root, y_root; unsigned int state; unsigned int keycode;
  int same_screen; } XKeyEvent;
typedef XKeyEvent XKeyPressedEvent;

typedef struct { int type; unsigned long serial; int send_event; void *display;
  Window window; int x, y; int width, height; int count; } XExposeEvent;

typedef struct { int type; unsigned long serial; int send_event; void *display;
  Window window; int x, y; unsigned int width, height; int border_width;
  Window above; int override_redirect; } XConfigureEvent;

typedef struct { int type; unsigned long serial; int send_event; void *display;
  Window window; Window root; Window subwindow; Time time;
  int x, y, x_root, y_root; unsigned int state; unsigned int button;
  int same_screen; } XButtonEvent;

typedef struct { int type; unsigned long serial; int send_event; void *display;
  Window window; Window root; Window subwindow; Time time;
  int x, y, x_root, y_root; unsigned int state; char is_hint;
  int same_screen; } XMotionEvent;

typedef struct { int type; unsigned long serial; int send_event; void *display;
  Window window; } XAnyEvent;

typedef union { int type; XAnyEvent xany; XKeyEvent xkey; XButtonEvent xbutton;
  XMotionEvent xmotion; XExposeEvent xexpose; XConfigureEvent xconfigure;
  long pad[24]; } XEvent;

typedef struct { unsigned long pixel; unsigned short red, green, blue;
  char flags; char pad; } XColor;

typedef struct { void *ext_data; int depth; int bits_per_pixel;
  int scanline_pad; } XVisualInfo;

typedef struct { void *ext_data; int width, height; int xoffset; int format;
  char *data; int byte_order; int bitmap_unit; int bitmap_bit_order;
  int bitmap_pad; int depth; int bytes_per_line; int bits_per_pixel;
  unsigned long red_mask, green_mask, blue_mask;
  void *obdata; void (*f)(void); } XImage;

typedef struct { int function; unsigned long plane_mask;
  unsigned long foreground, background; int line_width; int line_style;
  int cap_style; int join_style; int fill_style; int fill_rule; int arc_mode;
  Pixmap tile; Pixmap stipple; int ts_x_origin, ts_y_origin;
  unsigned long font; int subwindow_mode; int graphics_exposures;
  int clip_x_origin, clip_y_origin; Pixmap clip_mask;
  int dash_offset; char dashes; } XGCValues;

typedef struct { int x, y; unsigned int width, height; } XRectangle;

typedef struct { long flags; int x, y; int width, height;
  int min_width, min_height; int max_width, max_height;
  int width_inc, height_inc; struct { int x, y; } min_aspect, max_aspect;
  int base_width, base_height; int win_gravity; } XSizeHints;

typedef struct _XDisplay { int fd; int proto_major_version; } Display;

typedef struct { unsigned long flags; int function; unsigned long plane_mask;
  unsigned long foreground; unsigned long background; int line_width;
  unsigned long backing_pixel; unsigned long backing_store;
  unsigned long event_mask; Colormap colormap; } XSetWindowAttributes;

typedef struct { int x, y, width, height; unsigned int class; int depth;
  Colormap colormap; unsigned long backing_store; void *visual;
  } XWindowAttributes;

typedef struct { unsigned char *value; Atom encoding; int format;
  unsigned long nitems; } XTextProperty;

typedef struct { short x, y; } XPoint;
typedef struct { int dummy; } XComposeStatus;
typedef struct { short lbearing, rbearing, width, ascent, descent;
  unsigned short attributes; } XCharStruct;
typedef struct { int x, y, width, height; int border_width;
  int sibling; int stack_mode; } XWindowChanges;
typedef void Visual;

/* ── constants ── */
#define None 0L
#define False 0
#define True 1

#define GCForeground (1L<<2)
#define GCBackground (1L<<3)
#define GCLineWidth  (1L<<4)
#define GCFont       (1L<<14)
#define GCFunction   (1L<<0)

#define CWBackPixel    (1L<<1)
#define CWBorderPixel  (1L<<3)
#define CWBackingStore (1L<<6)
#define CWEventMask    (1L<<11)
#define CWColormap     (1L<<13)
#define CWX            (1<<0)
#define CWY            (1<<1)
#define CWWidth        (1<<2)
#define CWHeight       (1<<3)

#define ExposureMask           (1L<<15)
#define KeyPressMask           (1L<<0)
#define ButtonPressMask        (1L<<2)
#define ButtonReleaseMask      (1L<<3)
#define EnterWindowMask        (1L<<4)
#define LeaveWindowMask        (1L<<5)
#define PointerMotionMask      (1L<<6)
#define PointerMotionHintMask  (1L<<7)
#define Button1MotionMask      (1L<<8)
#define Button3MotionMask      (1L<<10)
#define StructureNotifyMask    (1L<<17)
#define SubstructureNotifyMask (1L<<19)
#define SubstructureRedirectMask (1L<<20)
#define FocusChangeMask        (1L<<21)
#define VisibilityChangeMask   (1L<<16)
#define PropertyChangeMask     (1L<<22)
#define ColormapChangeMask     (1L<<23)

#define InputOutput  1
#define InputOnly    2
#define CopyFromParent 0L
#define WhenMapped   1
#define Always       2
#define ZPixmap      2
#define TrueColor    4
#define CoordModeOrigin 0
#define Complex      0
#define AllPlanes    (~0L)
#define GXcopy       0x3

#define LineSolid      0
#define LineOnOffDash  1
#define CapButt        1
#define JoinMiter      0
#define JoinRound      1
#define FillSolid      0
#define EvenOddRule    0
#define ArcChord       0

#define DoRed   (1<<0)
#define DoGreen (1<<1)
#define DoBlue  (1<<2)

#define Expose           12
#define ConfigureNotify  22
#define KeyPress         2
#define ButtonPress      4
#define ButtonRelease    5
#define MotionNotify     6
#define EnterNotify      7
#define LeaveNotify      8
#define FocusIn          9
#define FocusOut        10
#define VisibilityNotify 15

#define PMinSize   (1L<<4)
#define PMaxSize   (1L<<5)
#define PSize      (1L<<3)
#define PPosition  (1L<<2)
#define USSize     (1L<<1)
#define USPosition (1L<<0)

#define DefaultScreen(d)       0
#define DefaultRootWindow(d)   0
#define DefaultColormap(d,s)   0
#define DefaultDepth(d,s)      24
#define DefaultVisual(d,s)     ((void*)0)
#define BlackPixel(d,s)        0
#define WhitePixel(d,s)        0xFFFFFF
#define XKeysymToKeycode(d,k)  (0)

#define XK_Left      0xff51
#define XK_Up        0xff52
#define XK_Right     0xff53
#define XK_Down      0xff54
#define XK_Return    0xff0d
#define XK_Escape    0xff1b
#define XK_BackSpace 0xff08
#define XK_Delete    0xffff
#define XK_Home      0xff50
#define XK_End       0xff57
#define XK_space     0x20

/* ── inline stub functions (compile-time resolution) ── */
static inline Display* XOpenDisplay(const char *n){(void)n;return(Display*)0;}
static inline int XCloseDisplay(Display *d){(void)d;return 0;}
static inline Window XCreateSimpleWindow(Display *d,Window p,int x,int y,unsigned w,unsigned h,unsigned bw,unsigned long bd,unsigned long bg){(void)d;(void)p;(void)x;(void)y;(void)w;(void)h;(void)bw;(void)bd;(void)bg;return 0;}
static inline Window XCreateWindow(Display *d,Window p,int x,int y,unsigned w,unsigned h,unsigned bw,int dep,unsigned cls,void *vis,unsigned long vm,void *att){(void)d;(void)p;(void)x;(void)y;(void)w;(void)h;(void)bw;(void)dep;(void)cls;(void)vis;(void)vm;(void)att;return 0;}
static inline int XMapWindow(Display *d,Window w){(void)d;(void)w;return 0;}
static inline int XUnmapWindow(Display *d,Window w){(void)d;(void)w;return 0;}
static inline int XDestroyWindow(Display *d,Window w){(void)d;(void)w;return 0;}
static inline GC XCreateGC(Display *d,Drawable dr,unsigned long vm,XGCValues *v){(void)d;(void)dr;(void)vm;(void)v;return 0;}
static inline int XFreeGC(Display *d,GC g){(void)d;(void)g;return 0;}
static inline int XSetForeground(Display *d,GC g,unsigned long c){(void)d;(void)g;(void)c;return 0;}
static inline int XSetBackground(Display *d,GC g,unsigned long c){(void)d;(void)g;(void)c;return 0;}
static inline int XDrawLine(Display *d,Drawable dr,GC g,int x1,int y1,int x2,int y2){(void)d;(void)dr;(void)g;(void)x1;(void)y1;(void)x2;(void)y2;return 0;}
static inline int XDrawRectangle(Display *d,Drawable dr,GC g,int x,int y,unsigned w,unsigned h){(void)d;(void)dr;(void)g;(void)x;(void)y;(void)w;(void)h;return 0;}
static inline int XFillRectangle(Display *d,Drawable dr,GC g,int x,int y,unsigned w,unsigned h){(void)d;(void)dr;(void)g;(void)x;(void)y;(void)w;(void)h;return 0;}
static inline int XDrawString(Display *d,Drawable dr,GC g,int x,int y,const char *s,int l){(void)d;(void)dr;(void)g;(void)x;(void)y;(void)s;(void)l;return 0;}
static inline int XDrawImageString(Display *d,Drawable dr,GC g,int x,int y,const char *s,int l){(void)d;(void)dr;(void)g;(void)x;(void)y;(void)s;(void)l;return 0;}
static inline int XFlush(Display *d){(void)d;return 0;}
static inline int XSync(Display *d,int disc){(void)d;(void)disc;return 0;}
static inline int XPutImage(Display *d,Drawable dr,GC g,XImage *i,int sx,int sy,int dx,int dy,unsigned w,unsigned h){(void)d;(void)dr;(void)g;(void)i;(void)sx;(void)sy;(void)dx;(void)dy;(void)w;(void)h;return 0;}
static inline XImage* XCreateImage(Display *d,void *v,unsigned dep,int fmt,int off,char *data,unsigned w,unsigned h,int bp,int bpl){(void)d;(void)v;(void)dep;(void)fmt;(void)off;(void)data;(void)w;(void)h;(void)bp;(void)bpl;return(XImage*)0;}
static inline int XDestroyImage(XImage *i){(void)i;return 0;}
static inline Status XAllocColor(Display *d,Colormap cm,XColor *c){(void)d;(void)cm;if(c){c->pixel=0;}return 1;}
static inline int XDefaultDepth(Display *d,int s){(void)d;(void)s;return 24;}
static inline int XDefaultScreen(Display *d){(void)d;return 0;}
static inline Window XDefaultRootWindow(Display *d){(void)d;return 0;}
static inline Window XRootWindow(Display *d,int s){(void)d;(void)s;return 0;}
static inline void* XDefaultVisual(Display *d,int s){(void)d;(void)s;return(void*)0;}
static inline Colormap XDefaultColormap(Display *d,int s){(void)d;(void)s;return 0;}
static inline int XNextEvent(Display *d,XEvent *e){(void)d;(void)e;return 0;}
static inline int XCheckWindowEvent(Display *d,Window w,long m,XEvent *e){(void)d;(void)w;(void)m;(void)e;return 0;}
static inline int XSelectInput(Display *d,Window w,long m){(void)d;(void)w;(void)m;return 0;}
static inline int XPending(Display *d){(void)d;return 0;}
static inline int XMaskEvent(Display *d,long m,XEvent *e){(void)d;(void)m;(void)e;return 0;}
static inline int XWindowEvent(Display *d,Window w,long m,XEvent *e){(void)d;(void)w;(void)m;(void)e;return 0;}
static inline int XLookupString(XKeyEvent *e,char *buf,int len,KeySym *ks,void *st){(void)e;(void)buf;(void)len;(void)ks;(void)st;return 0;}
static inline int XTextWidth(void *fs,const char *s,int c){(void)fs;(void)s;(void)c;return 0;}
static inline void* XLoadQueryFont(Display *d,const char *n){(void)d;(void)n;return(void*)0;}
static inline int XSetFont(Display *d,GC g,unsigned long f){(void)d;(void)g;(void)f;return 0;}
static inline int XResizeWindow(Display *d,Window w,unsigned wd,unsigned ht){(void)d;(void)w;(void)wd;(void)ht;return 0;}
static inline int XMoveResizeWindow(Display *d,Window w,int x,int y,unsigned wd,unsigned ht){(void)d;(void)w;(void)x;(void)y;(void)wd;(void)ht;return 0;}
static inline int XStoreName(Display *d,Window w,const char *n){(void)d;(void)w;(void)n;return 0;}
static inline void XSetWMNormalHints(Display *d,Window w,XSizeHints *h){(void)d;(void)w;(void)h;}
static inline int XFillPolygon(Display *d,Drawable dr,GC g,void *pts,int n,int sh,int m){(void)d;(void)dr;(void)g;(void)pts;(void)n;(void)sh;(void)m;return 0;}
static inline int XDrawPoint(Display *d,Drawable dr,GC g,int x,int y){(void)d;(void)dr;(void)g;(void)x;(void)y;return 0;}
static inline int XBell(Display *d,int pct){(void)d;(void)pct;return 0;}
static inline int XFree(void *p){(void)p;return 0;}
static inline int DisplayWidth(Display *d,int s){(void)d;(void)s;return 1920;}
static inline int DisplayHeight(Display *d,int s){(void)d;(void)s;return 1080;}
static inline int DisplayPlanes(Display *d,int s){(void)d;(void)s;return 24;}
static inline int ConnectionNumber(Display *d){(void)d;return -1;}
static inline int XSetLineAttributes(Display *d,GC g,unsigned lw,int ls,int cs,int js){(void)d;(void)g;(void)lw;(void)ls;(void)cs;(void)js;return 0;}
static inline XSizeHints* XAllocSizeHints(void){static XSizeHints h;return &h;}
static inline Status XStringListToTextProperty(char **l,int c,XTextProperty *tp){(void)l;(void)c;(void)tp;return 1;}
static inline void XSetWMProperties(Display *d,Window w,XTextProperty *wn,XTextProperty *in,char **argv,int argc,XSizeHints *nh,void *wh,void *ch){(void)d;(void)w;(void)wn;(void)in;(void)argv;(void)argc;(void)nh;(void)wh;(void)ch;}
static inline int XChangeWindowAttributes(Display *d,Window w,unsigned long vm,void *att){(void)d;(void)w;(void)vm;(void)att;return 0;}
static inline Font XLoadFont(Display *d,const char *n){(void)d;(void)n;return 0;}
static inline Status XGetWindowAttributes(Display *d,Window w,XWindowAttributes *wa){(void)d;(void)w;if(wa){wa->x=0;wa->y=0;wa->width=1920;wa->height=1080;wa->depth=24;}return 1;}
static inline int XClearWindow(Display *d,Window w){(void)d;(void)w;return 0;}
static inline int XClearArea(Display *d,Window w,int x,int y,unsigned wd,unsigned ht,int exp){(void)d;(void)w;(void)x;(void)y;(void)wd;(void)ht;(void)exp;return 0;}
static inline int XCopyArea(Display *d,Drawable s,Drawable dst,GC g,int sx,int sy,unsigned w,unsigned h,int dx,int dy){(void)d;(void)s;(void)dst;(void)g;(void)sx;(void)sy;(void)w;(void)h;(void)dx;(void)dy;return 0;}
static inline XImage* XGetImage(Display *d,Drawable dr,int x,int y,unsigned w,unsigned h,unsigned long pm,int fmt){(void)d;(void)dr;(void)x;(void)y;(void)w;(void)h;(void)pm;(void)fmt;return(XImage*)0;}
static inline unsigned long XGetPixel(XImage *i,int x,int y){(void)i;(void)x;(void)y;return 0;}
static inline int XPutPixel(XImage *i,int x,int y,unsigned long p){(void)i;(void)x;(void)y;(void)p;return 0;}
static inline int XSetClipMask(Display *d,GC g,Pixmap p){(void)d;(void)g;(void)p;return 0;}
static inline int XDrawPoints(Display *d,Drawable dr,GC g,XPoint *pts,int n,int m){(void)d;(void)dr;(void)g;(void)pts;(void)n;(void)m;return 0;}
static inline int XDrawArc(Display *d,Drawable dr,GC g,int x,int y,unsigned w,unsigned h,int a1,int a2){(void)d;(void)dr;(void)g;(void)x;(void)y;(void)w;(void)h;(void)a1;(void)a2;return 0;}
static inline int XFillArc(Display *d,Drawable dr,GC g,int x,int y,unsigned w,unsigned h,int a1,int a2){(void)d;(void)dr;(void)g;(void)x;(void)y;(void)w;(void)h;(void)a1;(void)a2;return 0;}
static inline int XAllocColorCells(Display *d,Colormap cm,int co,unsigned long *pm,unsigned np,unsigned long *px,unsigned npx){(void)d;(void)cm;(void)co;(void)pm;(void)np;(void)px;(void)npx;return 0;}
static inline int XStoreColors(Display *d,Colormap cm,XColor *c,int n){(void)d;(void)cm;(void)c;(void)n;return 0;}
static inline int XConfigureWindow(Display *d,Window w,unsigned vm,XWindowChanges *ch){(void)d;(void)w;(void)vm;(void)ch;return 0;}
static inline int XRaiseWindow(Display *d,Window w){(void)d;(void)w;return 0;}
static inline void XQueryTextExtents(Display *d,unsigned long fid,const char *s,int n,int *d1,int *d2,int *d3,XCharStruct *cs){(void)d;(void)fid;(void)s;(void)n;(void)d1;(void)d2;(void)d3;if(cs){cs->width=(short)(8*n);cs->ascent=12;cs->descent=3;}}
static inline Status XMatchVisualInfo(Display *d,int s,int dep,int cls,void *vi){(void)d;(void)s;(void)dep;(void)cls;(void)vi;return 0;}

#endif /* _XLIB_H_STUB */
""").lstrip()

XUTIL_H = textwrap.dedent(r"""
#ifndef _XUTIL_H_STUB
#define _XUTIL_H_STUB
#include <X11/Xlib.h>
#endif
""").lstrip()

KEYSYM_H = textwrap.dedent(r"""
#ifndef _KEYSYM_H_STUB
#define _KEYSYM_H_STUB
/* keysyms defined in Xlib.h stub */
#endif
""").lstrip()

XMU_DRAWING_H = textwrap.dedent(r"""
#ifndef _XMU_DRAWING_H_STUB
#define _XMU_DRAWING_H_STUB
#include <X11/Xlib.h>
static inline int XmuDrawRoundedRectangle(Display *d,Drawable dr,GC g,int x,int y,int w,int h,int ew,int eh){(void)d;(void)dr;(void)g;(void)x;(void)y;(void)w;(void)h;(void)ew;(void)eh;return 0;}
static inline int XmuFillRoundedRectangle(Display *d,Drawable dr,GC g,int x,int y,int w,int h,int ew,int eh){(void)d;(void)dr;(void)g;(void)x;(void)y;(void)w;(void)h;(void)ew;(void)eh;return 0;}
#endif
""").lstrip()

X11STUB_C = textwrap.dedent(r"""
/*
 * Link-time stubs for X11 symbols that are called via implicit
 * declarations (suppressed with -Wno-implicit-function-declaration)
 * and therefore not resolved by the header's static-inline stubs.
 */
#include <stdlib.h>
typedef unsigned long XID, Window, Drawable, Pixmap, Colormap, Atom, GC;
typedef void Display;
typedef struct { int x, y; } XPoint;
typedef struct { unsigned long pixel; unsigned short red, green, blue;
  char flags; char pad; } XColor;
typedef struct { int function; unsigned long plane_mask;
  unsigned long foreground, background; int line_width;
  unsigned long font; } XGCValues;

Window RootWindow(Display *d, int s) { (void)d;(void)s; return 0; }
int XChangeGC(Display *d, GC g, unsigned long vm, XGCValues *v) {
  (void)d;(void)g;(void)vm;(void)v; return 0; }
int XDrawLines(Display *d, Drawable dr, GC g, XPoint *pts, int n, int m) {
  (void)d;(void)dr;(void)g;(void)pts;(void)n;(void)m; return 0; }
int XFreeColors(Display *d, Colormap cm, unsigned long *px, int n, unsigned long pl) {
  (void)d;(void)cm;(void)px;(void)n;(void)pl; return 0; }
int XGetGCValues(Display *d, GC g, unsigned long vm, XGCValues *v) {
  (void)d;(void)g;(void)vm;
  if(v){v->foreground=0;v->background=0;v->font=0;} return 1; }
int XParseColor(Display *d, Colormap cm, const char *spec, XColor *c) {
  (void)d;(void)cm;(void)spec;
  if(c){c->red=0;c->green=0;c->blue=0;c->pixel=0;} return 1; }
int XReconfigureWMWindow(Display *d, Window w, int s, unsigned vm, void *ch) {
  (void)d;(void)w;(void)s;(void)vm;(void)ch; return 0; }
int XStoreColor(Display *d, Colormap cm, XColor *c) {
  (void)d;(void)cm;(void)c; return 0; }
""").lstrip()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <stub_dir>", file=sys.stderr)
        sys.exit(1)

    stub_dir = sys.argv[1]
    x11 = os.path.join(stub_dir, "X11")
    xmu = os.path.join(x11, "Xmu")
    os.makedirs(xmu, exist_ok=True)

    with open(os.path.join(x11, "Xlib.h"), "w") as f:
        f.write(XLIB_H)
    with open(os.path.join(x11, "Xutil.h"), "w") as f:
        f.write(XUTIL_H)
    with open(os.path.join(x11, "keysym.h"), "w") as f:
        f.write(KEYSYM_H)
    with open(os.path.join(xmu, "Drawing.h"), "w") as f:
        f.write(XMU_DRAWING_H)

    stub_c = os.path.join(stub_dir, "..", "x11stub.c")
    stub_o = os.path.join(stub_dir, "..", "x11stub.o")
    with open(stub_c, "w") as f:
        f.write(X11STUB_C)

    subprocess.check_call(["cc", "-O", "-c", stub_c, "-o", stub_o])
    print(f"X11 stubs generated in {stub_dir} + {stub_o}")


if __name__ == "__main__":
    main()
