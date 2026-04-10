#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/input.h>
#include <sys/select.h>

#include <drm.h>
#include <drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>

typedef signed int s32;
typedef signed long long s64;

enum {
    W = 128,
    H = 128,
    TW = 96,
    TH = 64,
    FIX_SHIFT = 8,
    FIX_ONE = 1 << FIX_SHIFT,
    CUBE_VERTS = 24,
    TRI_COUNT = 12,
    TILE_W = 32,
    TILE_H = 32
};

#define FOV_DEG 54.0
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Vtx2D { s32 x, y, w, u, v; };
struct Vtx3D { s32 x, y, z; };

struct ColorPacker {
    unsigned long rmask, gmask, bmask;
    int rshift, gshift, bshift;
    int rbits, gbits, bbits;
};

struct TriData {
    s32 min_x, max_x, min_y, max_y;
    s32 dwdx, dwdy, w_origin;
    s32 dudx, dudy, u_origin;
    s32 dvdx, dvdy, v_origin;
    s32 e_ab_step_x, e_ab_step_y, e_ab_origin;
    s32 e_bc_step_x, e_bc_step_y, e_bc_origin;
    s32 e_ca_step_x, e_ca_step_y, e_ca_origin;
    bool valid;
};

static s32 g_sin_tab[360];
static s32 g_cos_tab[360];
static unsigned int g_palette[256];
static s32 g_focal = 0;

static inline s32 q8_mul(s32 a, s32 b) { return (s32)(((s64)a * (s64)b) >> FIX_SHIFT); }
static inline s32 q8_div(s32 a, s32 b) { return (b == 0) ? 0 : (s32)(((s64)a << FIX_SHIFT) / (s64)b); }
static inline s32 min3(s32 a, s32 b, s32 c) { s32 m = (a < b) ? a : b; return (m < c) ? m : c; }
static inline s32 max3(s32 a, s32 b, s32 c) { s32 m = (a > b) ? a : b; return (m > c) ? m : c; }
static inline s32 min32(s32 a, s32 b) { return (a < b) ? a : b; }
static inline s32 max32(s32 a, s32 b) { return (a > b) ? a : b; }
static inline s32 clamp32(s32 v, s32 min_val, s32 max_val) {
    if (v < min_val) return min_val;
    if (v > max_val) return max_val;
    return v;
}

static inline s32 floor_q8_to_int(s32 v) {
    if (v >= 0) return v >> FIX_SHIFT;
    return -(((-v) + FIX_ONE - 1) >> FIX_SHIFT);
}

static inline s64 edge_func(const Vtx2D &a, const Vtx2D &b, s32 px, s32 py) {
    return (s64)(b.x - a.x) * (s64)(py - a.y) - (s64)(b.y - a.y) * (s64)(px - a.x);
}

static inline s32 edge_func32(s32 ax, s32 ay, s32 bx, s32 by, s32 px, s32 py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

static inline int is_top_left(const Vtx2D &a, const Vtx2D &b) {
    s32 dy = b.y - a.y;
    s32 dx = b.x - a.x;
    return (dy > 0) || (dy == 0 && dx < 0);
}

static void init_trig() {
    for (s32 i = 0; i < 360; ++i) {
        double rad = i * 3.14159265358979323846 / 180.0;
        g_sin_tab[i] = (s32)(sin(rad) * FIX_ONE);
        g_cos_tab[i] = (s32)(cos(rad) * FIX_ONE);
    }
}

static int count_bits(unsigned long mask) {
    int c = 0;
    while (mask) { c += (mask & 1UL) ? 1 : 0; mask >>= 1; }
    return c;
}

static int mask_shift(unsigned long mask) {
    int s = 0;
    if (mask == 0) return 0;
    while ((mask & 1UL) == 0) { mask >>= 1; ++s; }
    return s;
}

static unsigned int pack_color(const ColorPacker &cp, unsigned char r, unsigned char g, unsigned char b) {
    const unsigned int rmax = (cp.rbits >= 32) ? 0xffffffffU : ((1U << cp.rbits) - 1U);
    const unsigned int gmax = (cp.gbits >= 32) ? 0xffffffffU : ((1U << cp.gbits) - 1U);
    const unsigned int bmax = (cp.bbits >= 32) ? 0xffffffffU : ((1U << cp.bbits) - 1U);
    unsigned int rp = (cp.rbits == 0) ? 0U : ((unsigned int)r * rmax + 127U) / 255U;
    unsigned int gp = (cp.gbits == 0) ? 0U : ((unsigned int)g * gmax + 127U) / 255U;
    unsigned int bp = (cp.bbits == 0) ? 0U : ((unsigned int)b * bmax + 127U) / 255U;
    return ((rp << cp.rshift) & cp.rmask) | ((gp << cp.gshift) & cp.gmask) | ((bp << cp.bshift) & cp.bmask);
}

static void make_texture_8(uint8_t *tex, const ColorPacker &cp) {
    unsigned char colors[12][3] = {
        {240, 80, 80}, {80, 20, 20},
        {80, 240, 80}, {20, 80, 20},
        {80, 80, 240}, {20, 20, 80},
        {240, 240, 80}, {120, 120, 20},
        {80, 240, 240}, {20, 80, 80},
        {240, 80, 240}, {120, 20, 120}
    };
    for (int i = 0; i < 12; ++i) {
        g_palette[i] = pack_color(cp, colors[i][0], colors[i][1], colors[i][2]);
    }
    for (s32 y = 0; y < TH; ++y) {
        for (s32 x = 0; x < TW; ++x) {
            s32 tile_x = x >> 5; s32 tile_y = y >> 5;
            s32 tile_i = tile_y * 3 + tile_x;
            s32 checker = ((x >> 2) & 1) ^ ((y >> 2) & 1);
            tex[y * TW + x] = (uint8_t)(tile_i * 2 + (checker ? 0 : 1));
        }
    }
}

static inline uint8_t sample_texture_8(const uint8_t *tex, s32 u_q8, s32 v_q8) {
    s32 u = u_q8 >> FIX_SHIFT; s32 v = v_q8 >> FIX_SHIFT;
    u = (u < 0) ? 0 : ((u > TW - 1) ? (TW - 1) : u);
    v = (v < 0) ? 0 : ((v > TH - 1) ? (TH - 1) : v);
    return tex[v * TW + u];
}

static inline void setup_affine_attr(const Vtx2D &a, const Vtx2D &b, const Vtx2D &c,
                                     s32 qa, s32 qb, s32 qc, s32 area,
                                     s32 start_x, s32 start_y,
                                     s32 &step_x, s32 &step_y, s32 &start_val) {
    s64 ax = b.x - a.x, ay = b.y - a.y;
    s64 bx = c.x - a.x, by = c.y - a.y;
    s64 dqa = qb - qa, dqb = qc - qa;
    s64 A = dqa * by - dqb * ay, B = dqb * ax - dqa * bx;
    step_x = (s32)((A * FIX_ONE) / area);
    step_y = (s32)((B * FIX_ONE) / area);
    start_val = qa + (s32)((A * (start_x - a.x) + B * (start_y - a.y)) / area);
}

static void rotate_vertex(const Vtx3D &in, Vtx3D &out, s32 ang_x, s32 ang_y) {
    s32 sx = g_sin_tab[ang_x], cx = g_cos_tab[ang_x];
    s32 sy = g_sin_tab[ang_y], cy = g_cos_tab[ang_y];
    s32 x1 = q8_mul(in.x, cy) + q8_mul(in.z, sy);
    s32 z1 = q8_mul(in.z, cy) - q8_mul(in.x, sy);
    out.x = x1;
    out.y = q8_mul(in.y, cx) - q8_mul(z1, sx);
    out.z = q8_mul(in.y, sx) + q8_mul(z1, cx);
}

static inline s32 compute_w(s32 z_q8) {
    return (32767 * 256) / z_q8;
}

static void project_vertex(const Vtx3D &in, Vtx2D &out, s32 u, s32 v) {
    const s32 cam_z = 5 * FIX_ONE;
    const s32 cx = (W / 2) * FIX_ONE, cy = (H / 2) * FIX_ONE;
    s32 z = in.z + cam_z;
    if (z < (FIX_ONE / 4)) z = FIX_ONE / 4;
    out.x = cx + q8_div(q8_mul(in.x, g_focal), z);
    out.y = cy - q8_div(q8_mul(in.y, g_focal), z);
    out.w = compute_w(z);
    out.u = u;
    out.v = v;
}

static void rasterize_triangle_scalar(
    const TriData &td, const uint8_t *texture_8,
    uint8_t *cbuf, uint16_t *zbuf, int buf_w,
    int tx, int ty, int t_min_x, int t_max_x, int t_min_y, int t_max_y)
{
    int start_x = t_min_x & ~3;

    for (s32 y = t_min_y; y <= t_max_y; ++y) {
        s32 e_ab_row = td.e_ab_origin + start_x * td.e_ab_step_x + y * td.e_ab_step_y;
        s32 e_bc_row = td.e_bc_origin + start_x * td.e_bc_step_x + y * td.e_bc_step_y;
        s32 e_ca_row = td.e_ca_origin + start_x * td.e_ca_step_x + y * td.e_ca_step_y;

        s32 w_row = td.w_origin + start_x * td.dwdx + y * td.dwdy;
        s32 u_row = td.u_origin + start_x * td.dudx + y * td.dudy;
        s32 v_row = td.v_origin + start_x * td.dvdx + y * td.dvdy;

        uint16_t *zbuf_line = &zbuf[(y - ty) * buf_w];
        uint8_t *cbuf_line = &cbuf[(y - ty) * buf_w];

        for (s32 x = start_x; x <= t_max_x; ++x) {
            if (e_ab_row >= 0 && e_bc_row >= 0 && e_ca_row >= 0) {
                int local_x = x - tx;
                uint16_t w_val_16 = (uint16_t)clamp32(w_row, 0, 65535);

                if (w_val_16 > zbuf_line[local_x]) {
                    zbuf_line[local_x] = w_val_16;
                    cbuf_line[local_x] = sample_texture_8(texture_8, u_row, v_row);
                }
            }

            e_ab_row += td.e_ab_step_x;
            e_bc_row += td.e_bc_step_x;
            e_ca_row += td.e_ca_step_x;
            w_row += td.dwdx;
            u_row += td.dudx;
            v_row += td.dvdx;
        }
    }
}

// ---------- DRM/GBM/EGL presentation thread ----------
struct DRMContext {
    int fd;
    drmModeConnector *connector;
    drmModeEncoder *encoder;
    drmModeCrtc *original_crtc;
    uint32_t crtc_id;
    uint32_t connector_id;
    uint32_t mode_width, mode_height;
    drmModeModeInfo mode;
    gbm_device *gbm_dev;
    gbm_surface *gbm_surf;
    EGLDisplay egl_dpy;
    EGLContext egl_ctx;
    EGLSurface egl_surf;
    EGLConfig egl_cfg;
    GLuint texture;
    GLuint program;
    GLuint vbo;
    gbm_bo *current_bo;
    uint32_t current_fb_id;
};

static pthread_mutex_t g_present_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_present_cond = PTHREAD_COND_INITIALIZER;
static uint8_t *g_present_fb = NULL;
static bool g_new_frame_ready = false;
static bool g_present_quit = false;
static unsigned int g_palette_copy[256];

// Vertex shader source
static const char* vs_source =
    "attribute vec4 a_position;\n"
    "attribute vec2 a_texcoord;\n"
    "varying vec2 v_texcoord;\n"
    "void main() {\n"
    "    gl_Position = a_position;\n"
    "    v_texcoord = a_texcoord;\n"
    "}\n";

// Fragment shader source
static const char* fs_source =
    "precision mediump float;\n"
    "varying vec2 v_texcoord;\n"
    "uniform sampler2D u_texture;\n"
    "void main() {\n"
    "    gl_FragColor = texture2D(u_texture, v_texcoord);\n"
    "}\n";

static GLuint load_shader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        char log[256];
        glGetShaderInfoLog(shader, sizeof(log), NULL, log);
        fprintf(stderr, "Shader compile error: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint create_program() {
    GLuint vs = load_shader(GL_VERTEX_SHADER, vs_source);
    GLuint fs = load_shader(GL_FRAGMENT_SHADER, fs_source);
    if (!vs || !fs) return 0;
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint linked;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[256];
        glGetProgramInfoLog(prog, sizeof(log), NULL, log);
        fprintf(stderr, "Program link error: %s\n", log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

static bool init_drm_gbm_egl(DRMContext *ctx) {
    // Open DRM device
    ctx->fd = drmOpen("vc4", NULL);  // For Raspberry Pi, vc4 driver
    if (ctx->fd < 0) {
        ctx->fd = drmOpen("drm", NULL); // fallback
    }
    if (ctx->fd < 0) {
        fprintf(stderr, "Cannot open DRM device\n");
        return false;
    }

    // Get resources
    drmModeRes *res = drmModeGetResources(ctx->fd);
    if (!res) {
        fprintf(stderr, "Cannot get DRM resources\n");
        return false;
    }

    // Find connected connector
    ctx->connector = NULL;
    for (int i = 0; i < res->count_connectors; ++i) {
        drmModeConnector *conn = drmModeGetConnector(ctx->fd, res->connectors[i]);
        if (conn && conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
            ctx->connector = conn;
            break;
        }
        if (conn) drmModeFreeConnector(conn);
    }
    if (!ctx->connector) {
        fprintf(stderr, "No connected connector found\n");
        drmModeFreeResources(res);
        return false;
    }

    // Use first mode
    ctx->mode = ctx->connector->modes[0];
    ctx->mode_width = ctx->mode.hdisplay;
    ctx->mode_height = ctx->mode.vdisplay;

    // Find encoder
    ctx->encoder = NULL;
    for (int i = 0; i < res->count_encoders; ++i) {
        drmModeEncoder *enc = drmModeGetEncoder(ctx->fd, res->encoders[i]);
        if (enc && enc->encoder_id == ctx->connector->encoder_id) {
            ctx->encoder = enc;
            break;
        }
        if (enc) drmModeFreeEncoder(enc);
    }
    if (!ctx->encoder) {
        fprintf(stderr, "No matching encoder\n");
        drmModeFreeConnector(ctx->connector);
        drmModeFreeResources(res);
        return false;
    }

    // Save original CRTC
    ctx->crtc_id = ctx->encoder->crtc_id;
    ctx->original_crtc = drmModeGetCrtc(ctx->fd, ctx->crtc_id);
    ctx->connector_id = ctx->connector->connector_id;

    drmModeFreeResources(res);

    // Create GBM device
    ctx->gbm_dev = gbm_create_device(ctx->fd);
    if (!ctx->gbm_dev) {
        fprintf(stderr, "Cannot create GBM device\n");
        return false;
    }

    // Create GBM surface
    ctx->gbm_surf = gbm_surface_create(ctx->gbm_dev,
                                       ctx->mode_width, ctx->mode_height,
                                       GBM_FORMAT_XRGB8888,
                                       GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
    if (!ctx->gbm_surf) {
        fprintf(stderr, "Cannot create GBM surface\n");
        return false;
    }

    // EGL initialization
    ctx->egl_dpy = eglGetDisplay((EGLNativeDisplayType)ctx->gbm_dev);
    if (ctx->egl_dpy == EGL_NO_DISPLAY) {
        fprintf(stderr, "eglGetDisplay failed\n");
        return false;
    }
    EGLint major, minor;
    if (!eglInitialize(ctx->egl_dpy, &major, &minor)) {
        fprintf(stderr, "eglInitialize failed\n");
        return false;
    }

    // Choose config
    EGLint attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 0,
        EGL_DEPTH_SIZE, 0,
        EGL_STENCIL_SIZE, 0,
        EGL_NONE
    };
    EGLint num_configs;
    if (!eglChooseConfig(ctx->egl_dpy, attribs, &ctx->egl_cfg, 1, &num_configs) || num_configs == 0) {
        fprintf(stderr, "eglChooseConfig failed\n");
        return false;
    }

    // Create context
    EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    ctx->egl_ctx = eglCreateContext(ctx->egl_dpy, ctx->egl_cfg, EGL_NO_CONTEXT, ctx_attribs);
    if (ctx->egl_ctx == EGL_NO_CONTEXT) {
        fprintf(stderr, "eglCreateContext failed\n");
        return false;
    }

    // Create EGL surface from GBM surface
    ctx->egl_surf = eglCreateWindowSurface(ctx->egl_dpy, ctx->egl_cfg, (EGLNativeWindowType)ctx->gbm_surf, NULL);
    if (ctx->egl_surf == EGL_NO_SURFACE) {
        fprintf(stderr, "eglCreateWindowSurface failed\n");
        return false;
    }

    if (!eglMakeCurrent(ctx->egl_dpy, ctx->egl_surf, ctx->egl_surf, ctx->egl_ctx)) {
        fprintf(stderr, "eglMakeCurrent failed\n");
        return false;
    }

    // Create texture
    glGenTextures(1, &ctx->texture);
    glBindTexture(GL_TEXTURE_2D, ctx->texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Create shader program
    ctx->program = create_program();
    if (!ctx->program) return false;
    glUseProgram(ctx->program);

    // Vertex attributes
    GLint pos_loc = glGetAttribLocation(ctx->program, "a_position");
    GLint tex_loc = glGetAttribLocation(ctx->program, "a_texcoord");
    glEnableVertexAttribArray(pos_loc);
    glEnableVertexAttribArray(tex_loc);

    // VBO: 4 vertices (x,y) and 4 texcoords (u,v)
    static const float quad_verts[] = {
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 0.0f,
    };
    glGenBuffers(1, &ctx->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, ctx->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW);
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glVertexAttribPointer(tex_loc, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    return true;
}

static void cleanup_drm_gbm_egl(DRMContext *ctx) {
    if (ctx->current_bo) {
        gbm_surface_release_buffer(ctx->gbm_surf, ctx->current_bo);
        if (ctx->current_fb_id) drmModeRmFB(ctx->fd, ctx->current_fb_id);
    }
    if (ctx->vbo) glDeleteBuffers(1, &ctx->vbo);
    if (ctx->program) glDeleteProgram(ctx->program);
    if (ctx->texture) glDeleteTextures(1, &ctx->texture);
    eglMakeCurrent(ctx->egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroySurface(ctx->egl_dpy, ctx->egl_surf);
    eglDestroyContext(ctx->egl_dpy, ctx->egl_ctx);
    eglTerminate(ctx->egl_dpy);
    if (ctx->gbm_surf) gbm_surface_destroy(ctx->gbm_surf);
    if (ctx->gbm_dev) gbm_device_destroy(ctx->gbm_dev);
    // Restore original CRTC
    if (ctx->original_crtc) {
        drmModeSetCrtc(ctx->fd, ctx->original_crtc->crtc_id,
                       ctx->original_crtc->buffer_id,
                       ctx->original_crtc->x, ctx->original_crtc->y,
                       &ctx->connector_id, 1, &ctx->original_crtc->mode);
        drmModeFreeCrtc(ctx->original_crtc);
    }
    if (ctx->encoder) drmModeFreeEncoder(ctx->encoder);
    if (ctx->connector) drmModeFreeConnector(ctx->connector);
    if (ctx->fd >= 0) close(ctx->fd);
}

static void present_frame(DRMContext *ctx, const uint8_t *fb, const unsigned int *palette) {
    // Convert 8-bit indexed to RGB (24-bit)
    static uint8_t rgb_buffer[W * H * 3];
    for (int i = 0; i < W * H; ++i) {
        unsigned int col = palette[fb[i]];
        rgb_buffer[i*3 + 0] = (col >> 16) & 0xFF;
        rgb_buffer[i*3 + 1] = (col >> 8) & 0xFF;
        rgb_buffer[i*3 + 2] = col & 0xFF;
    }

    // Update texture
    glBindTexture(GL_TEXTURE_2D, ctx->texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_buffer);

    // Set viewport for 1:1 aspect ratio centered
    int scr_w = ctx->mode_width;
    int scr_h = ctx->mode_height;
    int side = (scr_w < scr_h) ? scr_w : scr_h;
    int xoff = (scr_w - side) / 2;
    int yoff = (scr_h - side) / 2;
    glViewport(xoff, yoff, side, side);
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw quad
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    eglSwapBuffers(ctx->egl_dpy, ctx->egl_surf);

    // Get the GBM buffer that was just displayed and set it as scanout via DRM
    struct gbm_bo *bo = gbm_surface_lock_front_buffer(ctx->gbm_surf);
    if (!bo) return;

    uint32_t fb_id;
    uint32_t handles[4] = { gbm_bo_get_handle(bo).u32, 0, 0, 0 };
    uint32_t pitches[4] = { gbm_bo_get_stride(bo), 0, 0, 0 };
    uint32_t offsets[4] = { 0, 0, 0, 0 };
    if (drmModeAddFB2(ctx->fd, ctx->mode_width, ctx->mode_height,
                      DRM_FORMAT_XRGB8888, handles, pitches, offsets, &fb_id, 0) != 0) {
        fprintf(stderr, "drmModeAddFB2 failed\n");
        gbm_surface_release_buffer(ctx->gbm_surf, bo);
        return;
    }

    // Set CRTC to display this framebuffer
    if (drmModeSetCrtc(ctx->fd, ctx->crtc_id, fb_id, 0, 0,
                       &ctx->connector_id, 1, &ctx->mode) != 0) {
        fprintf(stderr, "drmModeSetCrtc failed\n");
        drmModeRmFB(ctx->fd, fb_id);
        gbm_surface_release_buffer(ctx->gbm_surf, bo);
        return;
    }

    // Release previous buffer and framebuffer
    if (ctx->current_bo) {
        gbm_surface_release_buffer(ctx->gbm_surf, ctx->current_bo);
        if (ctx->current_fb_id) drmModeRmFB(ctx->fd, ctx->current_fb_id);
    }
    ctx->current_bo = bo;
    ctx->current_fb_id = fb_id;
}

void* present_thread_func(void* arg) {
    DRMContext *ctx = (DRMContext*)arg;
    uint8_t *local_fb = (uint8_t*)malloc(W * H * sizeof(uint8_t));
    unsigned int local_palette[256];
    memcpy(local_palette, g_palette_copy, sizeof(local_palette));

    while (true) {
        pthread_mutex_lock(&g_present_mutex);
        while (!g_new_frame_ready && !g_present_quit) {
            pthread_cond_wait(&g_present_cond, &g_present_mutex);
        }
        if (g_present_quit) {
            pthread_mutex_unlock(&g_present_mutex);
            break;
        }
        memcpy(local_fb, g_present_fb, W * H * sizeof(uint8_t));
        g_new_frame_ready = false;
        pthread_mutex_unlock(&g_present_mutex);

        present_frame(ctx, local_fb, local_palette);
    }

    free(local_fb);
    return NULL;
}

// ---------- Input via evdev ----------
static int open_keyboard_device() {
    // Try common event devices
    const char* devs[] = { "/dev/input/event0", "/dev/input/event1", "/dev/input/event2", "/dev/input/event3" };
    for (int i = 0; i < 4; ++i) {
        int fd = open(devs[i], O_RDONLY | O_NONBLOCK);
        if (fd >= 0) {
            // Could check device name but just assume keyboard
            return fd;
        }
    }
    return -1;
}

static int read_keyboard_events(int fd, bool &quit) {
    struct input_event ev;
    while (read(fd, &ev, sizeof(ev)) == sizeof(ev)) {
        if (ev.type == EV_KEY && ev.value == 1) { // key press
            if (ev.code == KEY_Q || ev.code == KEY_ESC) {
                quit = true;
                return 1;
            }
        }
    }
    return 0;
}

// ---------- Main ----------
int main() {
    uint8_t *frame_buffer = (uint8_t*)malloc(W * H * sizeof(uint8_t));
    g_present_fb = (uint8_t*)malloc(W * H * sizeof(uint8_t));
    uint8_t *texture_8 = (uint8_t*)malloc(TW * TH * sizeof(uint8_t));

    if (!frame_buffer || !g_present_fb || !texture_8) return 1;
    init_trig();

    double fov_rad = FOV_DEG * M_PI / 180.0;
    double half_fov = fov_rad * 0.5;
    double focal_pixels = (double)(W/2) / tan(half_fov);
    g_focal = (s32)(focal_pixels * FIX_ONE + 0.5);

    // Dummy color packer for palette generation (RGB888)
    ColorPacker packer;
    packer.rmask = 0xFF0000;
    packer.gmask = 0x00FF00;
    packer.bmask = 0x0000FF;
    packer.rshift = 16;
    packer.gshift = 8;
    packer.bshift = 0;
    packer.rbits = 8;
    packer.gbits = 8;
    packer.bbits = 8;
    make_texture_8(texture_8, packer);
    memcpy(g_palette_copy, g_palette, sizeof(g_palette));

    DRMContext drm_ctx;
    memset(&drm_ctx, 0, sizeof(drm_ctx));
    if (!init_drm_gbm_egl(&drm_ctx)) {
        fprintf(stderr, "Failed to initialize DRM/GBM/EGL\n");
        free(frame_buffer);
        free(g_present_fb);
        free(texture_8);
        return 1;
    }

    pthread_t present_tid;
    pthread_create(&present_tid, NULL, present_thread_func, &drm_ctx);

    // Cube vertices (same as original)
    Vtx3D obj[CUBE_VERTS];
    const s32 s = FIX_ONE;
    obj[0].x = -s; obj[0].y = -s; obj[0].z =  s; obj[1].x =  s; obj[1].y = -s; obj[1].z =  s;
    obj[2].x =  s; obj[2].y =  s; obj[2].z =  s; obj[3].x = -s; obj[3].y =  s; obj[3].z =  s;
    obj[4].x =  s; obj[4].y = -s; obj[4].z =  s; obj[5].x =  s; obj[5].y = -s; obj[5].z = -s;
    obj[6].x =  s; obj[6].y =  s; obj[6].z = -s; obj[7].x =  s; obj[7].y =  s; obj[7].z =  s;
    obj[8].x =  s; obj[8].y = -s; obj[8].z = -s; obj[9].x = -s; obj[9].y = -s; obj[9].z = -s;
    obj[10].x = -s; obj[10].y =  s; obj[10].z = -s; obj[11].x =  s; obj[11].y =  s; obj[11].z = -s;
    obj[12].x = -s; obj[12].y = -s; obj[12].z = -s; obj[13].x = -s; obj[13].y = -s; obj[13].z =  s;
    obj[14].x = -s; obj[14].y =  s; obj[14].z =  s; obj[15].x = -s; obj[15].y =  s; obj[15].z = -s;
    obj[16].x = -s; obj[16].y =  s; obj[16].z =  s; obj[17].x =  s; obj[17].y =  s; obj[17].z =  s;
    obj[18].x =  s; obj[18].y =  s; obj[18].z = -s; obj[19].x = -s; obj[19].y =  s; obj[19].z = -s;
    obj[20].x = -s; obj[20].y = -s; obj[20].z = -s; obj[21].x =  s; obj[21].y = -s; obj[21].z = -s;
    obj[22].x =  s; obj[22].y = -s; obj[22].z =  s; obj[23].x = -s; obj[23].y = -s; obj[23].z =  s;

    const s32 tri[TRI_COUNT][3] = {
        {0,1,2}, {0,2,3}, {4,5,6}, {4,6,7}, {8,9,10}, {8,10,11},
        {12,13,14}, {12,14,15}, {16,17,18}, {16,18,19}, {20,21,22}, {20,22,23}
    };

    s32 ang_x = 0, ang_y = 0;
    int running = 1;
    clock_t last_time = clock();
    int frames = 0;
    TriData tdata[TRI_COUNT];
    unsigned int microseconds = 16000;

    int kb_fd = open_keyboard_device();
    fd_set read_fds;

    while (running) {
        // Check keyboard input non-blocking
        if (kb_fd >= 0) {
            FD_ZERO(&read_fds);
            FD_SET(kb_fd, &read_fds);
            struct timeval tv = {0, 0};
            if (select(kb_fd + 1, &read_fds, NULL, NULL, &tv) > 0) {
                read_keyboard_events(kb_fd, running);
            }
        }

        usleep(microseconds);
        ang_x = (ang_x + 2) % 360;
        ang_y = (ang_y + 3) % 360;

        Vtx2D scr[CUBE_VERTS];
        for (s32 i = 0; i < CUBE_VERTS; ++i) {
            Vtx3D r;
            rotate_vertex(obj[i], r, ang_x, ang_y);
            r.z += 3 * FIX_ONE;
            s32 face = i >> 2, local = i & 3;
            s32 tile_x = (face == 0 || face == 3) ? 0 : ((face == 1 || face == 4) ? 1 : 2);
            s32 tile_y = (face < 3) ? 0 : 1;
            s32 du = (local == 1 || local == 2) ? (31 * FIX_ONE) : 0;
            s32 dv = (local == 2 || local == 3) ? (31 * FIX_ONE) : 0;
            project_vertex(r, scr[i], (tile_x * 32 * FIX_ONE) + du, (tile_y * 32 * FIX_ONE) + dv);
        }

        for (int i = 0; i < TRI_COUNT; ++i) {
            Vtx2D a = scr[tri[i][0]], b = scr[tri[i][1]], c = scr[tri[i][2]];
            s64 area = edge_func(a, b, c.x, c.y);
            if (area <= 0) {
                tdata[i].valid = false;
                continue;
            }
            tdata[i].valid = true;
            tdata[i].min_x = max32(0, floor_q8_to_int(min3(a.x, b.x, c.x)));
            tdata[i].max_x = min32(W - 1, floor_q8_to_int(max3(a.x, b.x, c.x)));
            tdata[i].min_y = max32(0, floor_q8_to_int(min3(a.y, b.y, c.y)));
            tdata[i].max_y = min32(H - 1, floor_q8_to_int(max3(a.y, b.y, c.y)));

            s32 spx = FIX_ONE >> 1, spy = FIX_ONE >> 1;
            setup_affine_attr(a, b, c, a.w, b.w, c.w, (s32)area, spx, spy,
                            tdata[i].dwdx, tdata[i].dwdy, tdata[i].w_origin);
            setup_affine_attr(a, b, c, a.u, b.u, c.u, (s32)area, spx, spy,
                            tdata[i].dudx, tdata[i].dudy, tdata[i].u_origin);
            setup_affine_attr(a, b, c, a.v, b.v, c.v, (s32)area, spx, spy,
                            tdata[i].dvdx, tdata[i].dvdy, tdata[i].v_origin);

            const int SUB = 4;
            s32 ax = a.x >> (FIX_SHIFT - SUB), ay = a.y >> (FIX_SHIFT - SUB);
            s32 bx = b.x >> (FIX_SHIFT - SUB), by = b.y >> (FIX_SHIFT - SUB);
            s32 cx = c.x >> (FIX_SHIFT - SUB), cy = c.y >> (FIX_SHIFT - SUB);

            s32 spx4 = 1 << (SUB - 1), spy4 = 1 << (SUB - 1);
            tdata[i].e_ab_origin = edge_func32(ax, ay, bx, by, spx4, spy4) + (is_top_left(a, b) ? 0 : -1);
            tdata[i].e_bc_origin = edge_func32(bx, by, cx, cy, spx4, spy4) + (is_top_left(b, c) ? 0 : -1);
            tdata[i].e_ca_origin = edge_func32(cx, cy, ax, ay, spx4, spy4) + (is_top_left(c, a) ? 0 : -1);

            tdata[i].e_ab_step_x = -(by - ay) << SUB;
            tdata[i].e_ab_step_y = (bx - ax) << SUB;
            tdata[i].e_bc_step_x = -(cy - by) << SUB;
            tdata[i].e_bc_step_y = (cx - bx) << SUB;
            tdata[i].e_ca_step_x = -(ay - cy) << SUB;
            tdata[i].e_ca_step_y = (ax - cx) << SUB;
        }

        memset(frame_buffer, 0, W * H);

        for (int ty = 0; ty < H; ty += TILE_H) {
            for (int tx = 0; tx < W; tx += TILE_W) {
                uint8_t tile_cbuf[TILE_H][TILE_W];
                uint16_t tile_zbuf[TILE_H][TILE_W];

                for (int y = 0; y < TILE_H; ++y) {
                    for (int x = 0; x < TILE_W; ++x) {
                        tile_zbuf[y][x] = 0;
                        tile_cbuf[y][x] = 0;
                    }
                }

                for (int i = 0; i < TRI_COUNT; ++i) {
                    if (!tdata[i].valid) continue;

                    s32 t_min_x = max32(tdata[i].min_x, tx);
                    s32 t_max_x = min32(tdata[i].max_x, tx + TILE_W - 1);
                    s32 t_min_y = max32(tdata[i].min_y, ty);
                    s32 t_max_y = min32(tdata[i].max_y, ty + TILE_H - 1);

                    if (t_min_x > t_max_x || t_min_y > t_max_y) continue;

                    rasterize_triangle_scalar(tdata[i], texture_8,
                                            &tile_cbuf[0][0], &tile_zbuf[0][0], TILE_W,
                                            tx, ty, t_min_x, t_max_x, t_min_y, t_max_y);
                }

                for (int y = 0; y < TILE_H; ++y) {
                    memcpy(&frame_buffer[(ty + y) * W + tx], &tile_cbuf[y][0], TILE_W * sizeof(uint8_t));
                }
            }
        }

        pthread_mutex_lock(&g_present_mutex);
        memcpy(g_present_fb, frame_buffer, W * H * sizeof(uint8_t));
        g_new_frame_ready = true;
        pthread_cond_signal(&g_present_cond);
        pthread_mutex_unlock(&g_present_mutex);

        frames++;
        clock_t now = clock();
        if ((now - last_time) >= CLOCKS_PER_SEC) {
            char title[128];
            sprintf(title, "FPS: %.2f | Scalar w-buffer (DRM/EGL)",
                   frames / ((double)(now - last_time) / CLOCKS_PER_SEC));
            // No X11 title bar, just print to stderr
            fprintf(stderr, "%s\n", title);
            frames = 0;
            last_time = now;
        }
    }

    pthread_mutex_lock(&g_present_mutex);
    g_present_quit = true;
    pthread_cond_signal(&g_present_cond);
    pthread_mutex_unlock(&g_present_mutex);
    pthread_join(present_tid, NULL);

    cleanup_drm_gbm_egl(&drm_ctx);
    if (kb_fd >= 0) close(kb_fd);
    free(frame_buffer);
    free(g_present_fb);
    free(texture_8);
    return 0;
}
