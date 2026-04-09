#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#include <bcm_host.h>
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

static volatile sig_atomic_t g_running = 1;

static pthread_mutex_t g_present_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_present_cond = PTHREAD_COND_INITIALIZER;
static uint8_t* g_present_fb = NULL;
static bool g_new_frame_ready = false;
static bool g_present_quit = false;

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

static void convert_indexed_to_rgb565(const uint8_t *src, unsigned short *dst, int w, int h) {
    for (int i = 0; i < w * h; ++i) {
        unsigned int rgb = g_palette[src[i]];
        unsigned char r = (rgb >> 16) & 0xFF;
        unsigned char g = (rgb >> 8) & 0xFF;
        unsigned char b = rgb & 0xFF;
        dst[i] = (unsigned short)(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
    }
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

static void handle_signal(int) {
    g_running = 0;
}

static const char* kVertexShader = R"GLSL(
attribute vec2 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)GLSL";

static const char* kFragmentShader = R"GLSL(
precision mediump float;
varying vec2 v_uv;
uniform sampler2D u_tex;
void main() {
    gl_FragColor = texture2D(u_tex, v_uv);
}
)GLSL";

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, NULL);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        GLsizei len = 0;
        glGetShaderInfoLog(sh, sizeof(log), &len, log);
        fprintf(stderr, "shader compile failed: %.*s\n", (int)len, log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static GLuint create_program() {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, kVertexShader);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kFragmentShader);
    if (!vs || !fs) return 0;

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "a_pos");
    glBindAttribLocation(prog, 1, "a_uv");
    glLinkProgram(prog);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof(log), &len, log);
        fprintf(stderr, "program link failed: %.*s\n", (int)len, log);
        glDeleteProgram(prog);
        prog = 0;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

struct PresentContext {
    int screen_width;
    int screen_height;
};

static void* present_thread_func(void* arg) {
    PresentContext* ctx = (PresentContext*)arg;

    uint8_t* local_fb = (uint8_t*)malloc(W * H);
    unsigned short* rgb_buffer = (unsigned short*)malloc(W * H * sizeof(unsigned short));
    if (!local_fb || !rgb_buffer) {
        free(local_fb);
        free(rgb_buffer);
        return NULL;
    }

    bcm_host_init();

    DISPMANX_DISPLAY_HANDLE_T dispman_display = vc_dispmanx_display_open(0);
    if (!dispman_display) {
        fprintf(stderr, "vc_dispmanx_display_open failed\n");
        free(local_fb);
        free(rgb_buffer);
        return NULL;
    }

    DISPMANX_UPDATE_HANDLE_T dispman_update = vc_dispmanx_update_start(0);
    VC_RECT_T dst_rect;
    VC_RECT_T src_rect;
    dst_rect.x = 0;
    dst_rect.y = 0;
    dst_rect.width = ctx->screen_width;
    dst_rect.height = ctx->screen_height;
    src_rect.x = 0;
    src_rect.y = 0;
    src_rect.width = ctx->screen_width << 16;
    src_rect.height = ctx->screen_height << 16;

    DISPMANX_ELEMENT_HANDLE_T dispman_element = vc_dispmanx_element_add(
        dispman_update,
        dispman_display,
        0,
        &dst_rect,
        0,
        &src_rect,
        DISPMANX_PROTECTION_NONE,
        NULL,
        NULL,
        DISPMANX_NO_ROTATE);

    EGL_DISPMANX_WINDOW_T nativewindow;
    nativewindow.element = dispman_element;
    nativewindow.width = ctx->screen_width;
    nativewindow.height = ctx->screen_height;

    vc_dispmanx_update_submit_sync(dispman_update);

    EGLDisplay egl_dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_dpy == EGL_NO_DISPLAY) {
        fprintf(stderr, "eglGetDisplay failed\n");
        vc_dispmanx_element_remove(dispman_update, dispman_element);
        vc_dispmanx_display_close(dispman_display);
        free(local_fb);
        free(rgb_buffer);
        return NULL;
    }

    EGLint major = 0, minor = 0;
    if (!eglInitialize(egl_dpy, &major, &minor)) {
        fprintf(stderr, "eglInitialize failed\n");
        vc_dispmanx_display_close(dispman_display);
        free(local_fb);
        free(rgb_buffer);
        return NULL;
    }

    eglBindAPI(EGL_OPENGL_ES_API);

    EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 5,
        EGL_GREEN_SIZE, 6,
        EGL_BLUE_SIZE, 5,
        EGL_ALPHA_SIZE, 0,
        EGL_DEPTH_SIZE, 0,
        EGL_STENCIL_SIZE, 0,
        EGL_NONE
    };

    EGLConfig egl_config = nullptr;
    EGLint num_configs = 0;
    if (!eglChooseConfig(egl_dpy, configAttribs, &egl_config, 1, &num_configs) || num_configs == 0) {
        fprintf(stderr, "eglChooseConfig failed\n");
        eglTerminate(egl_dpy);
        vc_dispmanx_display_close(dispman_display);
        free(local_fb);
        free(rgb_buffer);
        return NULL;
    }

    const EGLint ctxAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    EGLContext egl_ctx = eglCreateContext(egl_dpy, egl_config, EGL_NO_CONTEXT, ctxAttribs);
    if (egl_ctx == EGL_NO_CONTEXT) {
        fprintf(stderr, "eglCreateContext failed\n");
        eglTerminate(egl_dpy);
        vc_dispmanx_display_close(dispman_display);
        free(local_fb);
        free(rgb_buffer);
        return NULL;
    }

    EGLSurface egl_surf = eglCreateWindowSurface(egl_dpy, egl_config, &nativewindow, NULL);
    if (egl_surf == EGL_NO_SURFACE) {
        fprintf(stderr, "eglCreateWindowSurface failed\n");
        eglDestroyContext(egl_dpy, egl_ctx);
        eglTerminate(egl_dpy);
        vc_dispmanx_display_close(dispman_display);
        free(local_fb);
        free(rgb_buffer);
        return NULL;
    }

    if (!eglMakeCurrent(egl_dpy, egl_surf, egl_surf, egl_ctx)) {
        fprintf(stderr, "eglMakeCurrent failed\n");
        eglDestroySurface(egl_dpy, egl_surf);
        eglDestroyContext(egl_dpy, egl_ctx);
        eglTerminate(egl_dpy);
        vc_dispmanx_display_close(dispman_display);
        free(local_fb);
        free(rgb_buffer);
        return NULL;
    }

    GLuint prog = create_program();
    if (!prog) {
        eglMakeCurrent(egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroySurface(egl_dpy, egl_surf);
        eglDestroyContext(egl_dpy, egl_ctx);
        eglTerminate(egl_dpy);
        vc_dispmanx_display_close(dispman_display);
        free(local_fb);
        free(rgb_buffer);
        return NULL;
    }

    glUseProgram(prog);
    GLint pos_loc = glGetAttribLocation(prog, "a_pos");
    GLint uv_loc = glGetAttribLocation(prog, "a_uv");
    GLint tex_loc = glGetUniformLocation(prog, "u_tex");

    const GLfloat vertices[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f,  1.0f
    };
    const GLfloat texcoords[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLuint texture_id = 0;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, W, H, 0, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, NULL);
    glUniform1i(tex_loc, 0);

    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glViewport(0, 0, ctx->screen_width, ctx->screen_height);

    int side = (ctx->screen_width < ctx->screen_height) ? ctx->screen_width : ctx->screen_height;
    int vx = (ctx->screen_width - side) / 2;
    int vy = (ctx->screen_height - side) / 2;

    clock_t last_time = clock();
    int frames = 0;

    while (g_running) {
        pthread_mutex_lock(&g_present_mutex);
        while (!g_new_frame_ready && !g_present_quit && g_running) {
            pthread_cond_wait(&g_present_cond, &g_present_mutex);
        }
        if (g_present_quit || !g_running) {
            pthread_mutex_unlock(&g_present_mutex);
            break;
        }
        memcpy(local_fb, g_present_fb, W * H);
        g_new_frame_ready = false;
        pthread_mutex_unlock(&g_present_mutex);

        convert_indexed_to_rgb565(local_fb, rgb_buffer, W, H);

        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, rgb_buffer);

        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(vx, vy, side, side);
        glUseProgram(prog);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glUniform1i(tex_loc, 0);
        glEnableVertexAttribArray((GLuint)pos_loc);
        glEnableVertexAttribArray((GLuint)uv_loc);
        glVertexAttribPointer((GLuint)pos_loc, 2, GL_FLOAT, GL_FALSE, 0, vertices);
        glVertexAttribPointer((GLuint)uv_loc, 2, GL_FLOAT, GL_FALSE, 0, texcoords);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDisableVertexAttribArray((GLuint)pos_loc);
        glDisableVertexAttribArray((GLuint)uv_loc);
        eglSwapBuffers(egl_dpy, egl_surf);

        frames++;
        clock_t now = clock();
        if ((now - last_time) >= CLOCKS_PER_SEC) {
            double fps = frames / ((double)(now - last_time) / CLOCKS_PER_SEC);
            fprintf(stderr, "FPS: %.2f | EGL fullscreen GLES2\n", fps);
            frames = 0;
            last_time = now;
        }
    }

    glDeleteTextures(1, &texture_id);
    glDeleteProgram(prog);
    eglMakeCurrent(egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroySurface(egl_dpy, egl_surf);
    eglDestroyContext(egl_dpy, egl_ctx);
    eglTerminate(egl_dpy);
    vc_dispmanx_display_close(dispman_display);

    free(local_fb);
    free(rgb_buffer);
    return NULL;
}

int main() {
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    uint8_t *frame_buffer = (uint8_t*)malloc(W * H * sizeof(uint8_t));
    g_present_fb = (uint8_t*)malloc(W * H * sizeof(uint8_t));
    uint8_t *texture_8 = (uint8_t*)malloc(TW * TH * sizeof(uint8_t));

    if (!frame_buffer || !g_present_fb || !texture_8) return 1;
    init_trig();

    double fov_rad = FOV_DEG * M_PI / 180.0;
    double half_fov = fov_rad * 0.5;
    double focal_pixels = (double)(W / 2) / tan(half_fov);
    g_focal = (s32)(focal_pixels * FIX_ONE + 0.5);

    bcm_host_init();

    int screen_w = 0, screen_h = 0;
    if (graphics_get_display_size(0, (unsigned int*)&screen_w, (unsigned int*)&screen_h) < 0) {
        fprintf(stderr, "graphics_get_display_size failed\n");
        return 1;
    }

    ColorPacker packer{};
    packer.rmask = 0xF800;
    packer.gmask = 0x07E0;
    packer.bmask = 0x001F;
    packer.rshift = 11;
    packer.gshift = 5;
    packer.bshift = 0;
    packer.rbits = 5;
    packer.gbits = 6;
    packer.bbits = 5;
    make_texture_8(texture_8, packer);

    PresentContext pctx;
    pctx.screen_width = screen_w;
    pctx.screen_height = screen_h;

    pthread_t present_tid;
    pthread_create(&present_tid, NULL, present_thread_func, &pctx);

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
    clock_t last_time = clock();
    int frames = 0;
    TriData tdata[TRI_COUNT];

    while (g_running) {
        usleep(16000);

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
            fprintf(stderr, "Frames: %d\n", frames);
            frames = 0;
            last_time = now;
        }
    }

    pthread_mutex_lock(&g_present_mutex);
    g_present_quit = true;
    pthread_cond_signal(&g_present_cond);
    pthread_mutex_unlock(&g_present_mutex);
    pthread_join(present_tid, NULL);

    free(frame_buffer);
    free(g_present_fb);
    free(texture_8);
    return 0;
}
