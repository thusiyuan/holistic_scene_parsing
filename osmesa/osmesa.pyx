cdef extern void render_image()
cdef extern int init_context(int w, int h)
cdef extern void free_context(const char *name)

cdef extern void sph(double angle)
cdef extern void Sphere(float radius, int slices, int stacks)

def render():
    render_image()

def init_ctx(int width, int height):
    r = init_context(width, height)
    if r == -1:
        raise Exception("init_context failed")

def free_ctx(str name):
    free_context(name)

def sphere(angle):
    sph(angle)

def sphere2(float radius, int slices, int stacks):
    Sphere(radius, slices, stacks)

