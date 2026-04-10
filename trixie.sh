g++ -O3 -march=armv6zk -mfpu=vfp -mfloat-abi=hard -mtune=arm1176jzf-s \
    -ffast-math -funroll-loops -fomit-frame-pointer \
    main.cpp \
    -o cube_drm \
    -ldrm -lgbm -lEGL -lGLESv2 -lm -lpthread \
    -std=c++98
