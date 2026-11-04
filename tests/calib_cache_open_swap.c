/* After the first successful open of WATCH_PATH, rename REPLACEMENT_PATH
 * onto it. The fd already returned still refers to the original inode, so
 * the bytes read from that fd stay the bytes that were opened. A later
 * open of the same path sees the replacement. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int swapped = 0;

static int (*real_open)(const char *, int, ...);
static int (*real_open64)(const char *, int, ...);
static int (*real_openat)(int, const char *, int, ...);
static int (*real_openat64)(int, const char *, int, ...);

static int watched(const char *path)
{
    const char *watch = getenv("WATCH_PATH");
    return watch != NULL && path != NULL && strcmp(path, watch) == 0;
}

static void swap_once(void)
{
    const char *replacement;
    const char *watch;
    if (swapped)
        return;
    replacement = getenv("REPLACEMENT_PATH");
    watch = getenv("WATCH_PATH");
    if (replacement == NULL || watch == NULL)
        return;
    swapped = 1;
    if (rename(replacement, watch) != 0)
        dprintf(STDERR_FILENO, "cache swap failed: %s\n", strerror(errno));
}

static int finish(int fd, const char *path)
{
    if (fd >= 0 && watched(path))
        swap_once();
    return fd;
}

int open(const char *path, int flags, ...)
{
    int fd;
    if (real_open == NULL)
        real_open = dlsym(RTLD_NEXT, "open");
    if (flags & O_CREAT) {
        va_list ap;
        mode_t mode;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
        fd = real_open(path, flags, mode);
    } else {
        fd = real_open(path, flags);
    }
    return finish(fd, path);
}

int open64(const char *path, int flags, ...)
{
    int fd;
    if (real_open64 == NULL)
        real_open64 = dlsym(RTLD_NEXT, "open64");
    if (flags & O_CREAT) {
        va_list ap;
        mode_t mode;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
        fd = real_open64(path, flags, mode);
    } else {
        fd = real_open64(path, flags);
    }
    return finish(fd, path);
}

int openat(int dirfd, const char *path, int flags, ...)
{
    int fd;
    if (real_openat == NULL)
        real_openat = dlsym(RTLD_NEXT, "openat");
    if (flags & O_CREAT) {
        va_list ap;
        mode_t mode;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
        fd = real_openat(dirfd, path, flags, mode);
    } else {
        fd = real_openat(dirfd, path, flags);
    }
    return finish(fd, path);
}

int openat64(int dirfd, const char *path, int flags, ...)
{
    int fd;
    if (real_openat64 == NULL)
        real_openat64 = dlsym(RTLD_NEXT, "openat64");
    if (flags & O_CREAT) {
        va_list ap;
        mode_t mode;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
        fd = real_openat64(dirfd, path, flags, mode);
    } else {
        fd = real_openat64(dirfd, path, flags);
    }
    return finish(fd, path);
}
