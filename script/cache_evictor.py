#!/usr/bin/env python3
# 我们战役大文件的页缓存周期逐出(容器内无 drop_caches 权限的替代方案)
# 只对 yuanqi_ops 自有目录的大文件做 posix_fadvise(DONTNEED),不碰其他用户
import os, sys, time
ROOTS = ["/opt/data/private/yuanqi_ops/data", "/opt/data/private/yuanqi_ops/logs", "/opt/data/private/hechengzhu/ZHY/data/experiments", "/opt/data/private/hechengzhu/ZHY/data/training"]
MIN_BYTES = 50 * 1024 * 1024
def evict():
    total = 0
    for root in ROOTS:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                p = os.path.join(dp, fn)
                try:
                    if os.path.getsize(p) >= MIN_BYTES:
                        fd = os.open(p, os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                        total += 1
                except OSError:
                    pass
    return total
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        print("evicted files:", evict())
    else:
        while True:
            n = evict()
            print(time.strftime("%F %T"), "evicted:", n, flush=True)
            time.sleep(7200)
