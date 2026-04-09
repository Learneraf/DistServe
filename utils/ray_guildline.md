# Ray Guildline

1. 关闭Ray Head Node:

```bash
pkill -f ray
```

2. 清理临时文件

```bash
rm -rf /users/rh/tmp/ray
rm -rf /users/rh/tmp/session_*
```

3. 重新启动Ray Head Node

```bash
ray start --head --port 6381 --dashboard-port 8267 --temp-dir ~/ray_tmp
```

4. 设置Ray Address

```bash
export RAY_ADDRESS=10.129.165.27:6381
```
