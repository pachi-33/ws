git信息
username:pachi33
passwd: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
# 警告：此文件保存明文凭据。脚本会自动把权限收紧为 600。
# GitHub 的 GIT_PASSWORD 必须填写 Personal Access Token，而不是账户登录密码。
GIT_USERNAME=pachi-33
GIT_PASSWORD=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


统一工作目录：/home/y00951466

| server                             | 描述             | 容器                  |     |
| ---------------------------------- | -------------- | ------------------- | --- |
| root:SGLangNB2026@141.61.19.134:22 | 一体机（950PR * 8） | y00951466_test_0914 |     |
| root:Huawei12#$@141.61.45.113:22   | 950DT * 8      |                     |     |
| root:Huawei12#$@141.61.45.113:22   | 950DT * 8      |                     |     |
| root:Admin%408088@141.61.73.103:22 | 950DT * 8      |                     |     |
|                                    |                |                     |     |
|                                    |                |                     |     |
|                                    |                |                     |     |
|                                    |                |                     |     |
|                                    |                |                     |     |

sglang推理环境最新镜像：https://ascend.github.io/sglang/


同步本空间：

# 强制推送已有提交
env_tools/sync_workspace.sh push

# 自动提交全部改动并强制推送
env_tools/sync_workspace.sh push --commit -m "sync workspace"

# 强制重置到远端最新
env_tools/sync_workspace.sh reset

# 同时删除未跟踪文件
env_tools/sync_workspace.sh reset --clean

# 删除所有未跟踪及忽略文件，但保留凭据
env_tools/sync_workspace.sh reset --clean-ignored



bash <你的启动脚本> 2>&1 |
  tee -a /home/y00951466/logs/sglang_$(date +%Y%m%d_%H%M%S).log