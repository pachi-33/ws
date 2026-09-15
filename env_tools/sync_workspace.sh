#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_NAME="${0##*/}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"
readonly CREDENTIALS_FILE="${SYNC_WORKSPACE_CREDENTIALS_FILE:-$SCRIPT_DIR/sync_workspace.credentials}"

GIT_AUTH_USERNAME="${SYNC_GIT_USERNAME:-}"
GIT_AUTH_PASSWORD="${SYNC_GIT_PASSWORD:-}"

load_plaintext_credentials() {
  local key value
  [[ -f "$CREDENTIALS_FILE" ]] || return 0

  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    value="${value%$'\r'}"
    case "$key" in
      GIT_USERNAME) [[ -n "$GIT_AUTH_USERNAME" ]] || GIT_AUTH_USERNAME="$value" ;;
      GIT_PASSWORD) [[ -n "$GIT_AUTH_PASSWORD" ]] || GIT_AUTH_PASSWORD="$value" ;;
    esac
  done < "$CREDENTIALS_FILE"
}

load_plaintext_credentials

# Git 通过 GIT_ASKPASS 再次调用本脚本时，只输出它请求的凭据。
if [[ "${SYNC_WORKSPACE_ASKPASS:-0}" == "1" ]]; then
  case "${1:-}" in
    *Username*|*username*) printf '%s\n' "$GIT_AUTH_USERNAME" ;;
    *Password*|*password*) printf '%s\n' "$GIT_AUTH_PASSWORD" ;;
    *) exit 1 ;;
  esac
  exit 0
fi

REMOTE="origin"
BRANCH=""
AUTO_COMMIT=0
COMMIT_MESSAGE=""
CLEAN_UNTRACKED=0
CLEAN_IGNORED=0

usage() {
  cat <<EOF
用法:
  $SCRIPT_NAME push  [选项]   强制将本地分支覆盖到远端
  $SCRIPT_NAME reset [选项]   将本地分支硬重置到远端最新版本
  $SCRIPT_NAME status [选项]  显示本地与远端状态

通用选项:
  -r, --remote NAME       远端名称（默认: origin）
  -b, --branch NAME       分支名称（默认: 当前分支）
  -h, --help              显示帮助

push 选项:
  -c, --commit            先暂存全部改动并自动提交，再强制推送
  -m, --message TEXT      自动提交信息（同时启用 --commit）

reset 选项:
      --clean             同时删除未跟踪文件和目录
      --clean-ignored     同时删除未跟踪及被忽略文件（最彻底）

示例:
  $SCRIPT_NAME push
  $SCRIPT_NAME push --commit -m "sync workspace"
  $SCRIPT_NAME reset
  $SCRIPT_NAME reset --clean

认证:
  优先读取同目录的 sync_workspace.credentials，也支持环境变量：
    SYNC_GIT_USERNAME=用户名
    SYNC_GIT_PASSWORD=密码或 GitHub PAT
  未配置上述凭据时，自动复用 SSH Agent、Git Credential Manager 或 macOS Keychain。
EOF
}

die() {
  printf '错误: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '==> %s\n' "$*"
}

require_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "选项 $1 缺少参数"
}

configure_auth() {
  local remote_url helper
  remote_url="$(git remote get-url "$REMOTE")"

  case "$remote_url" in
    https://*)
      if [[ -n "$GIT_AUTH_PASSWORD" ]]; then
        [[ -n "$GIT_AUTH_USERNAME" ]] || die "已配置 GIT_PASSWORD，但 GIT_USERNAME 为空"
        if [[ -f "$CREDENTIALS_FILE" ]]; then
          chmod 600 "$CREDENTIALS_FILE" 2>/dev/null \
            || die "无法将凭据文件权限设为 600: $CREDENTIALS_FILE"
          git check-ignore -q "$CREDENTIALS_FILE" \
            || die "凭据文件没有被 Git 忽略，已拒绝继续以防泄露: $CREDENTIALS_FILE"
        fi
        export SYNC_WORKSPACE_ASKPASS=1
        export SYNC_GIT_USERNAME="$GIT_AUTH_USERNAME"
        export SYNC_GIT_PASSWORD="$GIT_AUTH_PASSWORD"
        export GIT_ASKPASS="$SCRIPT_PATH"
        info "使用明文凭据文件进行免交互认证"
      else
        helper="$(git config --get credential.helper || true)"
        if [[ -z "$helper" && "$(uname -s)" == "Darwin" ]]; then
          git config credential.helper osxkeychain
          info "已为当前仓库启用 macOS Keychain 凭据助手"
        fi
      fi
      ;;
    git@*|ssh://*)
      [[ -n "${SSH_AUTH_SOCK:-}" ]] || info "提示: SSH Agent 未检测到，将使用本机 SSH 配置"
      ;;
  esac

  # 禁止 Git 弹出交互式用户名/密码输入；认证应由安全凭据助手自动完成。
  export GIT_TERMINAL_PROMPT=0
}

resolve_branch() {
  local checked_out
  checked_out="$(git branch --show-current)"
  [[ -n "$checked_out" ]] || die "当前处于 detached HEAD，请先检出一个本地分支"
  if [[ -z "$BRANCH" ]]; then
    BRANCH="$checked_out"
  fi
  git check-ref-format --branch "$BRANCH" >/dev/null 2>&1 || die "无效分支名: $BRANCH"
  [[ "$BRANCH" == "$checked_out" ]] \
    || die "为避免重置错误分支，请先执行: git switch $BRANCH"
}

fetch_remote() {
  info "获取 $REMOTE/$BRANCH 的最新状态"
  git fetch --prune "$REMOTE" "+refs/heads/$BRANCH:refs/remotes/$REMOTE/$BRANCH"
}

command_push() {
  if (( AUTO_COMMIT )); then
    git add -A
    if git diff --cached --quiet; then
      info "没有需要提交的改动"
    else
      if [[ -z "$COMMIT_MESSAGE" ]]; then
        COMMIT_MESSAGE="sync workspace $(date '+%Y-%m-%d %H:%M:%S')"
      fi
      git commit -m "$COMMIT_MESSAGE"
    fi
  fi

  fetch_remote
  info "强制推送本地 $BRANCH 到 $REMOTE/$BRANCH"
  # force-with-lease 仍是强制推送，但能避免覆盖 fetch 之后刚产生的新远端提交。
  git push --force-with-lease="refs/heads/$BRANCH:$(git rev-parse "refs/remotes/$REMOTE/$BRANCH")" \
    "$REMOTE" "refs/heads/$BRANCH:refs/heads/$BRANCH"
  info "推送完成"
}

command_reset() {
  fetch_remote
  info "硬重置本地 $BRANCH 到 $REMOTE/$BRANCH（本地已跟踪改动将丢失）"
  git reset --hard "refs/remotes/$REMOTE/$BRANCH"

  if (( CLEAN_IGNORED )); then
    info "删除未跟踪及被忽略的文件和目录（保留同步凭据）"
    git clean -fdx -e /env_tools/sync_workspace.credentials
  elif (( CLEAN_UNTRACKED )); then
    info "删除未跟踪文件和目录"
    git clean -fd
  fi
  info "重置完成"
}

command_status() {
  fetch_remote
  git status --short --branch
  printf '\n本地 HEAD:  %s\n' "$(git rev-parse --short HEAD)"
  printf '远端 HEAD:  %s\n' "$(git rev-parse --short "refs/remotes/$REMOTE/$BRANCH")"
}

main() {
  [[ $# -gt 0 ]] || { usage; exit 2; }
  local action="$1"
  shift

  case "$action" in
    push|reset|status) ;;
    -h|--help|help) usage; exit 0 ;;
    *) die "未知操作: $action（使用 --help 查看帮助）" ;;
  esac

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -r|--remote)
        require_value "$@"; REMOTE="$2"; shift 2 ;;
      -b|--branch)
        require_value "$@"; BRANCH="$2"; shift 2 ;;
      -c|--commit)
        [[ "$action" == "push" ]] || die "$1 只适用于 push"
        AUTO_COMMIT=1; shift ;;
      -m|--message)
        [[ "$action" == "push" ]] || die "$1 只适用于 push"
        require_value "$@"; AUTO_COMMIT=1; COMMIT_MESSAGE="$2"; shift 2 ;;
      --clean)
        [[ "$action" == "reset" ]] || die "$1 只适用于 reset"
        CLEAN_UNTRACKED=1; shift ;;
      --clean-ignored)
        [[ "$action" == "reset" ]] || die "$1 只适用于 reset"
        CLEAN_IGNORED=1; shift ;;
      -h|--help)
        usage; exit 0 ;;
      --)
        shift; [[ $# -eq 0 ]] || die "存在多余参数: $*" ;;
      *) die "未知选项: $1" ;;
    esac
  done

  local repo_root
  repo_root="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)" \
    || die "脚本所在目录不属于 Git 仓库: $SCRIPT_DIR"
  cd -- "$repo_root"

  git remote get-url "$REMOTE" >/dev/null 2>&1 || die "远端不存在: $REMOTE"
  resolve_branch
  git show-ref --verify --quiet "refs/heads/$BRANCH" || die "本地分支不存在: $BRANCH"
  configure_auth

  case "$action" in
    push) command_push ;;
    reset) command_reset ;;
    status) command_status ;;
  esac
}

main "$@"
