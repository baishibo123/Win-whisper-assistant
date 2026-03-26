#!/usr/bin/env bash
input=$(cat)
cwd=$(echo "$input" | jq -r '.cwd')
chroot_val="${debian_chroot:-}"
chroot_part=""
[ -n "$chroot_val" ] && chroot_part="($chroot_val)"
printf "%s\033[01;32m%s@%s\033[00m:\033[01;34m%s\033[00m" \
  "$chroot_part" \
  "$(whoami)" \
  "$(hostname -s)" \
  "$cwd"
