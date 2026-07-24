#!/usr/bin/env bash
# The host driver is loaded, but this managed session does not persist the
# NVIDIA character devices in ordinary shells. Run GPU commands through this
# wrapper in the same privileged device session.
set -euo pipefail

for bdf in 1a 1b 3d 3e 88 89 b1; do
  udevadm trigger --action=add "/sys/bus/pci/devices/0000:${bdf}:00.0"
done
udevadm trigger --action=add --subsystem-match=misc || true
udevadm settle --timeout=10 || true

if ! nvidia-smi -L >/dev/null; then
  echo "NVIDIA devices are still unavailable after udev trigger" >&2
  exit 2
fi
exec "$@"
