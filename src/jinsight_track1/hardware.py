import json, platform, shutil, subprocess, sys
def inventory():
    vm=shutil.disk_usage("."); return {"collected_at":__import__('datetime').datetime.now().isoformat(),"os":platform.platform(),"python":sys.version,"cpu":{"logical":__import__('os').cpu_count()},"memory":None,"gpus":[],"nvidia_driver":None,"cuda":None,"torch":None,"storage":{"free_bytes":vm.free,"total_bytes":vm.total},"network_checks":{},"data_inventory":None,"measurement_notes":["Memory/GPU and network values are null unless separately measured."]}
def as_json(): return json.dumps(inventory(),indent=2)
