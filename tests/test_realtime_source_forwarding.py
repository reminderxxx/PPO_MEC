import json
from pathlib import Path
import subprocess
import sys
import time

from src.runtime.handoff_state_runtime import send_packet
from src.runtime.workflow_suffix_recovery import canonical_json_bytes


def test_result_forwarding_uses_independent_receiver_and_conserves_packet(tmp_path):
    script=Path(__file__).resolve().parents[1]/"scripts/run_realtime_source_forwarding.py"
    process=subprocess.Popen([sys.executable,str(script),"--receiver",str(tmp_path)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+5
        while not (tmp_path/"port.json").exists():
            assert process.poll() is None
            if time.monotonic()>deadline:raise TimeoutError("receiver initialization")
            time.sleep(.01)
        port=json.loads((tmp_path/"port.json").read_text())
        payload={"kind":"final_result","workflow_id":"test","vehicle_id":"v",
            "n1_input_sha256":"a"*64,"token_ids":[1,2],"decoded_text":"example"}
        packet=canonical_json_bytes(payload)
        sent=send_packet(port["port"],packet,mbps=10,latency_seconds=.001)
        stdout,stderr=process.communicate(timeout=5)
        assert process.returncode==0,stderr
        received=json.loads((tmp_path/"received.json").read_text())
        assert received["result"]==payload
        assert sent["packet_sha256"]==received["transport"]["packet_sha256"]
        assert sent["forward_bytes"]==len(packet)+8
        assert port["pid"]==process.pid
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=5)
