# Vendored from unitreerobotics/xr_teleoperate (Apache-2.0) via the user's
# G1_teleop_dex fork: teleop/utils/motion_switcher.py. See NOTICE.
# On hardware, call Enter_Debug_Mode() BEFORE any rt/lowcmd publishing so no
# locomotion/AI mode fights the low-level arm commands on the suspended robot.
# for motion switcher
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
# for loco client
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
import time

def ensure_debug_mode(verbose: bool = True, client_factory=None,
                      timeout_s: float = 10.0, poll_s: float = 1.0):
    """Check-first debug-mode entry -> (ok, msg). Called by Robot.connect("real").

    CheckMode FIRST: if no high-level mode is active (result['name'] empty) the robot
    is ALREADY in debug mode and we return immediately WITHOUT calling ReleaseMode --
    releasing on a robot the operator already put in debug mode drops it back OUT of
    low-level control (hardware-observed: rt/lowcmd then publishes into the void and
    the arms never move). Only when a named mode (ai/loco/...) is active do we
    ReleaseMode + re-check in a bounded loop. The vendored MotionSwitcher class below
    (Enter_Debug_Mode's blind release-while loop) is exactly that footgun -- kept
    verbatim for provenance; use this function instead.

    ``client_factory`` (tests) overrides the default MotionSwitcherClient construction;
    DDS must already be initialised (ChannelFactoryInitialize) before the default path.
    Never raises: any RPC/DDS failure returns (False, msg).
    """
    def _say(m):
        if verbose:
            print(f"[motion_switcher] {m}")
    try:
        if client_factory is None:
            # lazy so a fake factory never needs the SDK path exercised
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient as _MSC)

            def client_factory():
                c = _MSC()
                c.SetTimeout(1.0)
                c.Init()
                return c
        msc = client_factory()
        _status, result = msc.CheckMode()
        name = (result or {}).get("name") or ""
        if not name:
            msg = "already in debug mode (untouched)"
            _say(msg)
            return True, msg
        first = name
        _say(f"active high-level mode '{first}' -> releasing (up to {timeout_s:.0f}s)")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            msc.ReleaseMode()
            time.sleep(poll_s)
            _status, result = msc.CheckMode()
            name = (result or {}).get("name") or ""
            if not name:
                msg = f"released '{first}' -> now in debug mode"
                _say(msg)
                return True, msg
        msg = f"mode '{name}' still active after {timeout_s:.0f}s -- NOT in debug mode"
        _say(msg)
        return False, msg
    except Exception as e:  # noqa: BLE001 - never let a mode check kill the connect
        msg = f"MotionSwitcher unavailable ({e!r}) -- cannot confirm debug mode"
        _say(msg)
        return False, msg


# MotionSwitcher used to switch mode between debug mode and ai mode
class MotionSwitcher:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(1.0)
        self.msc.Init()

    def Enter_Debug_Mode(self):
        try:
            status, result = self.msc.CheckMode()
            while result['name']:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)
            return status, result
        except Exception as e:
            return None, None
    
    def Exit_Debug_Mode(self):
        try:
            status, result = self.msc.SelectMode(nameOrAlias='ai')
            return status, result
        except Exception as e:
            return None, None

    def CheckMode(self):
        """Return (status, result); result['name'] empty => no mode active."""
        return self.msc.CheckMode()

class LocoClientWrapper:
    def __init__(self):
        self.client = LocoClient()
        self.client.SetTimeout(0.0001)
        self.client.Init()

    def Enter_Damp_Mode(self):
        self.client.Damp()
    
    def Move(self, vx, vy, vyaw):
        self.client.Move(vx, vy, vyaw, continous_move=False)

if __name__ == '__main__':
    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation
    ms = MotionSwitcher()
    status, result = ms.Enter_Debug_Mode()
    print("Enter debug mode:", status, result)
    time.sleep(5)
    status, result = ms.Exit_Debug_Mode()
    print("Exit debug mode:", status, result)
    time.sleep(2)
