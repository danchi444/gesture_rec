import asyncio
from bleak import BleakClient, BleakScanner
import pyautogui

GESTURE_CHAR_UUID = "24b077e2-a798-49ea-821f-824bd3998bde"

gestures = ['4', '8', 'alpha', 'double', 'flick']

gesture_to_action = {
    "8": lambda: pyautogui.press("0"),       # ispocetka
    "double": lambda: pyautogui.press("l"),  # 5 sec >>
    "flick": lambda: pyautogui.press("k"),   # pokreni/stani
    "4": lambda: pyautogui.typewrite("120") or pyautogui.press("enter"),  # spec. vrijeme
    "alpha": lambda: pyautogui.press("j"),   # 5 sec <<
}

async def run():
    print("Scanning for magic_wand...")
    device = await BleakScanner.find_device_by_name("magic_wand", timeout=10.0)
    if not device:
        print("magic_wand not found.")
        return

    async with BleakClient(device) as client:
        print("Connected to magic_wand.")

        def handle_notification(_, data):
            gesture_idx = int.from_bytes(data, byteorder='little', signed=True)
            if 0 <= gesture_idx < len(gestures):
                gesture = gestures[gesture_idx]
                print(f"Gesture received: {gesture} (index {gesture_idx})")
                action = gesture_to_action.get(gesture)
                if action:
                    action()
                else:
                    print(f"No action mapped for gesture: {gesture}")
            else:
                print(f"Received invalid gesture index: {gesture_idx}")

        await client.start_notify(GESTURE_CHAR_UUID, handle_notification)
        print("Listening for gestures. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(run())