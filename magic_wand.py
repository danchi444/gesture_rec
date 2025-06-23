import asyncio
import tkinter as tk
from tkinter import messagebox
from bleak import BleakClient, BleakScanner
import pyautogui
import threading
import queue
from PIL import Image, ImageTk
import os
import sys

GESTURE_CHAR_UUID = "24b077e2-a798-49ea-821f-824bd3998bde"
GESTURE_NAMES = ['double', 'flick', 'infinity', 'junk', 'kiss']

class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Magic Wand")
        self.root.geometry("540x540")
        self.root.configure(bg="#E0FFFF")

        self.hotkey_entries = {}
        self.status_label = tk.Label(root, text="Ready to connect", bd=1, relief="sunken", anchor="w", bg="#ADD8E6", fg="black")
        self.status_label.pack(side="bottom", fill="x", padx=5, pady=5)

        self.gesture_descriptions = {
            "kiss": "To perform this gesture, extend you hand as if offering it to someone.",
            "infinity": "To perform this gesture, move your hand in an infinity figure while keeping you wrist at a fixed point starting from the center of the symbol moving up and to the right.",
            "double": "To perform this gesture rotate you wrist 180 degrees to the right and back twice.",
            "flick": "To perform this gesture, keep your elbow fixed and throw your hand back while bending at the elbow, not the wrist."
        }
        self.tooltip_window = None

        self.fish_canvas = None
        self.fish_image = None
        self.fish_items = []
        self.fish_x_positions = []
        self.fish_speed = 1
        
        self.client = None
        self.ble_thread = None
        self.asyncio_loop = None
        self.task_queue = queue.Queue()
        self.is_listening = False
        self.ble_connection_task = None

        self.create_widgets()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.process_queue()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=15, pady=15, bg="#E0FFFF")
        main_frame.pack(expand=True, fill="both")

        instruction_text = (
            "Map hotkeys to gestures below. For key combinations (e.g., Ctrl+C), "
            "type them as 'ctrl+c'. Use the 'Capture' button for single key presses "
            "or combinations like 'ctrl+j'."
        )
        instruction_label = tk.Label(main_frame, text=instruction_text, wraplength=400, justify="left", bg="#E0FFFF", fg="#191970", font=("Arial", 9), anchor="w")
        instruction_label.pack(pady=(0, 15), fill="x")

        for i, gesture in enumerate(GESTURE_NAMES): 
            if gesture != 'junk':
                row_frame = tk.Frame(main_frame, bg="#E0FFFF")
                row_frame.pack(pady=5, fill="x")

                info_button = tk.Label(row_frame, text="i", width=2, height=1, relief="solid", bd=1,
                                    font=("Arial", 8, "bold"), bg="#00BFFF", fg="white", cursor="hand2",
                                    highlightbackground="#00BFFF", highlightthickness=1, borderwidth=0)
                info_button.pack(side="left", padx=(0, 5))
                info_button.bind("<Enter>", lambda e, g=gesture, b=info_button: self.show_tooltip(e, g, b))
                info_button.bind("<Leave>", self.hide_tooltip)

                label = tk.Label(row_frame, text=f"{gesture.capitalize()}:", width=10, anchor="w", bg="#E0FFFF", fg="#483D8B", font=("Arial", 10, "bold"))
                label.pack(side="left", padx=(0, 10))

                entry = tk.Entry(row_frame, width=30, bg="white", fg="black", insertbackground="purple", bd=2, relief="groove")
                entry.pack(side="left", expand=True, fill="x")
                self.hotkey_entries[gesture] = entry

                if gesture == 'infinity':
                    entry.insert(0, '0')
                elif gesture == 'double':
                    entry.insert(0, 'l')
                elif gesture == 'flick':
                    entry.insert(0, 'k')
                elif gesture == 'kiss':
                    entry.insert(0, '120+enter')
                
                capture_button = tk.Button(row_frame, text="Capture", command=lambda g=gesture: self.start_hotkey_capture(g),
                                        bg="#FF69B4", fg="white", activebackground="#FFC0CB", relief="raised", bd=3, font=("Arial", 9, "bold"))
                capture_button.pack(side="right", padx=(5,0))

        self.start_button = tk.Button(main_frame, text="Start listening for gestures", command=self.start_listening,
                                       bg="#8A2BE2", fg="white", activebackground="#DDA0DD", relief="ridge", bd=4, font=("Arial", 11, "bold"))
        self.start_button.pack(pady=25)

        self.fish_canvas = tk.Canvas(main_frame, height=50, bg="#E0FFFF", highlightthickness=0)
        self.fish_canvas.pack(side="bottom", fill="x", padx=10, pady=5)
        
        self.root.after(100, self.load_fish_image)

    def load_fish_image(self):
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(__file__)
            
            image_path = os.path.join(base_path, "media", "ribice.webp")
            
            original_image = Image.open(image_path)
            
            self.root.update_idletasks()
            app_width = self.root.winfo_width()
            
            if app_width == 1:
                self.root.after(100, self.load_fish_image)
                return

            target_width = app_width // 3
            
            width_percent = (target_width / float(original_image.size[0]))
            target_height = int((float(original_image.size[1]) * float(width_percent)))

            self.fish_image = ImageTk.PhotoImage(original_image.resize((target_width, target_height), Image.Resampling.LANCZOS))
            
            self.fish_canvas.config(height=target_height + 10)

            self.root.update_idletasks()
            canvas_width = self.fish_canvas.winfo_width()
            
            for item in self.fish_items:
                self.fish_canvas.delete(item)
            self.fish_items = []
            self.fish_x_positions = []

            num_fishes_to_draw = (canvas_width // target_width) + 2

            for i in range(num_fishes_to_draw):
                x_pos = i * target_width
                self.fish_x_positions.append(x_pos)
                item = self.fish_canvas.create_image(x_pos, self.fish_canvas.winfo_height() // 2, image=self.fish_image, anchor="w")
                self.fish_items.append(item)
            
            self.animate_fish()

        except FileNotFoundError:
            print(f"Error: ribice.webp not found at expected path: {image_path}")
            self.fish_canvas.destroy()
            self.root.geometry("540x486")
            messagebox.showwarning("Image Not Found", f"Could not load ribice.webp. Please ensure it's in a 'media' subfolder next to the executable.")
        except Exception as e:
            print(f"An error occurred loading fish image: {e}")
            self.fish_canvas.destroy()
            self.root.geometry("540x486")
            messagebox.showerror("Image Error", f"An error occurred loading ribice.webp: {e}")

    def animate_fish(self):
        canvas_width = self.fish_canvas.winfo_width()
        
        if canvas_width == 0 or not self.fish_items or not self.fish_image:
            self.root.after(100, self.animate_fish)
            return

        fish_width = self.fish_image.width()
        
        max_current_x_overall = 0
        if self.fish_x_positions:
            max_current_x_overall = max(self.fish_x_positions)

        for i, item in enumerate(self.fish_items):
            self.fish_x_positions[i] += self.fish_speed
            
            if self.fish_x_positions[i] > canvas_width:
                self.fish_x_positions[i] = -fish_width

            self.fish_canvas.coords(item, self.fish_x_positions[i], self.fish_canvas.winfo_height() // 2)
        
        self.root.after(50, self.animate_fish)

    def show_tooltip(self, event, gesture_name, widget):
        x = widget.winfo_rootx() + widget.winfo_width() + 5
        y = widget.winfo_rooty() + widget.winfo_height() + 5

        if self.tooltip_window:
            self.hide_tooltip(None)

        self.tooltip_window = tk.Toplevel(self.root)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        description = self.gesture_descriptions.get(gesture_name, "No description available.")
        label = tk.Label(self.tooltip_window, text=description, background="#FFFFCC", relief="solid",
                         borderwidth=1, wraplength=200, justify="left", padx=5, pady=5, fg="#483D8B", font=("Arial", 9))
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    def start_hotkey_capture(self, gesture_name):
        entry_widget = self.hotkey_entries[gesture_name]
        entry_widget.delete(0, tk.END)
        self.update_status(f"Press hotkey for '{gesture_name}'. Click away or press Enter when done.", "orange")

        self.root.bind("<KeyPress>", lambda event, g=gesture_name: self.on_key_press_capture(event, g))
        self.root.bind("<Return>", lambda event: self.end_hotkey_capture())

        self.root.bind("<Button-1>", lambda event: self.end_hotkey_capture_on_click_outside(event)) 

    def on_key_press_capture(self, event, gesture_name):
        entry_widget = self.hotkey_entries[gesture_name]
        key = event.keysym.lower()

        modifiers = []
        if event.state & 0x0004: modifiers.append('ctrl')
        if event.state & 0x0008: modifiers.append('alt')
        if event.state & 0x0001: modifiers.append('shift')

        if key in ['control_l', 'control_r', 'alt_l', 'alt_r', 'shift_l', 'shift_r']:
            pass
        else:
            combined_key = '+'.join(modifiers + [key])
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, combined_key)

    def end_hotkey_capture(self):
        self.root.unbind("<KeyPress>")
        self.root.unbind("<Return>")
        self.root.unbind("<Button-1>")
        self.update_status("Hotkey captured. Ready to connect.")

    def end_hotkey_capture_on_click_outside(self, event):
        if not (event.widget in self.hotkey_entries.values() or
                event.widget == self.start_button or
                any(event.widget == btn for btn in self.root.winfo_children() if isinstance(btn, tk.Button) and btn.cget("text") == "Capture")):
            self.end_hotkey_capture()

    def start_listening(self):
        if not self.is_listening:
            self.is_listening = True
            self.start_button.config(state=tk.DISABLED, text="Connecting...")
            self.update_status("Scanning for magic_wand...", "blue")

            self.asyncio_loop = asyncio.new_event_loop() 
            self.ble_thread = threading.Thread(target=self.run_ble_loop)
            self.ble_thread.daemon = True
            self.ble_thread.start()
        else:
            self.is_listening = False
            self.start_button.config(state=tk.DISABLED, text="Disconnecting...")
            self.update_status("Disconnecting from magic_wand...", "orange")
            
            if self.asyncio_loop.is_running() and self.ble_connection_task and not self.ble_connection_task.done():
                self.asyncio_loop.call_soon_threadsafe(self.ble_connection_task.cancel)
            else:
                self.asyncio_loop.call_soon_threadsafe(self.asyncio_loop.create_task, self._perform_disconnection_and_cleanup())

    def run_ble_loop(self):
        asyncio.set_event_loop(self.asyncio_loop)
        try:
            self.ble_connection_task = self.asyncio_loop.create_task(self.connect_and_listen())
            self.asyncio_loop.run_until_complete(self.ble_connection_task)
        except asyncio.CancelledError:
            print("BLE loop task was cancelled. Initiating cleanup.")
            self.asyncio_loop.run_until_complete(self._perform_disconnection_and_cleanup())
        except Exception as e:
            print(f"Error running BLE loop: {e}")
            self.asyncio_loop.run_until_complete(self._perform_disconnection_and_cleanup(error=e))
        finally:
            if self.asyncio_loop.is_running():
                self.asyncio_loop.stop()
            if not self.asyncio_loop.is_running():
                self.asyncio_loop.close()
            print("BLE loop finished and closed.")

    async def connect_and_listen(self):
        try:
            device = await BleakScanner.find_device_by_name("magic_wand", timeout=60.0)
            if not device:
                self.queue_task(lambda: self.update_status("magic_wand not found within 1 minute.", "red"))
                self.queue_task(lambda: self.start_button.config(state=tk.NORMAL, text="Start listening for gestures"))
                self.is_listening = False
                return

            self.client = BleakClient(device)
            await self.client.connect()
            self.queue_task(lambda: self.update_status("Connected to magic_wand.", "green"))
            self.queue_task(lambda: self.start_button.config(state=tk.NORMAL, text="Stop listening for gestures")) 

            def handle_notification(_, data):
                gesture_idx = int.from_bytes(data, byteorder='little', signed=True)
                self.queue_task(lambda: self.process_gesture(gesture_idx))

            await self.client.start_notify(GESTURE_CHAR_UUID, handle_notification)
            self.queue_task(lambda: self.update_status("Listening for gestures...", "blue"))

            while self.client.is_connected and self.is_listening: 
                await asyncio.sleep(1)
            
            if self.client and self.client.is_connected and not self.is_listening:
                 print("Connect loop ended naturally (self.is_listening became False). Initiating cleanup.")
                 self.asyncio_loop.create_task(self._perform_disconnection_and_cleanup())

        except asyncio.CancelledError:
            print("Connect and listen coroutine explicitly cancelled. Cleanup will be handled by run_ble_loop.")
            raise
        except Exception as e:
            print(f"Connect and listen unexpected error: {e}")
            self.queue_task(lambda: self.update_status(f"Connection error: {e}", "red"))
            self.queue_task(lambda: self.start_button.config(state=tk.NORMAL, text="Start listening for gestures"))
            self.is_listening = False
            raise

    async def _perform_disconnection_and_cleanup(self, error=None):
        """
        Handles the actual BLE disconnection and subsequent GUI cleanup.
        Runs within the BLE thread's asyncio loop.
        `error`: An exception if cleanup is triggered due to an error.
        """
        print("Attempting _perform_disconnection_and_cleanup...")
        try:
            if self.client and self.client.is_connected:
                await self.client.stop_notify(GESTURE_CHAR_UUID)
                await self.client.disconnect()
                print("Bluetooth client disconnected gracefully during cleanup.")
            elif self.client and not self.client.is_connected:
                print("Bluetooth client already disconnected when _perform_disconnection_and_cleanup called.")
            else:
                print("Bluetooth client object not initialized when _perform_disconnection_and_cleanup called.")

            self.queue_task(lambda: self.start_button.config(state=tk.NORMAL, text="Start listening for gestures"))
            if error:
                self.queue_task(lambda: self.update_status(f"Disconnected (Error: {error}).", "red"))
            else:
                self.queue_task(lambda: self.update_status("Disconnected.", "black"))
            self.queue_task(lambda: setattr(self, 'is_listening', False))

        except Exception as e:
            print(f"Critical error during _perform_disconnection_and_cleanup: {e}")
            self.queue_task(lambda: self.update_status(f"Critical Disconnect Error: {e}", "red"))
        finally:
            self.client = None


    def process_gesture(self, gesture_idx):
        if 0 <= gesture_idx < len(GESTURE_NAMES): 
            gesture_name = GESTURE_NAMES[gesture_idx]
            hotkey_str = self.hotkey_entries[gesture_name].get().strip()

            self.update_status(f"Gesture received: {gesture_name} (index {gesture_idx})", "purple")
            print(f"Gesture received: {gesture_name} (index {gesture_idx}), hotkey: '{hotkey_str}'")

            if hotkey_str: 
                if '+' in hotkey_str:
                    keys = hotkey_str.split('+')
                    pyautogui.hotkey(*keys)
                elif len(hotkey_str) > 1 and not hotkey_str.isdigit() and not hotkey_str.isalpha():
                    pyautogui.typewrite(hotkey_str)
                else:
                    pyautogui.press(hotkey_str)
                self.update_status(f"Performed action for: {gesture_name} - '{hotkey_str}'", "green")
            else:
                self.update_status(f"No hotkey mapped for gesture: {gesture_name}", "orange")
                print(f"No hotkey mapped for gesture: {gesture_name}")
        else:
            self.update_status(f"Received invalid gesture index: {gesture_idx}", "red")
            print(f"Received invalid gesture index: {gesture_idx}")

    def update_status(self, message, color="black"):
        self.status_label.config(text=message, fg=color)

    def queue_task(self, task):
        self.task_queue.put(task)

    def process_queue(self):
        try:
            while True:
                task = self.task_queue.get_nowait()
                task()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    async def disconnect_client(self):
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(GESTURE_CHAR_UUID)
                await self.client.disconnect()
                print("Bluetooth client disconnected gracefully via disconnect_client.")
            except Exception as e:
                print(f"Error during Bluetooth disconnection in disconnect_client: {e}")
        else:
            print("Client not connected when disconnect_client was called.")

    def on_closing(self):
        if self.is_listening:
            self.is_listening = False 
        
        if self.asyncio_loop.is_running():
            if self.ble_connection_task and not self.ble_connection_task.done():
                self.asyncio_loop.call_soon_threadsafe(self.ble_connection_task.cancel)
            else:
                self.asyncio_loop.call_soon_threadsafe(self.asyncio_loop.create_task, self._perform_disconnection_and_cleanup())

        if self.ble_thread and self.ble_thread.is_alive():
            self.ble_thread.join(timeout=3) 

        self.root.destroy()
        print("Application closed.")

if __name__ == "__main__":
    pyautogui.FAILSAFE = False

    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()