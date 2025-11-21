import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.integrate import odeint
# as of 21/11/2025 i am still learning phython syntax so took help from github copilots inline code suggestions
# but all the code logic and structure was developed by me
# Constants 
G = 6.67430e-11  # Gravitational constant

# 3-Body Problem Equations of Motion 
def n_body_equations(y, t, m):
    """
    Defines the differential equations for the N-body problem.
    """
    num_bodies = len(m)
    dim = 3

    positions = y[:num_bodies * dim].reshape(num_bodies, dim)
    velocities = y[num_bodies * dim:].reshape(num_bodies, dim)

    dydt = np.zeros_like(y)
    dposdt = velocities.flatten()
    dveldt = np.zeros_like(velocities)

    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec)

                # Softening parameter to prevent extremely large forces at very small distances
                # Using 1.0e8 meters (100,000 km) as a typical stellar softening length
                softening_length = 1.0e8
                r_mag_cubed = (r_mag**2 + softening_length**2)**(1.5)

                dveldt[i] += G * m[j] * r_vec / r_mag_cubed

    dydt[:num_bodies * dim] = dposdt
    dydt[num_bodies * dim:] = dveldt.flatten()
    return dydt

# --- Simulation Class ---
class ThreeBodySimulation:
    def __init__(self, master):
        self.master = master
        master.title("3-Body Problem Simulator")

        # --- Simulation Parameters ---
        # Default masses (kg) - Jupiter mass order
        self.masses = [1.898e27, 1.898e27, 1.898e27]
        # Default positions (meters) - Solar System scale order
        self.initial_positions = [
            np.array([-5.0e10, 0.0, 0.0]),
            np.array([5.0e10, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0])
        ]
        # Default velocities (m/s) - Orbital speed order
        self.initial_velocities = [
            np.array([0.0, 3.0e3, 0.0]),
            np.array([0.0, -3.0e3, 0.0]),
            np.array([-5.0e2, 0.0, 0.0])
        ]
        self.colors = ['red', 'green', 'blue']
        self.num_bodies = 3

        self.time_step = 1000 # Simulated seconds per integration step
        self.simulation_time_scale = 1.0 # Multiplier for visual speed (can be any float)
        self.total_simulated_time = 0.0

        self.max_history_points = 5000

        self.is_running = False
        self.trace_data = [[] for _ in range(self.num_bodies)]

        # --- Plotting/View State ---
        self.view_limits = None # Stores current x,y,z limits, None for auto-scale
        self.press_x_canvas = None # Mouse X position on canvas when button pressed
        self.press_y_canvas = None # Mouse Y position on canvas when button pressed
        self.button_press_event = None # Store the matplotlib event to know which button was pressed

        # Initial 3D view angles
        self.current_elev = 30 # Elevation
        self.current_azim = -60 # Azimuth

        # Initialize current state (will be properly set by reset_simulation)
        self.current_positions = [pos.copy() for pos in self.initial_positions]
        self.current_velocities = [vel.copy() for vel in self.initial_velocities]
        self.current_y = np.concatenate([np.array(self.current_positions).flatten(),
                                         np.array(self.current_velocities).flatten()])

        # --- GUI Setup ---
        self.create_widgets()
        self.initialize_plot()
        self.reset_simulation() # Ensure initial state is correctly plotted


    def create_widgets(self):
        # --- Input Frame ---
        input_frame = ttk.LabelFrame(self.master, text="Initial Conditions", padding="15")
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10, expand=False)
        input_frame.config(width=380)
        input_frame.pack_propagate(False)

        self.entries = []
        for i in range(self.num_bodies):
            body_frame = ttk.LabelFrame(input_frame, text=f"Body {i+1} ({self.colors[i].capitalize()})", padding="10")
            body_frame.pack(pady=7, fill=tk.X)

            ttk.Label(body_frame, text="Mass (kg):").grid(row=0, column=0, sticky=tk.W, pady=2)
            mass_entry = ttk.Entry(body_frame, width=28)
            mass_entry.insert(0, f"{self.masses[i]:.4e}")
            mass_entry.grid(row=0, column=1, sticky=tk.E, padx=5, pady=2)

            ttk.Label(body_frame, text="Pos (x,y,z m):").grid(row=1, column=0, sticky=tk.W, pady=2)
            pos_entry = ttk.Entry(body_frame, width=28)
            pos_entry.insert(0, ','.join(f"{p:.4e}" for p in self.initial_positions[i]))
            pos_entry.grid(row=1, column=1, sticky=tk.E, padx=5, pady=2)

            ttk.Label(body_frame, text="Vel (vx,vy,vz m/s):").grid(row=2, column=0, sticky=tk.W, pady=2)
            vel_entry = ttk.Entry(body_frame, width=28)
            vel_entry.insert(0, ','.join(f"{v:.4e}" for v in self.initial_velocities[i]))
            vel_entry.grid(row=2, column=1, sticky=tk.E, padx=5, pady=2)

            self.entries.append({'mass': mass_entry, 'pos': pos_entry, 'vel': vel_entry})

        ttk.Button(input_frame, text="Apply & Reset Simulation", command=self.apply_conditions).pack(pady=15)

        # --- Simulation Controls Frame ---
        control_frame = ttk.LabelFrame(self.master, text="Simulation Controls", padding="15")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.play_pause_button = ttk.Button(control_frame, text="Play", command=self.toggle_simulation)
        self.play_pause_button.pack(side=tk.LEFT, padx=7)

        ttk.Button(control_frame, text="Reset", command=self.reset_simulation).pack(side=tk.LEFT, padx=7)
        
        ttk.Button(control_frame, text="Auto-Scale View", command=self.auto_scale_view).pack(side=tk.LEFT, padx=7)

        # Time Scale Entry instead of Slider
        ttk.Label(control_frame, text="Time Scale (x):").pack(side=tk.LEFT, padx=(15, 0))
        self.time_scale_entry = ttk.Entry(control_frame, width=10)
        self.time_scale_entry.insert(0, str(self.simulation_time_scale))
        self.time_scale_entry.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(control_frame, text="Apply", command=self.update_time_scale_from_entry).pack(side=tk.LEFT, padx=(5, 7))

        self.time_label = ttk.Label(control_frame, text="Simulated Time: 0.0 s")
        self.time_label.pack(side=tk.RIGHT, padx=7)

    def apply_conditions(self):
        """Reads user input from entry fields and updates initial conditions."""
        try:
            new_masses = []
            new_positions = []
            new_velocities = []
            for i in range(self.num_bodies):
                mass_str = self.entries[i]['mass'].get()
                pos_str = self.entries[i]['pos'].get()
                vel_str = self.entries[i]['vel'].get()

                new_masses.append(float(mass_str))
                new_positions.append(np.array(list(map(float, pos_str.replace(' ', '').split(',')))))
                new_velocities.append(np.array(list(map(float, vel_str.replace(' ', '').split(',')))))

                # Update entry fields with formatted values for consistency
                self.entries[i]['mass'].delete(0, tk.END)
                self.entries[i]['mass'].insert(0, f"{new_masses[i]:.4e}")
                self.entries[i]['pos'].delete(0, tk.END)
                self.entries[i]['pos'].insert(0, ','.join(f"{p:.4e}" for p in new_positions[i]))
                self.entries[i]['vel'].delete(0, tk.END)
                self.entries[i]['vel'].insert(0, ','.join(f"{v:.4e}" for v in new_velocities[i]))

            self.masses = new_masses
            self.initial_positions = new_positions
            self.initial_velocities = new_velocities

            self.reset_simulation()
            self.auto_scale_view() # Auto-scale after applying new conditions
        except ValueError as ve:
            messagebox.showerror("Input Error", f"Please check your number format (e.g., 1.23e-4 or 123.45). For position/velocity, use comma-separated values like '1e10,0,0'. Error: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during input parsing: {e}")

    def update_time_scale_from_entry(self):
        """Updates the simulation speed from the entry widget."""
        try:
            new_scale = float(self.time_scale_entry.get())
            if new_scale <= 0:
                messagebox.showwarning("Input Warning", "Time scale must be a positive number. Using default 1.0.")
                self.simulation_time_scale = 1.0
                self.time_scale_entry.delete(0, tk.END)
                self.time_scale_entry.insert(0, "1.0")
            else:
                self.simulation_time_scale = new_scale
        except ValueError:
            messagebox.showerror("Input Error", "Invalid time scale. Please enter a numerical value (e.g., 1, 10, 0.5).")
            self.time_scale_entry.delete(0, tk.END)
            self.time_scale_entry.insert(0, str(self.simulation_time_scale)) # Revert to previous valid value

    def initialize_plot(self):
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # FORCEFULLY DISABLE DEFAULT MATPLOTLIB INTERACTION
        self.ax.mouse_init(rotate_btn=None, zoom_btn=None)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.ax.set_xlabel("X-axis (m)")
        self.ax.set_ylabel("Y-axis (m)")
        self.ax.set_zlabel("Z-axis (m)")
        self.ax.set_title(" ")
        self.ax.grid(True)

        self.body_spheres = [
            self.ax.scatter([], [], [], color=self.colors[i], s=150, alpha=0.8, edgecolors='w', linewidth=0.5, label=f'Body {i+1}')
            for i in range(self.num_bodies)
        ]
        self.body_paths = [
            self.ax.plot([], [], [], color=self.colors[i], linestyle='--', alpha=0.6, linewidth=1.5)[0]
            for i in range(self.num_bodies)
        ]
        self.ax.legend()

        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)

        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.canvas.mpl_connect('scroll_event', self.on_mouse_wheel)

    def on_button_press(self, event):
        if event.inaxes == self.ax:
            self.press_x_canvas = event.x
            self.press_y_canvas = event.y
            self.button_press_event = event

            # Store current limits in case of a pan/interaction
            self.initial_xlim = self.ax.get_xlim()
            self.initial_ylim = self.ax.get_ylim()
            self.initial_zlim = self.ax.get_zlim()

    def on_button_release(self, event):
        self.press_x_canvas = None
        self.press_y_canvas = None
        self.button_press_event = None

    def on_mouse_motion(self, event):
        if self.press_x_canvas is None or self.press_y_canvas is None:
            return
        if event.inaxes != self.ax:
            return

        dx_canvas = event.x - self.press_x_canvas
        dy_canvas = event.y - self.press_y_canvas

        if self.button_press_event.button == 1: # Left-click for CUSTOM 3D Rotation
            self.current_azim = (self.current_azim - dx_canvas * 0.2) % 360
            self.current_elev = (self.current_elev - dy_canvas * 0.2) 
            self.current_elev = np.clip(self.current_elev, -90, 90)

            self.ax.view_init(elev=self.current_elev, azim=self.current_azim)
            self.canvas.draw_idle()

        self.press_x_canvas = event.x
        self.press_y_canvas = event.y

    def on_mouse_wheel(self, event):
        if event.inaxes != self.ax:
            return
        
        # this is the same function used in autocad software for zooming and was directly adapted here 
        zoom_direction = 1.1 if event.button == 'up' else 1/1.1 # 'up' zooms in, 'down' zooms out

        # Calculate the center of the current view
        cx = (self.ax.get_xlim()[0] + self.ax.get_xlim()[1]) / 2
        cy = (self.ax.get_ylim()[0] + self.ax.get_ylim()[1]) / 2
        cz = (self.ax.get_zlim()[0] + self.ax.get_zlim()[1]) / 2

        # Apply zoom around the center
        new_xlim = [(self.ax.get_xlim()[0] - cx) * zoom_direction + cx,
                    (self.ax.get_xlim()[1] - cx) * zoom_direction + cx]
        new_ylim = [(self.ax.get_ylim()[0] - cy) * zoom_direction + cy,
                    (self.ax.get_ylim()[1] - cy) * zoom_direction + cy]
        new_zlim = [(self.ax.get_zlim()[0] - cz) * zoom_direction + cz,
                    (self.ax.get_zlim()[1] - cz) * zoom_direction + cz]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.ax.set_zlim(new_zlim)
        self.view_limits = (new_xlim, new_ylim, new_zlim) # Store manual limits
        self.canvas.draw_idle()

    def auto_scale_view(self):
        """Resets plot limits to auto-scale based on data."""
        self.view_limits = None # Clear any manually set limits
        self.set_plot_limits()  # Re-calculate limits based on all data
        # Reset view angles to default
        self.current_elev = 30
        self.current_azim = -60
        self.ax.view_init(elev=self.current_elev, azim=self.current_azim)
        self.canvas.draw_idle()

    def set_plot_limits(self):
        """Dynamically adjusts plot limits or applies stored manual limits."""
        # If view_limits are manually set (e.g., by mouse interaction), use them
        if self.view_limits is not None:
            self.ax.set_xlim(self.view_limits[0])
            self.ax.set_ylim(self.view_limits[1])
            self.ax.set_zlim(self.view_limits[2])
            return

        # Otherwise, calculate limits based on all data points
        all_x = []
        all_y = []
        all_z = []

        # Include all points from the trace data
        for i in range(self.num_bodies):
            if self.trace_data[i]:
                trace_array = np.array(self.trace_data[i])
                all_x.extend(trace_array[:, 0])
                all_y.extend(trace_array[:, 1])
                all_z.extend(trace_array[:, 2])

        # Include current positions
        current_pos_array = np.array(self.current_positions)
        all_x.extend(current_pos_array[:, 0])
        all_y.extend(current_pos_array[:, 1])
        all_z.extend(current_pos_array[:, 2])

        if not all_x: # Fallback if no data at all (e.g., just after initial setup with 0,0,0)
            initial_x = [pos[0] for pos in self.initial_positions]
            initial_y = [pos[1] for pos in self.initial_positions]
            initial_z = [pos[2] for pos in self.initial_positions]

            if initial_x: # If initial positions are not all 0,0,0
                min_x, max_x = min(initial_x), max(initial_x)
                min_y, max_y = min(initial_y), max(initial_y)
                min_z, max_z = min(initial_z), max(initial_z)
            else: # Absolutely no data, use a sensible default range
                min_x, max_x = -1e11, 1e11 # Larger default range to encompass potential data
                min_y, max_y = -1e11, 1e11
                min_z, max_z = -1e11, 1e11
        else:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            min_z, max_z = min(all_z), max(all_z)

        # Calculate range for all axes
        x_range = max_x - min_x
        y_range = max_y - min_y
        z_range = max_z - min_z

        # Use the largest range to set cubic limits for better 3D perception
        max_overall_range = max(x_range, y_range, z_range)
        if max_overall_range == 0: # If all points are the same, use a default range
            max_overall_range = 2e11 # Default range if bodies are at the same point

        # Add some padding to the limits
        padding = max_overall_range * 0.1
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        mid_z = (min_z + max_z) / 2

        self.ax.set_xlim(mid_x - max_overall_range / 2 - padding, mid_x + max_overall_range / 2 + padding)
        self.ax.set_ylim(mid_y - max_overall_range / 2 - padding, mid_y + max_overall_range / 2 + padding)
        self.ax.set_zlim(mid_z - max_overall_range / 2 - padding, mid_z + max_overall_range / 2 + padding)


    def reset_simulation(self):
        """Resets the simulation to initial conditions."""
        self.is_running = False
        self.play_pause_button.config(text="Play")
        self.total_simulated_time = 0.0

        self.current_positions = [pos.copy() for pos in self.initial_positions]
        self.current_velocities = [vel.copy() for vel in self.initial_velocities]

        self.current_y = np.concatenate([np.array(self.current_positions).flatten(),
                                         np.array(self.current_velocities).flatten()])

        self.trace_data = [[] for _ in range(self.num_bodies)]
        self.view_limits = None # Clear manual view limits on reset
        
        self.time_label.config(text="Simulated Time: 0.0 s")
        self.update_plot_elements() # Redraw plot to show reset state

    def toggle_simulation(self):
        """Toggles the simulation between playing and paused."""
        self.is_running = not self.is_running
        if self.is_running:
            self.play_pause_button.config(text="Pause")
            self.update_simulation()
        else:
            self.play_pause_button.config(text="Play")

    def update_plot_elements(self):
        """Updates the positions of spheres and paths on the plot."""
        for i in range(self.num_bodies):
            x, y, z = self.current_positions[i]
            self.body_spheres[i]._offsets3d = ([x], [y], [z])

            self.trace_data[i].append([x, y, z])
            if len(self.trace_data[i]) > self.max_history_points:
                self.trace_data[i].pop(0)

            if self.trace_data[i]:
                self.body_paths[i].set_data_3d(*zip(*self.trace_data[i]))
            else:
                self.body_paths[i].set_data_3d([], [], []) # Clear path if no data

        self.set_plot_limits() # Adjust plot limits (either auto or manual)
        self.canvas.draw_idle() # Redraw the canvas (optimized for animation)

    def update_simulation(self):
        """Performs one step of the simulation and updates the plot."""
        if self.is_running:
            sim_dt = self.time_step * self.simulation_time_scale
            times = np.array([0, sim_dt])

            sol = odeint(n_body_equations, self.current_y, times, args=(self.masses,))
            self.current_y = sol[-1]

            positions_flat = self.current_y[:self.num_bodies * 3]
            velocities_flat = self.current_y[self.num_bodies * 3:]

            self.current_positions = positions_flat.reshape(self.num_bodies, 3)
            self.current_velocities = velocities_flat.reshape(self.num_bodies, 3)

            self.total_simulated_time += sim_dt
            self.time_label.config(text=f"Simulated Time: {self.total_simulated_time:.1f} s")

            self.update_plot_elements()

            self.master.after(50, self.update_simulation) # Schedule the next update

def main():
    root = tk.Tk()
    app = ThreeBodySimulation(root)
    root.mainloop()

if __name__ == "__main__":
    main()
