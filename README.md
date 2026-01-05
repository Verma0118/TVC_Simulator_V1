# 🚀 Thrust Vector Control (TVC) Simulator

An interactive 6-DOF rocket flight simulator with real-time 3D visualization and gimbal control, designed to teach thrust vector control concepts to beginners.

## Features

- **6-DOF Physics Engine**: Full rigid body dynamics with quaternion attitude representation
- **Real-time 3D Visualization**: PyQt5-based GUI with embedded Matplotlib 3D canvas
- **Gimbal Control**: ±60° gimbal range with predictive thrust vector visualization
- **Flight Statistics**: Track max altitude, max velocity, flight time, and total distance
- **Quick Presets**: 4 preset scenarios (Vertical, Turn Left/Right, Hover) for rapid exploration
- **Keyboard Shortcuts**: Space/S/R/G/E/H for Start, Step, Reset, Stage, Export, Help
- **Flight Data Export**: Save flight logs as CSV for analysis
- **Playback**: Replay recorded flights frame-by-frame
- **Ground Collision Detection**: Automatic pause with feedback on hard landings
- **Dark/Light Theme**: Toggle between dark and light color schemes
- **Beginner-Friendly Help System**: Comprehensive guide with tips and explanations

## Installation

1. **Create Python virtual environment:**
	```bash
	cd /Users/aarav/VS_Code-Python\ Projects
	python3 -m venv .venv
	source .venv/bin/activate
	```

2. **Install dependencies:**
	```bash
	pip install PyQt5 matplotlib numpy
	```

## Usage

1. **Run the simulator:**
	```bash
	python tvc3d_gui_v2.py
	```

2. **Basic Controls:**
	- Adjust **Gimbal X/Y** sliders to change thrust angle (watch the pink arrow preview)
	- Set **Throttle** slider for engine power (0-100%)
	- Press **Start** (or Space) to begin simulation
	- Press **Step** (or S) for frame-by-frame control
	- Press **Reset** (or R) to clear and restart

3. **Camera Control:**
	- Mouse scroll wheel to zoom in/out
	- Double-click to reset zoom
	- Toggle "Camera Track" to follow vehicle
	- Toggle "Top-down" for orthographic view

4. **Quick Presets:**
	- **Vertical**: Straight up flight (0° gimbal, full throttle)
	- **Turn Right**: 40° right gimbal, full throttle
	- **Turn Left**: 40° left gimbal, full throttle
	- **Hover**: Reduced throttle to maintain altitude

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Start/Pause |
| **S** | Single step |
| **R** | Reset simulation |
| **G** | Stage separation (drop 10kg) |
| **E** | Export flight data (CSV) |
| **H** | Show help dialog |

## Understanding the Display

### HUD (Heads-Up Display)
- **Velocity**: Current speed (m/s)
- **Altitude**: Height above ground (m)
- **Throttle**: Engine power percentage
- **Gimbal**: Current thrust deflection angles
- **Max Alt/Vel**: Peak values during flight
- **Time/Distance**: Flight duration and total path length

### 3D Scene
- **Orange sphere**: Vehicle (rocket)
- **Blue line**: Flight trajectory
- **Green arrow**: Velocity vector
- **RGB axes**: Vehicle body frame (Red=X, Green=Y, Blue=Z)
- **Orange rings**: Scale markers on ground plane
- **Magenta arrow & arc**: Predicted thrust direction (when paused)
- **Gray plane**: Ground at z=0

## Physics Model

The simulator models a 6-DOF rigid body with:
- **Thrust control**: Gimbal deflection (±60°) alters thrust direction
- **Gravity**: 9.81 m/s² downward
- **Minimum throttle**: 20% to maintain gimbal authority
- **Mass**: 100 kg initial, reduces with fuel consumption
- **Dynamics**: 4th-order Runge-Kutta integration at 0.01s timestep

## Flight Tips

1. **Small angles (5-10°)** produce gentle, smooth maneuvers
2. **Large angles (20-40°)** create dramatic turns and barrel rolls
3. **Lower throttle** (20-40%) can help with precision hovering
4. **Full throttle** (100%) gives maximum acceleration and control authority
5. **Camera tracking** automatically follows the vehicle for better viewing
6. **Trail length** slider controls how much history is displayed (10-5000 points)

## Data Export

Save flight data as CSV for external analysis:
1. After completing a flight, press **Export CSV** (or E)
2. Choose save location
3. Data includes: time, position, velocity, orientation (quaternion), mass

Columns: `t, x, y, z, vx, vy, vz, qx, qy, qz, qw, mass`

## Troubleshooting

**GUI doesn't open?**
- Ensure `tvc3d.py` is in the same directory
- Check that all dependencies are installed: `pip install PyQt5 matplotlib numpy`

**Simulation runs very slowly?**
- Reduce trail length to improve rendering performance
- Close other applications to free up system resources

**Vehicle doesn't respond to gimbal changes?**
- Ensure you press **Start** to begin the simulation
- Gimbal changes are locked once simulation starts; adjust before starting
- Check that throttle is above 20% minimum

## Educational Value

This simulator helps learners understand:
- How thrust vectoring provides attitude control without aerodynamic surfaces
- The relationship between gimbal angle and trajectory
- 6-DOF rigid body dynamics and quaternion representations
- Basic rocket flight mechanics and fuel depletion
- The importance of gimbal authority and minimum throttle

## Files

- `tvc3d_gui_v2.py` - Main GUI application (PyQt5, Matplotlib)
- `tvc3d.py` - Physics engine (6-DOF dynamics, RK4 integration)
- `test.py` - Basic testing utilities
- `README.md` - This file

## Performance Notes

- Tested and optimized for smooth real-time performance
- Typical frame rate: 60+ FPS on modern hardware
- Physics integration timestep: 0.01 seconds
- Trail rendering scales with history length for stability

## License & Attribution

Educational simulator created for learning thrust vector control concepts. Feel free to modify, extend, and distribute as needed.

---

**Ready to launch?** Press Space to start your first flight! 🚀
