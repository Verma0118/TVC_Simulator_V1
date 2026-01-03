#!/usr/bin/env python3
"""
PyQt5-based TVC GUI v2 — embeds a Matplotlib 3D canvas and a right-side control panel.

Run: python3 tvc3d_gui_v2.py
"""
import sys
import math
import csv
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tvc3d import TVC3DSim, quat_to_euler, quat_rotate, torque_to_gimbal, attitude_controller_pid


def _hex_to_rgba(h, alpha=1.0):
    h = h.lstrip('#')
    lv = len(h)
    if lv == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    elif lv == 3:
        r, g, b = int(h[0]*2, 16), int(h[1]*2, 16), int(h[2]*2, 16)
    else:
        return (0, 0, 0, alpha)
    return (r/255.0, g/255.0, b/255.0, alpha)


class AnimatedButton(QtWidgets.QPushButton):
    """QPushButton with a subtle drop-shadow that animates on hover."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create a drop shadow effect but start with zero blur
        self._shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        # subtler orange glow for hover
        self._shadow.setColor(QtGui.QColor(255, 140, 60, 140))
        self._shadow.setBlurRadius(0)
        self._shadow.setOffset(0, 1)
        self.setGraphicsEffect(self._shadow)
        self._blur_anim = QtCore.QPropertyAnimation(self._shadow, b'blurRadius', self)
        self._blur_anim.setDuration(200)
        self._blur_anim.setEasingCurve(QtCore.QEasingCurve.OutQuad)

    def enterEvent(self, ev):
        try:
            self._blur_anim.stop()
            self._blur_anim.setStartValue(self._shadow.blurRadius())
            self._blur_anim.setEndValue(10)
            self._blur_anim.start()
        except Exception:
            pass
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        try:
            self._blur_anim.stop()
            self._blur_anim.setStartValue(self._shadow.blurRadius())
            self._blur_anim.setEndValue(0)
            self._blur_anim.start()
        except Exception:
            pass
        super().leaveEvent(ev)


class Mpl3DCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111, projection='3d')
        # make canvas background compatible with dark theme by default
        try:
            fig.patch.set_facecolor('#121418')
            self.ax.set_facecolor('#121418')
        except Exception:
            pass
        super().__init__(fig)
        self.setParent(parent)


class TVCMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('🚀 Thrust Vector Control (TVC) Simulator')
        self.resize(1300, 750)

        # central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # left: matplotlib 3D canvas
        self.canvas = Mpl3DCanvas(self, width=7, height=6)
        # set a sensible default 3D view
        try:
            self.canvas.ax.view_init(elev=25, azim=-60)
        except Exception:
            pass
        layout.addWidget(self.canvas, stretch=2)

        # right: control panel inside a scroll area so all controls are reachable
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        ctrl = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl)
        ctrl_layout.setSpacing(8)
        ctrl_layout.setContentsMargins(8,8,8,8)
        scroll.setWidget(ctrl)
        layout.addWidget(scroll, stretch=1)

        # === HUD at top for immediate visibility ===
        hud_box = QtWidgets.QGroupBox('HUD')
        hud_box.setStyleSheet("QGroupBox { font-weight: 700; margin-top: 4px; margin-bottom: 2px; padding-top: 12px; }")
        hud_layout = QtWidgets.QVBoxLayout()
        hud_layout.setSpacing(3)
        hud_layout.setContentsMargins(4, 4, 4, 4)
        self.hud_vel = QtWidgets.QLabel('Velocity: 0.0 m/s')
        self.hud_alt = QtWidgets.QLabel('Altitude: 0.0 m')
        self.hud_thr = QtWidgets.QLabel('Throttle: 100%')
        self.hud_gimb = QtWidgets.QLabel('Gimbal X: 0.0°   Gimbal Y: 0.0°')
        self.hud_stats = QtWidgets.QLabel('Max Alt: 0.0m | Max Vel: 0.0m/s')
        self.hud_stats.setStyleSheet('QLabel { font-size: 9pt; color: #64b5f6; }')
        self.hud_flight = QtWidgets.QLabel('Time: 0.0s | Distance: 0.0m')
        self.hud_flight.setStyleSheet('QLabel { font-size: 9pt; color: #81c784; }')
        # small colored legend using HTML spans
        legend_html = (
            '<div><span style="display:inline-block;width:12px;height:8px;background:#1f77b4;margin-right:6px;"></span>Flight path</div>'
            '<div><span style="display:inline-block;width:12px;height:12px;background:#ff7f0e;border-radius:6px;margin-right:6px;"></span>Vehicle</div>'
            '<div><span style="display:inline-block;width:12px;height:2px;background:#2ca02c;margin-right:6px;display:inline-block;vertical-align:middle;"></span>Velocity</div>'
        )
        self.hud_legend = QtWidgets.QLabel(legend_html)
        self.hud_legend.setTextFormat(QtCore.Qt.RichText)
        hud_layout.addWidget(self.hud_vel)
        hud_layout.addWidget(self.hud_alt)
        hud_layout.addWidget(self.hud_thr)
        hud_layout.addWidget(self.hud_gimb)
        hud_layout.addWidget(self.hud_stats)
        hud_layout.addWidget(self.hud_flight)
        hud_layout.addWidget(self.hud_legend)
        hud_box.setLayout(hud_layout)
        ctrl_layout.addWidget(hud_box)
        ctrl_layout.addSpacing(2)

        # === Primary control buttons (Start/Stop/Step/Reset) ===
        self.start_btn = AnimatedButton('Start')
        self.start_btn.setMinimumHeight(38)
        self.start_btn.setToolTip('Start/Pause simulation (Space)')
        self.step_btn = AnimatedButton('Step')
        self.step_btn.setMinimumHeight(32)
        self.step_btn.setToolTip('Advance one timestep (S)')
        self.reset_btn = AnimatedButton('Reset')
        self.reset_btn.setMinimumHeight(32)
        self.reset_btn.setToolTip('Reset simulation (R)')
        self.stage_btn = AnimatedButton('Stage Sep')
        self.stage_btn.setMinimumHeight(32)
        self.stage_btn.setToolTip('Trigger stage separation (G)')
        
        # arrange in 2x2 grid for compact appearance
        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setSpacing(6)
        btn_grid.addWidget(self.start_btn, 0, 0, 1, 2)
        btn_grid.addWidget(self.step_btn, 1, 0)
        btn_grid.addWidget(self.reset_btn, 1, 1)
        btn_grid.addWidget(self.stage_btn, 2, 0, 1, 2)
        ctrl_layout.addLayout(btn_grid)
        ctrl_layout.addSpacing(8)
        
        # === Help & Quick Start ===
        self.help_btn = AnimatedButton('? Quick Help')
        self.help_btn.setMinimumHeight(28)
        self.help_btn.setToolTip('Show beginner guide and controls (H)')
        ctrl_layout.addWidget(self.help_btn)
        ctrl_layout.addSpacing(4)
        
        # === Preset Configurations ===
        preset_label = QtWidgets.QLabel('Quick Presets:')
        preset_label.setStyleSheet('QLabel { font-weight: bold; }')
        ctrl_layout.addWidget(preset_label)
        preset_grid = QtWidgets.QGridLayout()
        preset_grid.setSpacing(4)
        self.preset_hover = AnimatedButton('Hover')
        self.preset_hover.setToolTip('Set gimbal for stationary hover')
        self.preset_right = AnimatedButton('Turn Right')
        self.preset_right.setToolTip('Set 40° aggressive right turn')
        self.preset_left = AnimatedButton('Turn Left')
        self.preset_left.setToolTip('Set 40° aggressive left turn')
        self.preset_ascent = AnimatedButton('Vertical')
        self.preset_ascent.setToolTip('Vertical ascent (0° gimbal)')
        preset_grid.addWidget(self.preset_hover, 0, 0)
        preset_grid.addWidget(self.preset_right, 0, 1)
        preset_grid.addWidget(self.preset_left, 1, 0)
        preset_grid.addWidget(self.preset_ascent, 1, 1)
        ctrl_layout.addLayout(preset_grid)
        ctrl_layout.addSpacing(8)

        # === Flight mode & gimbal controls ===
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(['auto', 'manual'])
        ctrl_layout.addWidget(QtWidgets.QLabel('Flight Mode'))
        ctrl_layout.addWidget(self.mode_combo)
        ctrl_layout.addSpacing(6)

        # gimbal sliders
        # Gimbal sliders: range represents tenths of degrees (±600 => ±60.0°)
        gx_label = QtWidgets.QLabel('Gimbal X (Roll/Pitch):')
        gx_label.setToolTip('Tilts thrust left/right. Positive = tilt right, Negative = tilt left')
        ctrl_layout.addWidget(gx_label)
        self.gx_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gx_slider.setRange(-600,600)
        self.gx_slider.setValue(0)
        self.gx_slider.setToolTip('Adjust thrust angle left/right')
        self.gy_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gy_slider.setRange(-600,600)
        self.gy_slider.setValue(0)
        self.gy_slider.setToolTip('Adjust thrust angle forward/back')
        h_gx = QtWidgets.QHBoxLayout()
        h_gx.setSpacing(6)
        self.gx_val = QtWidgets.QLabel('0.0°')
        self.gx_val.setFixedWidth(45)
        self.gx_val.setAlignment(QtCore.Qt.AlignRight)
        h_gx.addWidget(self.gx_slider)
        h_gx.addWidget(self.gx_val)
        ctrl_layout.addLayout(h_gx)
        
        ctrl_layout.addWidget(QtWidgets.QLabel('Gimbal Y (°)'))
        h_gy = QtWidgets.QHBoxLayout()
        h_gy.setSpacing(6)
        self.gy_val = QtWidgets.QLabel('0.0°')
        self.gy_val.setFixedWidth(45)
        self.gy_val.setAlignment(QtCore.Qt.AlignRight)
        h_gy.addWidget(self.gy_slider)
        h_gy.addWidget(self.gy_val)
        ctrl_layout.addLayout(h_gy)
        ctrl_layout.addSpacing(6)

        # === Throttle control ===
        ctrl_layout.addWidget(QtWidgets.QLabel('Throttle (%)'))
        h_thr = QtWidgets.QHBoxLayout()
        h_thr.setSpacing(6)
        self.throttle_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.throttle_slider.setRange(0,100)
        self.throttle_slider.setValue(100)
        self.thr_val = QtWidgets.QLabel('100%')
        self.thr_val.setFixedWidth(45)
        self.thr_val.setAlignment(QtCore.Qt.AlignRight)
        h_thr.addWidget(self.throttle_slider)
        h_thr.addWidget(self.thr_val)
        ctrl_layout.addLayout(h_thr)
        ctrl_layout.addSpacing(8)

        # === Trail & view options ===
        ctrl_layout.addWidget(QtWidgets.QLabel('Trail Length'))
        self.trail_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.trail_slider.setRange(10, 5000)
        self.trail_slider.setValue(1000)
        self.trail_label = QtWidgets.QLabel('1000')
        self.trail_label.setFixedWidth(45)
        self.trail_label.setAlignment(QtCore.Qt.AlignRight)
        h_trail = QtWidgets.QHBoxLayout()
        h_trail.setSpacing(6)
        h_trail.addWidget(self.trail_slider)
        h_trail.addWidget(self.trail_label)
        ctrl_layout.addLayout(h_trail)
        ctrl_layout.addSpacing(4)

        # camera / view options
        self.auto_center_chk = QtWidgets.QCheckBox('Auto-center camera')
        self.auto_center_chk.setChecked(True)
        ctrl_layout.addWidget(self.auto_center_chk)
        self.camera_track_chk = QtWidgets.QCheckBox('Camera tracking (smooth follow)')
        self.camera_track_chk.setChecked(False)
        ctrl_layout.addWidget(self.camera_track_chk)

        # scale label
        self.scale_label = QtWidgets.QLabel('Scale: 0.0 m')
        ctrl_layout.addWidget(self.scale_label)
        
        # top-down / orthographic toggle
        self.topdown_chk = QtWidgets.QCheckBox('Top-down (orthographic)')
        self.topdown_chk.setChecked(False)
        ctrl_layout.addWidget(self.topdown_chk)
        self.topdown_chk.toggled.connect(self._draw_scene)
        ctrl_layout.addSpacing(8)

        # === Theme & appearance ===
        h_theme = QtWidgets.QHBoxLayout()
        self.theme_label = QtWidgets.QLabel('Theme: Dark')
        self.theme_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.theme_slider.setRange(0, 1)
        self.theme_slider.setValue(1)
        self.theme_slider.setFixedWidth(80)
        self.theme_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.theme_slider.setTickInterval(1)
        h_theme.addWidget(self.theme_label)
        h_theme.addWidget(self.theme_slider)
        ctrl_layout.addLayout(h_theme)
        ctrl_layout.addSpacing(4)

        # === Data & export buttons ===
        self.export_btn = AnimatedButton('Export CSV')
        self.export_btn.setMinimumHeight(30)
        ctrl_layout.addWidget(self.export_btn)

        self.playback_btn = AnimatedButton('Play Log')
        self.playback_btn.setMinimumHeight(30)
        ctrl_layout.addWidget(self.playback_btn)

        self.view_btn = AnimatedButton('View Run Data')
        self.view_btn.setMinimumHeight(30)
        self.view_btn.setToolTip('Open a table view of the recorded run')
        self.view_btn.setEnabled(False)
        ctrl_layout.addWidget(self.view_btn)

        self.stop_btn = AnimatedButton('Stop & Finalize')
        self.stop_btn.setMinimumHeight(30)
        self.stop_btn.setToolTip('Stop the run (finalize log)')
        ctrl_layout.addWidget(self.stop_btn)
        ctrl_layout.addSpacing(8)

        # status area (detailed text)
        self.status_box = QtWidgets.QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(100)
        ctrl_layout.addWidget(self.status_box)

        # push to bottom
        ctrl_layout.addStretch()

        # simulator
        self.sim = TVC3DSim()
        self.dt = 0.01  # smaller timestep for slower, more accurate movement
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(self.dt*1000))
        self.timer.timeout.connect(self._on_step)
        self.running = False
        
        # captured gimbal/throttle settings (set when Start is pressed)
        self.run_gimbal_x = 0.0  # radians
        self.run_gimbal_y = 0.0  # radians
        self.run_throttle = 1.0  # 0-1 scale

        # zoom state for canvas (mouse wheel)
        self.user_zoom = 1.0
        self.zoom_min = 0.2
        self.zoom_max = 6.0
        
        # camera tracking state for smooth following
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_offset = np.array([0.0, 0.0, 0.0])

        # run logging / playback
        self.run_log_states = []
        self.run_log_times = []
        self.stage_events = []
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.setInterval(int(self.dt*1000))
        self.playback_timer.timeout.connect(self._playback_step)
        self.playback_index = 0
        self.playback_running = False

        # state (same format as tvc3d)
        # Initialize pointing upward (+z): quaternion from 5° pitch tilt for visualization
        angle = math.radians(5.0)
        q0 = np.array([math.cos(angle/2), math.sin(angle/2), 0.0, 0.0])  # rotation about X axis for pitch
        self.state = np.zeros(14)
        self.state[6:10] = q0 / np.linalg.norm(q0)  # normalize quaternion
        self.state[13] = self.sim.mass0
        # flight path history
        self.pos_hist = [self.state[0:3].copy()]

        # plotting handles
        self.traj = None
        self.rocket_art = None

        # connect signals
        self.start_btn.clicked.connect(self._on_start)
        self.step_btn.clicked.connect(self._on_step)
        self.reset_btn.clicked.connect(self._on_reset)
        self.stage_btn.clicked.connect(self._on_stage)
        self.export_btn.clicked.connect(self._on_export)
        self.playback_btn.clicked.connect(self._on_playback)
        self.stop_btn.clicked.connect(self._on_stop_button)
        self.view_btn.clicked.connect(self._show_run_data)
        self.help_btn.clicked.connect(self._show_help)
        self.preset_hover.clicked.connect(lambda: self._apply_preset(0, 0, 60))
        self.preset_right.clicked.connect(lambda: self._apply_preset(-400, 0, 100))
        self.preset_left.clicked.connect(lambda: self._apply_preset(400, 0, 100))
        self.preset_ascent.clicked.connect(lambda: self._apply_preset(0, 0, 100))
        self.gx_slider.valueChanged.connect(self._update_slider_labels)
        self.gy_slider.valueChanged.connect(self._update_slider_labels)
        self.throttle_slider.valueChanged.connect(self._update_slider_labels)
        self.trail_slider.valueChanged.connect(lambda: self.trail_label.setText(str(self.trail_slider.value())))
        
        # redraw scene when gimbal sliders change (for arrow preview and HUD update)
        self.gx_slider.valueChanged.connect(self._draw_scene)
        self.gy_slider.valueChanged.connect(self._draw_scene)

        # canvas interactions: mouse wheel zoom and double-click reset
        try:
            self.canvas.mpl_connect('scroll_event', self._on_scroll)
            self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        except Exception:
            pass

        # small visual polish: larger buttons, slider ticks and styles
        for s in (self.gx_slider, self.gy_slider, self.throttle_slider, self.trail_slider):
            s.setTickPosition(QtWidgets.QSlider.TicksBelow)
            s.setTickInterval(20)

        self._apply_styles()
        # connect theme slider
        self.theme_slider.valueChanged.connect(self._on_theme_slider)
        # apply initial theme (dark by default)
        self._apply_theme('dark' if self.theme_slider.value() == 1 else 'light')

        # initial draw
        self._draw_scene()

        # run finished flag
        self.run_finished = False
        
        # welcome message
        self.status_box.append('🚀 Welcome to TVC Simulator! Press H for help or try Quick Presets.')
        self.status_box.append('💡 Tip: Adjust gimbal sliders and watch the pink arrow preview!')
        
        # flight statistics
        self.max_altitude = 0.0
        self.max_velocity = 0.0
        self.flight_start_time = 0.0
        self.total_distance = 0.0
        self.prev_pos = np.array([0.0, 0.0, 0.0])
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        if key == QtCore.Qt.Key_Space:
            self._on_start()
        elif key == QtCore.Qt.Key_S:
            self._on_step()
        elif key == QtCore.Qt.Key_R:
            self._on_reset()
        elif key == QtCore.Qt.Key_G:
            self._on_stage()
        elif key == QtCore.Qt.Key_E:
            self._on_export()
        elif key == QtCore.Qt.Key_H:
            self._show_help()
        else:
            super().keyPressEvent(event)
    
    def _apply_preset(self, gx, gy, throttle):
        """Apply preset gimbal and throttle configuration"""
        self.gx_slider.setValue(gx)
        self.gy_slider.setValue(gy)
        self.throttle_slider.setValue(throttle)
        self.status_box.append(f'✓ Preset applied: Gimbal X={gx/10:.1f}° Y={gy/10:.1f}° Throttle={throttle}%')
    
    def _show_help(self):
        """Show beginner help dialog"""
        help_text = """<h2>TVC Simulator - Quick Start Guide</h2>
        
        <h3>What is Thrust Vector Control (TVC)?</h3>
        <p>TVC is a technology that allows rockets to steer by tilting the engine thrust direction. 
        By changing the angle of the rocket's exhaust, you can control which direction it moves!</p>
        
        <h3>Basic Controls:</h3>
        <ul>
        <li><b>Gimbal X/Y</b> - Tilt the thrust angle left/right and forward/back</li>
        <li><b>Throttle</b> - Control how much engine power (0-100%)</li>
        <li><b>Start/Pause</b> - Begin or pause the simulation</li>
        <li><b>Reset</b> - Start over from the beginning</li>
        </ul>
        
        <h3>Quick Presets:</h3>
        <ul>
        <li><b>Vertical</b> - Straight up flight (0° gimbal)</li>
        <li><b>Turn Right/Left</b> - Bank turns in each direction</li>
        <li><b>Hover</b> - Try to maintain position (reduced throttle)</li>
        </ul>
        
        <h3>How to Use:</h3>
        <ol>
        <li>Adjust gimbal sliders to set thrust angle (watch pink arrow)</li>
        <li>Set throttle for desired power</li>
        <li>Press <b>Start</b> (or Space) to run simulation</li>
        <li>Watch the vehicle respond to your settings!</li>
        <li>Press <b>Reset</b> to try again</li>
        </ol>
        
        <h3>Keyboard Shortcuts:</h3>
        <p><b>Space</b> - Start/Pause | <b>S</b> - Step | <b>R</b> - Reset | <b>G</b> - Stage | <b>E</b> - Export | <b>H</b> - Help</p>
        
        <h3>Tips:</h3>
        <ul>
        <li>Small gimbal angles (5-10°) provide gentle turns</li>
        <li>Larger angles (20-30°) make sharp maneuvers</li>
        <li>Camera tracking follows the vehicle automatically</li>
        <li>Export your flight data as CSV for analysis</li>
        </ul>
        """
        
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle('TVC Simulator - Help')
        msg.setTextFormat(QtCore.Qt.RichText)
        msg.setText(help_text)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()
        
        # flight statistics
        self.max_altitude = 0.0
        self.max_velocity = 0.0
        self.flight_start_time = 0.0

    def _on_start(self):
        self.running = not self.running
        if self.running:
            # capture gimbal and throttle settings from sliders when starting
            self.run_gimbal_x = math.radians(self.gx_slider.value() / 10.0)
            self.run_gimbal_y = math.radians(self.gy_slider.value() / 10.0)
            self.run_throttle = self.throttle_slider.value() / 100.0
            gx_deg = self.gx_slider.value() / 10.0
            gy_deg = self.gy_slider.value() / 10.0
            thr_pct = self.throttle_slider.value()
            self.status_box.append(f'▶ RUN: Gimbal X={gx_deg:.1f}° Y={gy_deg:.1f}° Throttle={thr_pct}%')
            
            # beginner tip on first run
            if len(self.run_log_times) == 0:
                self.status_box.append('💡 Tip: Watch the pink arrow - it shows thrust direction!')
            
            self.start_btn.setText('Pause')
            self.timer.start()
        else:
            self.start_btn.setText('Start')
            self.timer.stop()

    def _on_step(self):
        # perform one integration step and redraw
        self._step_sim()
        self._draw_scene()

    def _on_reset(self):
        angle = math.radians(5.0)
        q0 = np.array([math.cos(angle/2), 0.0, math.sin(angle/2), 0.0])
        self.state = np.zeros(14)
        self.state[6:10] = q0
        self.state[13] = self.sim.mass0
        self.pos_hist = [self.state[0:3].copy()]
        # reset all sliders to defaults
        self.gx_slider.setValue(0)
        self.gy_slider.setValue(0)
        self.throttle_slider.setValue(100)
        self.trail_slider.setValue(1000)
        # reset run log state and stage events BEFORE drawing
        self.run_log_states = []
        self.run_log_times = []
        self.stage_events = []
        # reset flight statistics
        self.max_altitude = 0.0
        self.max_velocity = 0.0
        self.total_distance = 0.0
        self.prev_pos = self.state[0:3].copy()
        self.view_btn.setEnabled(False)
        self.run_finished = False
        self._draw_scene()  # draw after clearing everything

    def _on_stage(self):
        # simple mass drop
        self.state[13] = max(0.0, self.state[13] - 10.0)
        self.sim.T = 0.0
        # record stage event (time, position)
        t = self.run_log_times[-1] if len(self.run_log_times) > 0 else 0.0
        self.stage_events.append((t, self.state[0:3].copy()))
        self.status_box.append(f'Stage separation at t={t:.2f}s pos={self.state[0:3]}')
        self._draw_scene()

    def _on_canvas_click(self, event):
        # double click resets zoom
        try:
            if event.dblclick:
                self.user_zoom = 1.0
                self.status_box.append('Zoom reset')
                self._draw_scene()
        except Exception:
            pass

    def _on_scroll(self, event):
        # zoom in/out with mouse wheel; event.step may vary, use event.button for modern mpl
        try:
            step = 1
            # older mpl uses event.step, newer uses event.button ('up'/'down')
            if hasattr(event, 'step'):
                step = event.step
            elif hasattr(event, 'button'):
                step = 1 if event.button == 'up' else -1
            # adjust zoom factor
            factor = 0.9 if step > 0 else 1.1
            self.user_zoom *= factor
            self.user_zoom = max(self.zoom_min, min(self.zoom_max, self.user_zoom))
            self.status_box.append(f'Zoom: {self.user_zoom:.2f}x')
            self._draw_scene()
        except Exception:
            pass

    def _on_export(self):
        # auto-generate filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f'tvc3d_run_{timestamp}.csv'
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save CSV', default_name, 'CSV files (*.csv)')
        if not fname:
            return
        # export full run log if available
        try:
            with open(fname, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['t','x','y','z','vx','vy','vz','qx','qy','qz','qw','mass'])
                if len(self.run_log_times) > 0 and len(self.run_log_states) == len(self.run_log_times):
                    for tt, st in zip(self.run_log_times, self.run_log_states):
                        row = [tt, *st[0:3].tolist(), *st[3:6].tolist(), *st[6:10].tolist(), st[13]]
                        w.writerow(row)
                else:
                    # fallback: write current state only
                    w.writerow([0.0, *self.state[0:3].tolist(), *self.state[3:6].tolist(), *self.state[6:10].tolist(), self.state[13]])
            self.status_box.append(f'✓ Exported {len(self.run_log_times)} datapoints to {fname}')
        except Exception as e:
            self.status_box.append(f'✗ Export failed: {e}')

    def _on_playback(self):
        # toggle playback of recorded run
        if len(self.run_log_states) == 0:
            self.status_box.append('No recorded run to play back')
            return
        if self.playback_running:
            self.playback_running = False
            self.playback_timer.stop()
            self.playback_btn.setText('Play Log')
            # restore main timer if it was running
            if getattr(self, 'was_running', False):
                self.timer.start()
                self.running = True
                self.start_btn.setText('Pause')
        else:
            # start playback from beginning
            self.playback_index = 0
            self.playback_running = True
            self.playback_btn.setText('Stop')
            # pause simulator timer if running
            self.was_running = self.timer.isActive()
            if self.timer.isActive():
                self.timer.stop()
                self.running = False
                self.start_btn.setText('Start')
            self.playback_timer.start()

    def _on_stop_button(self):
        # finalize run: stop timer and mark finished
        if self.timer.isActive():
            self.timer.stop()
        self.running = False
        self.start_btn.setText('Start')
        self.run_finished = True
        # enable view button if we have data
        if len(self.run_log_states) > 0:
            self.view_btn.setEnabled(True)
        self.status_box.append('Run stopped. You can view recorded data.')

    def _show_run_data(self):
        # show recorded run in a dialog with table and export button
        if len(self.run_log_states) == 0:
            self.status_box.append('No run data available')
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Run Data')
        dlg.resize(900, 500)
        v = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(dlg)
        n = len(self.run_log_states)
        cols = ['t','x','y','z','vx','vy','vz','qx','qy','qz','qw','mass']
        table.setColumnCount(len(cols))
        table.setRowCount(n)
        table.setHorizontalHeaderLabels(cols)
        for i, st in enumerate(self.run_log_states):
            t = self.run_log_times[i] if i < len(self.run_log_times) else 0.0
            vals = [t, *st[0:3].tolist(), *st[3:6].tolist(), *st[6:10].tolist(), st[13]]
            for j, val in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(f"{val:.6g}")
                table.setItem(i, j, it)
        table.resizeColumnsToContents()
        v.addWidget(table)
        h = QtWidgets.QHBoxLayout()
        exp = QtWidgets.QPushButton('Export CSV')
        exp.clicked.connect(lambda: self._export_table_csv(dlg, cols))
        h.addStretch()
        h.addWidget(exp)
        v.addLayout(h)
        dlg.exec_()

    def _export_table_csv(self, parent, cols):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Run CSV', 'tvc3d_run_log.csv', 'CSV files (*.csv)')
        if not fname:
            return
        try:
            with open(fname, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(cols)
                for i, st in enumerate(self.run_log_states):
                    t = self.run_log_times[i] if i < len(self.run_log_times) else 0.0
                    row = [t, *st[0:3].tolist(), *st[3:6].tolist(), *st[6:10].tolist(), st[13]]
                    w.writerow(row)
            self.status_box.append(f'Exported run CSV to {fname}')
        except Exception as e:
            self.status_box.append(f'Export failed: {e}')

    def _on_theme_slider(self, val: int):
        mode = 'dark' if val == 1 else 'light'
        self.theme_label.setText(f'Theme: {"Dark" if val==1 else "Light"}')
        self._apply_theme(mode)

    def _playback_step(self):
        # advance one step in recorded log and redraw
        if self.playback_index >= len(self.run_log_states):
            self.playback_running = False
            self.playback_timer.stop()
            self.playback_btn.setText('Play Log')
            self.status_box.append('Playback finished')
            # restore main timer if it was running
            if getattr(self, 'was_running', False):
                self.timer.start()
                self.running = True
                self.start_btn.setText('Pause')
            return
        st = self.run_log_states[self.playback_index]
        self.state = st.copy()
        # build pos_hist for visualization up to current playback index
        start_idx = max(0, self.playback_index - (self.trail_slider.value() if hasattr(self, 'trail_slider') else 1000))
        self.pos_hist = [s[0:3].copy() for s in self.run_log_states[start_idx:self.playback_index+1]]
        self.playback_index += 1
        self._draw_scene()

    def _update_slider_labels(self):
        # update displayed numeric labels next to sliders
        self.gx_val.setText(f"{self.gx_slider.value()/10.0:.1f}°")
        self.gy_val.setText(f"{self.gy_slider.value()/10.0:.1f}°")
        self.thr_val.setText(f"{self.throttle_slider.value()}%")

    def _step_sim(self):
        # use captured gimbal and throttle settings (set when Start was pressed)
        gx = self.run_gimbal_x
        gy = self.run_gimbal_y
        throttle = self.run_throttle
        
        # apply throttle relative to base thrust; maintain minimum thrust (20%) for gimbal authority
        if self.sim.T != 0:
            # store base_T if missing
            if not hasattr(self, 'base_T'):
                self.base_T = self.sim.T
            # clamp throttle to minimum 20% to maintain gimbal control
            throttle_clamped = max(0.2, throttle)
            self.sim.T = self.base_T * throttle_clamped
        
        # integrate
        self.state = self.sim.rk4_step(self.state, np.array([gx, gy]), self.dt)
        
        # update flight statistics
        current_alt = self.state[2]
        current_vel = np.linalg.norm(self.state[3:6])
        self.max_altitude = max(self.max_altitude, current_alt)
        self.max_velocity = max(self.max_velocity, current_vel)
        
        # track total distance traveled
        current_pos = self.state[0:3]
        distance_step = np.linalg.norm(current_pos - self.prev_pos)
        self.total_distance += distance_step
        self.prev_pos = current_pos.copy()
        
        # ground collision detection
        if current_alt <= 0.1 and current_vel > 5.0:
            # hard impact
            self.running = False
            self.timer.stop()
            self.start_btn.setText('Start')
            self.status_box.append(f'💥 IMPACT! Altitude: {current_alt:.2f}m, Velocity: {current_vel:.2f}m/s')
            self.status_box.append('Tip: Try gentler gimbal angles for softer landings')
        elif current_alt <= 0.0:
            # soft landing or stopped
            self.state[2] = 0.0  # clamp to ground
            self.state[3:6] = np.zeros(3)  # stop velocity
        
        # record flight path and full state
        self.pos_hist.append(self.state[0:3].copy())
        self.run_log_states.append(self.state.copy())
        t = self.run_log_times[-1] + self.dt if len(self.run_log_times) > 0 else self.dt
        self.run_log_times.append(t)
        # trim history according to trail length
        maxp = self.trail_slider.value() if hasattr(self, 'trail_slider') else 1000
        if len(self.pos_hist) > maxp:
            self.pos_hist = self.pos_hist[-maxp:]
        if len(self.run_log_states) > maxp*4:
            self.run_log_states = self.run_log_states[-maxp*4:]

    def _draw_scene(self):
        ax = self.canvas.ax
        ax.cla()
        pos = self.state[0:3]
        # oriented body axes (precompute for vehicle glyph)
        body_x = quat_rotate(self.state[6:10], np.array([1.0,0.0,0.0]))
        body_y = quat_rotate(self.state[6:10], np.array([0.0,1.0,0.0]))
        body_z = quat_rotate(self.state[6:10], np.array([0.0,0.0,1.0]))
        scale = 1.0
        # choose text color consistent with theme
        is_dark = True if getattr(self, 'theme_slider', None) and self.theme_slider.value() == 1 else False
        text_color = '#e6eef8' if is_dark else '#5a3a1a'
        ax.set_xlabel('x (m)', color=text_color, fontsize=10); ax.set_ylabel('y (m)', color=text_color, fontsize=10); ax.set_zlabel('z (m)', color=text_color, fontsize=10)
        grid_color = '#333' if is_dark else '#e6d9cc'
        ax.grid(True, color=grid_color, linewidth=0.3, alpha=0.5)
        # tick label colors
        try:
            ax.xaxis.set_tick_params(colors=text_color, labelcolor=text_color)
            ax.yaxis.set_tick_params(colors=text_color, labelcolor=text_color)
            ax.zaxis.set_tick_params(colors=text_color, labelcolor=text_color)
        except Exception:
            pass
        # set pane colors to match background tint (subtle)
        try:
            bg = '#121418' if is_dark else '#ffffff'
            pane_rgba = _hex_to_rgba(bg, 0.02 if is_dark else 1.0)
            ax.w_xaxis.set_pane_color(pane_rgba)
            ax.w_yaxis.set_pane_color(pane_rgba)
            ax.w_zaxis.set_pane_color(pane_rgba)
        except Exception:
            pass
        # draw ground plane centered under current view
        try:
            allpos = np.array(self.pos_hist)
            if allpos.size > 0:
                cx, cy, cz = pos
                # base window expands with the path extent
                win = max(6.0, np.max(np.abs(allpos - pos)) + 1.0)
                # apply user zoom (mouse wheel)
                if hasattr(self, 'user_zoom'):
                    win *= self.user_zoom
                # create a modest grid for ground plane at z=0
                g = np.linspace(cx - win, cx + win, 2)
                X, Y = np.meshgrid(g, g)
                Z = np.zeros_like(X)
                ax.plot_surface(X, Y, Z, color=(0.92,0.92,0.92), alpha=0.6, linewidth=0, shade=False)
                # concentric distance markers on ground plane to indicate scale
                thetas = np.linspace(0, 2*math.pi, 120)
                rings = [win*0.25, win*0.5, win]
                for idx, r in enumerate(rings):
                    xs = cx + r * np.cos(thetas)
                    ys = cy + r * np.sin(thetas)
                    zs = np.zeros_like(xs)
                    # make the inner two rings more visible (primary launch radii)
                    if idx < 2:
                        ax.plot(xs, ys, zs, color='#ff8a00', lw=1.6, alpha=0.95)
                        try:
                            ax.text(cx + r, cy, 0.0, f"{r:.0f} m", color='#ffb86b', fontsize=9, horizontalalignment='left')
                        except Exception:
                            pass
                    else:
                        ax.plot(xs, ys, zs, color='0.6', lw=0.8, alpha=0.6)
                        try:
                            ax.text(cx + r, cy, 0.0, f"{r:.0f} m", color='0.45', fontsize=8, horizontalalignment='left')
                        except Exception:
                            pass
                # lightweight grid lines along x/y for orientation
                grd = np.linspace(cx - win, cx + win, 9)
                for gx in grd:
                    ax.plot([gx, gx], [cy - win, cy + win], [0, 0], color='0.9', lw=0.4)
                for gy in grd:
                    ax.plot([cx - win, cx + win], [gy, gy], [0, 0], color='0.9', lw=0.4)
                # set scale indicator label
                if hasattr(self, 'scale_label'):
                    self.scale_label.setText(f"Scale: {win:.1f} m")
                # canvas title
                ax.set_title('Inertial frame (meters)', color=text_color)
                # compass indicator in axes fraction coords (fixed to corner)
                try:
                    ax.annotate('', xy=(0.95, 0.85), xytext=(0.95, 0.75), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', color='k'))
                    ax.text(0.95, 0.87, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=9)
                except Exception:
                    pass
        except Exception:
            pass
        # draw flight path
        allpos = np.array(self.pos_hist)
        if allpos.shape[0] > 0:
            n = allpos.shape[0]
            if n > 1:
                # main trajectory line
                ax.plot(allpos[:,0], allpos[:,1], allpos[:,2], color='C0', lw=1.5, alpha=0.9)
                # small faded markers along path to help visual following
                alphas = np.linspace(0.15, 0.9, n)
                step = max(1, n // 80)
                for i in range(0, n, step):
                    ax.scatter([allpos[i,0]], [allpos[i,1]], [allpos[i,2]], color='C0', alpha=alphas[i], s=10)
            # draw a clear circular vehicle marker (larger for visibility)
            try:
                # marker size scales inversely with zoom so it stays visible
                z = getattr(self, 'user_zoom', 1.0)
                msize = max(60, int(120 / max(0.2, z)))
                ax.scatter([pos[0]], [pos[1]], [pos[2]], color='#ff7f0e', edgecolors='#3a2a10', linewidths=0.8, s=msize, label='Vehicle')
            except Exception:
                ax.scatter([pos[0]], [pos[1]], [pos[2]], color='C1', s=80, label='Vehicle')
        # draw oriented body axes with tapered tips
        try:
            axes = [(body_x, (1.0,0.2,0.2,0.9)), (body_y, (0.2,0.9,0.2,0.9)), (body_z, (0.2,0.5,0.9,0.9))]
            for vec, col in axes:
                base = pos
                tip = pos + vec * (0.9 * scale)
                stem = [base.tolist(), (pos + 0.7*vec*scale).tolist()]
                ax.plot([p[0] for p in stem], [p[1] for p in stem], [p[2] for p in stem], color=col, lw=2.2)
                # small triangular cone tip
                tlen = 0.25 * scale
                # create two vectors perpendicular to vec for the cone base
                perp1 = np.cross(vec, np.array([0.0,0.0,1.0]))
                if np.linalg.norm(perp1) < 1e-6:
                    perp1 = np.cross(vec, np.array([0.0,1.0,0.0]))
                perp1 /= np.linalg.norm(perp1)
                perp2 = np.cross(vec, perp1)
                base_circle = [tip - vec * tlen + 0.08 * (math.cos(a)*perp1 + math.sin(a)*perp2) for a in (0, 2.09, 4.18)]
                face = [[tip.tolist(), base_circle[0].tolist(), base_circle[1].tolist()], [tip.tolist(), base_circle[1].tolist(), base_circle[2].tolist()], [tip.tolist(), base_circle[2].tolist(), base_circle[0].tolist()]]
                cone = Poly3DCollection(face, facecolors=col, linewidths=0.1)
                ax.add_collection3d(cone)
        except Exception:
            # fallback simple lines
            ax.plot([pos[0], pos[0]+scale*body_x[0]], [pos[1], pos[1]+scale*body_x[1]], [pos[2], pos[2]+scale*body_x[2]], color='r')
            ax.plot([pos[0], pos[0]+scale*body_y[0]], [pos[1], pos[1]+scale*body_y[1]], [pos[2], pos[2]+scale*body_y[2]], color='g')
            ax.plot([pos[0], pos[0]+scale*body_z[0]], [pos[1], pos[1]+scale*body_z[1]], [pos[2], pos[2]+scale*body_z[2]], color='b')
        
        # draw predictive thrust vector (based on current gimbal slider settings)
        if not self.running:  # only show prediction when not running
            try:
                # compute thrust direction using CURRENT gimbal slider values (not run_gimbal)
                gx_pred = math.radians(self.gx_slider.value() / 10.0)
                gy_pred = math.radians(self.gy_slider.value() / 10.0)
                # thrust in body frame: [sin(gx)*T, -sin(gy)*T, cos(gx)*cos(gy)*T]
                tb_pred = np.array([
                    -math.sin(gx_pred) * self.sim.T,
                    math.sin(gy_pred) * self.sim.T,
                    math.cos(gx_pred) * math.cos(gy_pred) * self.sim.T,
                ])
                # rotate to inertial frame
                thrust_pred_inertial = quat_rotate(self.state[6:10], tb_pred)
                # normalize and scale for visualization
                thrust_mag = np.linalg.norm(thrust_pred_inertial)
                if thrust_mag > 1e-6:
                    thrust_dir = thrust_pred_inertial / thrust_mag
                    arrow_len = 2.5 * scale
                    ax.quiver(pos[0], pos[1], pos[2], thrust_dir[0], thrust_dir[1], thrust_dir[2], 
                             length=arrow_len, color='#ff00ff', linewidth=2.5, alpha=0.7, label='Predicted thrust')
                    
                    # draw arc from body_z to thrust direction showing gimbal deflection
                    gimbal_angle_rad = math.sqrt(gx_pred**2 + gy_pred**2)
                    gimbal_angle_deg = math.degrees(gimbal_angle_rad)
                    
                    # create arc by interpolating from body_z to thrust_dir
                    num_arc_pts = 50  # more points for smoother curve
                    arc_radius = 2.0 * scale  # larger radius
                    arc_pts = []
                    for i in range(num_arc_pts + 1):
                        t = i / num_arc_pts
                        # slerp-like interpolation between body_z and thrust_dir
                        angle_t = gimbal_angle_rad * t
                        pt = pos + arc_radius * (math.cos(angle_t) * body_z + math.sin(angle_t) * (np.cross(body_z, thrust_dir) / (np.linalg.norm(np.cross(body_z, thrust_dir)) + 1e-6)))
                        arc_pts.append(pt)
                    arc_pts = np.array(arc_pts)
                    ax.plot(arc_pts[:, 0], arc_pts[:, 1], arc_pts[:, 2], color='#ff00ff', linewidth=4.5, alpha=0.95)  # thicker, more opaque
                    
                    # add angle text label at arc midpoint
                    mid_idx = len(arc_pts) // 2
                    label_pos = arc_pts[mid_idx]
                    ax.text(label_pos[0], label_pos[1], label_pos[2], f'{gimbal_angle_deg:.1f}°', 
                           color='#ff00ff', fontsize=14, weight='bold', ha='center')
            except Exception:
                pass
            except Exception:
                pass
        
        # velocity vector for current state
        vel = self.state[3:6]
        vnorm = np.linalg.norm(vel)
        if vnorm > 1e-6:
            arrow_len = max(0.5, min(3.0, vnorm * 0.12))
            ax.quiver(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], length=arrow_len, color='C2', linewidth=1.5)
        # legend proxies
        proxies = [Line2D([0],[0], color='C0', lw=1.5), Line2D([0],[0], marker='o', color='w', markerfacecolor='C1', markersize=8), Line2D([0],[0], color='C2', lw=2)]
        lg = ax.legend(proxies, ['Flight path', 'Vehicle', 'Velocity'], loc='upper left')
        try:
            for txt in lg.get_texts():
                txt.set_color(text_color)
            lg.get_frame().set_facecolor('#1b2330' if text_color.startswith('#e6') else '#fff')
        except Exception:
            pass

        # status text
        roll,pitch,yaw = quat_to_euler(self.state[6:10])
        t = self.run_log_times[-1] if len(self.run_log_times) > 0 else 0.0
        stat = f"t={t:.2f}s  pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})  mass={self.state[13]:.1f} kg\nroll={math.degrees(roll):.1f}° pitch={math.degrees(pitch):.1f}° yaw={math.degrees(yaw):.1f}°"
        self.status_box.setPlainText(stat)

        # update HUD labels (immediately reflect live state)
        try:
            speed = np.linalg.norm(self.state[3:6])
            alt = pos[2]
            self.hud_vel.setText(f"Velocity: {speed:.2f} m/s")
            self.hud_alt.setText(f"Altitude: {alt:.2f} m")
            self.hud_thr.setText(f"Throttle: {int(self.throttle_slider.value())}%")
            self.hud_gimb.setText(f"Gimbal X: {self.gx_slider.value()/10.0:.1f}°   Gimbal Y: {self.gy_slider.value()/10.0:.1f}°")
            self.hud_stats.setText(f"Max Alt: {self.max_altitude:.1f}m | Max Vel: {self.max_velocity:.1f}m/s")
            self.hud_flight.setText(f"Time: {t:.1f}s | Distance: {self.total_distance:.1f}m")
        except Exception:
            pass

        # autoscale camera around current position
        try:
            allpos = np.array(self.pos_hist)
            if allpos.size > 0:
                cx, cy, cz = pos
                win = max(6.0, np.max(np.abs(allpos - pos)) + 1.0)
                # apply top-down/orthographic if requested
                if getattr(self, 'topdown_chk', None) and self.topdown_chk.isChecked():
                    try:
                        # set orthographic projection if supported
                        if hasattr(ax, 'set_proj_type'):
                            ax.set_proj_type('ortho')
                    except Exception:
                        pass
                    # top-down view: look down from +Z
                    ax.view_init(elev=90, azim=-90)
                    ax.set_xlim(cx - win, cx + win)
                    ax.set_ylim(cy - win, cy + win)
                    ax.set_zlim(0, max(1.0, cz + win))
                else:
                    try:
                        if hasattr(ax, 'set_proj_type'):
                            ax.set_proj_type('persp')
                    except Exception:
                        pass
                    # apply camera tracking or auto-center
                    if getattr(self, 'camera_track_chk', None) and self.camera_track_chk.isChecked():
                        # camera always centered on vehicle
                        follow_dist = win * 1.2
                        ax.set_xlim(pos[0] - follow_dist, pos[0] + follow_dist)
                        ax.set_ylim(pos[1] - follow_dist, pos[1] + follow_dist)
                        ax.set_zlim(max(0.0, pos[2] - follow_dist*0.3), pos[2] + follow_dist)
                    elif getattr(self, 'auto_center_chk', None) and self.auto_center_chk.isChecked():
                        ax.set_xlim(cx - win, cx + win)
                        ax.set_ylim(cy - win, cy + win)
                        ax.set_zlim(max(0.0, cz - win*0.2), cz + win)
                # draw stage markers (vertical lines) at recorded stage events
                for ev in self.stage_events:
                    t_ev, p_ev = ev
                    ax.plot([p_ev[0], p_ev[0]], [p_ev[1], p_ev[1]], [0.0, max(1.0, cz + win)], color='k', ls='--', lw=1.0)
        except Exception:
            pass

        # optimized draw with anti-aliasing disabled for performance
        try:
            self.canvas.draw_idle()
        except Exception:
            self.canvas.draw()

    def _apply_styles(self):
        # Base style setup — theme-specific details are handled by _apply_theme
        # Set some default global properties
        self.setStyleSheet("""
            QScrollArea { border: none; }
            QHeaderView::section { padding: 4px; }
        """)

    def _on_toggle_theme(self, checked: bool):
        self._apply_theme('dark' if checked else 'light')

    def _apply_theme(self, theme: str = 'light'):
        # theme-aware style adjustments with smooth animation
        if theme == 'dark':
            # teal-accent dark theme — flat & sophisticated
            btn_css = """
            QPushButton {
                background: #1a3a52;
                color: #64b5f6;
                border: 1px solid rgba(66,165,245,0.3);
                border-radius: 6px;
                padding: 10px 14px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #1f4a66;
                border: 1px solid rgba(100,181,246,0.5);
                color: #90caf9;
            }
            QPushButton:pressed {
                background: #0d2435;
                border: 1px solid rgba(100,181,246,0.6);
            }
            QGroupBox { 
                color: #e6eef8; 
                font-weight: 700;
                border: 1px solid rgba(66,165,245,0.15);
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QLabel { color: #e6eef8; }
            QTextEdit { background: #0f1113; color: #e6eef8; border: 1px solid rgba(66,165,245,0.15); }
            QCheckBox { color: #e6eef8; }
            QComboBox { color: #e6eef8; background: #1a1a1e; border: 1px solid rgba(66,165,245,0.15); }
            """
            sld_css = """
            QSlider::groove:horizontal { height: 6px; background: #1a1a1e; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #42a5f5; border-radius: 3px; }
            QSlider::handle:horizontal { background: #64b5f6; width: 14px; margin: -4px 0; border-radius: 7px; border: 1px solid rgba(66,165,245,0.4); }
            QSlider::add-page:horizontal { background: #222; border-radius: 3px; }
            """
            base = "QWidget { background: #0f1113; color: #e6eef8; }"
        else:
            # light theme: teal accents, flat design
            btn_css = """
            QPushButton {
                background: #d4ecf7;
                color: #1a4d6d;
                border: 1px solid rgba(66,165,245,0.3);
                border-radius: 6px;
                padding: 10px 14px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #c5e9f5;
                border: 1px solid rgba(66,165,245,0.5);
                color: #0d3d5c;
            }
            QPushButton:pressed {
                background: #81c3f7;
                border: 1px solid rgba(66,165,245,0.6);
            }
            QGroupBox { 
                color: #1a4d6d; 
                font-weight: 700;
                border: 1px solid rgba(66,165,245,0.2);
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QLabel { color: #1a4d6d; }
            QTextEdit { background: #fafbfc; color: #1a4d6d; border: 1px solid rgba(66,165,245,0.2); }
            QCheckBox { color: #1a4d6d; }
            QComboBox { color: #1a4d6d; background: #eef7fb; border: 1px solid rgba(66,165,245,0.2); }
            """
            sld_css = """
            QSlider::groove:horizontal { height: 6px; background: #d4e8f3; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #2196f3; border-radius: 3px; }
            QSlider::handle:horizontal { background: #1976d2; width: 14px; margin: -4px 0; border-radius: 7px; border: 1px solid rgba(66,165,245,0.4); }
            QSlider::add-page:horizontal { background: #e3f2fd; border-radius: 3px; }
            """
            base = "QWidget { background: #f5f9fb; color: #1a4d6d; }"

        try:
            # apply with a smooth fade (use setStyleSheet which triggers Qt's internal repaint)
            self.setStyleSheet(base + btn_css + sld_css)
            # trigger a gentle repaint animation on child widgets
            for widget in self.findChildren(QtWidgets.QWidget):
                try:
                    widget.update()
                except Exception:
                    pass
        except Exception:
            pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = TVCMainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
