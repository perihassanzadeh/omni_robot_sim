import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Robot Parameters and Setup ---
r = 0.019  # Wheel radius (m) - 38mm diameter = 19mm radius
L = 0.051  # Distance from robot center to wheel (m) - 2 inches = 0.051m

# INPUT PARAMETERS
ROBOT_ORIENT = 0.0  # Constant robot body orientation (degrees)
V_robot = 0.5       # Constant translational speed (m/s) - limited to 0.5 m/s for motor capabilities
omega_z = 0.0       # Desired angular velocity (rad/s)

# KINEMATICALLY CORRECT Wheel Angles (120 degree symmetry)
# W1: -90 deg, W2: 30 deg, W3: 150 deg (Relative to robot's forward X-axis)
theta_wheel_mount = np.array([-90, 30, 150]) * (np.pi / 180.0)

# --- VISUALIZATION CONSTANT (CRITICAL) ---
# Rotation needed to place the midpoint (-30 deg) of W1 and W2 at the plot's Top (90 deg).
VIS_ROTATION_RAD = 120.0 * (np.pi / 180.0)

# --- 2. Inverse Kinematics Matrix (Math Model) ---
J_inv = (1 / r) * np.array([
    [np.sin(theta_wheel_mount[0]), -np.cos(theta_wheel_mount[0]), -L],
    [np.sin(theta_wheel_mount[1]), -np.cos(theta_wheel_mount[1]), -L],
    [np.sin(theta_wheel_mount[2]), -np.cos(theta_wheel_mount[2]), -L]
])

# --- 3. Simulation Data Generation (Clockwise Order - FIXED START) ---
# Generate angles from 0 to 360 degrees going clockwise
# 0 degrees = top (pointing to blue dot), 90 degrees = right, 180 = bottom, 270 = left
alpha_degrees = np.arange(0, 361, 1)  # 0 to 360 degrees
alpha_radians = alpha_degrees * (np.pi / 180.0)

omega_w1, omega_w2, omega_w3 = [], [], []

for alpha_vis in alpha_radians:
    # Convert visual angle (0° = up) to math angle (0° = right)
    # Visual 0° (up) = Math 90° (up), so add 90°
    # Visual angles increase clockwise, math angles increase counter-clockwise
    alpha_math = np.pi/2 - alpha_vis  # Negate and add 90° to account for opposite rotation direction
    
    V_x = V_robot * np.cos(alpha_math)
    V_y = V_robot * np.sin(alpha_math)
    V_body = np.array([V_x, V_y, omega_z])
    
    omega_vec = J_inv @ V_body
    
    omega_w1.append(omega_vec[0])
    omega_w2.append(omega_vec[1])
    omega_w3.append(omega_vec[2])

omega_w1, omega_w2, omega_w3 = np.array(omega_w1), np.array(omega_w2), np.array(omega_w3)

# Reorder omega arrays to match visual wheel positions
# Visual order: W1 at ~60°, W2 at ~-60°, W3 at ~180° (bottom)
# Math order: index 0 = -90°, index 1 = 30°, index 2 = 150°
# Mapping: visual W1 → omega[1], visual W2 → omega[2], visual W3 → omega[0]
omega_w1_vis = omega_w2.copy()  # Visual W1 (at 60°) gets omega from math 30°
omega_w2_vis = omega_w3.copy()   # Visual W2 (at -60°) gets omega from math 150°
omega_w3_vis = omega_w1.copy()   # Visual W3 (at 180°) gets omega from math -90°


# --- 4. Visualization (Static Plot Function) ---
def plot_wheel_velocities(ax):
    """Generates the static plot of wheel velocities vs. driving direction."""
    # Use reordered omega values to match visual display
    # Visual W1 (at ~60°) → omega_w1_vis, Visual W2 (at ~-60°) → omega_w2_vis, Visual W3 (at ~180°) → omega_w3_vis
    ax.plot(alpha_degrees, omega_w1_vis, label='Wheel 1 ($\u03C9_1$ at ~60\u00B0)', linewidth=2)
    ax.plot(alpha_degrees, omega_w2_vis, label='Wheel 2 ($\u03C9_2$ at ~-60\u00B0)', linewidth=2)
    ax.plot(alpha_degrees, omega_w3_vis, label='Wheel 3 ($\u03C9_3$ at ~180\u00B0)', linewidth=2)

    ax.set_title('3-Wheel Omni Robot: Wheel Velocities (120\u00B0 Symmetric)')
    ax.set_xlabel('Driving Direction $\u03B1$ (degrees)')
    ax.set_ylabel('Required Wheel Angular Velocity $\u03C9$ (rad/s)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.legend(loc='upper left') 
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))

# --- 5. Visualization (Dynamic Robot Plot Setup) ---
fig = plt.figure(figsize=(14, 6))

# Subplot 1: Wheel Velocities
ax1 = fig.add_subplot(121)
plot_wheel_velocities(ax1)
# Initialize marker at the starting point (alpha=0, pointing up)
line_marker, = ax1.plot([alpha_degrees[0]] * 3, [omega_w1[0], omega_w2[0], omega_w3[0]], 
                        'o', color='black', markersize=6) 

# Subplot 2: Robot Visualization
ax2 = fig.add_subplot(122)
ax2.set_xlim(-1.5*L, 1.5*L)
ax2.set_ylim(-1.5*L, 1.5*L)
ax2.set_aspect('equal', adjustable='box')
ax2.set_title('Robot Direction ($\u03B1$) and Wheel Motor Directions')
ax2.axis('off')

# Draw Robot Body (Red Circle)
robot_body = plt.Circle((0, 0), L, color='red', fill=False, linewidth=2)
ax2.add_artist(robot_body)

# Draw Robot Front (Blue Dot at 0 degrees, at the top)
robot_front, = ax2.plot([0], [L], 'o', color='blue', markersize=8, zorder=5) 

# --- Initialize Direction Arrow ---
# Start at alpha=0 degrees (pointing up towards blue dot at top)
# Visual angle: 0° = up, 90° = right, 180° = down, 270° = left
# For visual alpha=0° (up), we want arrow to point straight up (0, 1) in plot coordinates
# Don't apply wheel rotation to movement direction - it should point directly in visual direction
alpha_start = 0.0  # Visual angle in degrees (0 = up)
alpha_start_rad = alpha_start * (np.pi / 180.0)

# Convert visual angle to plot coordinates
# Visual 0° = up = (0, 1) in plot coords
plot_x_start = V_robot * np.sin(alpha_start_rad)  # sin(0) = 0
plot_y_start = V_robot * np.cos(alpha_start_rad)  # cos(0) = 1 (points up)

# Draw Driving Direction Arrow pointing up initially (towards blue dot)
# Scale factor to make arrow longer (1.5x the original length)
direction_arrow_scale = 1.5
direction_arrow = ax2.quiver(0, 0, plot_x_start * L * direction_arrow_scale, plot_y_start * L * direction_arrow_scale, 
                            color='grey', scale=1, scale_units='xy', angles='xy', width=0.008)
# ------------------------------------------

# --- Add Wheel Markers and Labels ---
# Create mapping: visual wheel order (by position) to math angle index
# After rotation: W1 (math -90°) → visual ~60°, W2 (math 30°) → visual ~-60°, W3 (math 150°) → visual ~180°
# We need to map visual positions to the correct omega values
wheel_labels = ['W1', 'W2', 'W3']
wheel_positions = []  # Store wheel positions for arrow updates
wheel_math_indices = []  # Map visual wheel index to math angle index

for i, angle_rad in enumerate(theta_wheel_mount):
    wheel_x_math = L * np.cos(angle_rad)
    wheel_y_math = L * np.sin(angle_rad)

    # Apply 120 deg rotation to the wheel's position vector
    plot_x = wheel_x_math * np.cos(VIS_ROTATION_RAD) - wheel_y_math * np.sin(VIS_ROTATION_RAD)
    plot_y = wheel_x_math * np.sin(VIS_ROTATION_RAD) + wheel_y_math * np.cos(VIS_ROTATION_RAD)
    
    wheel_positions.append((plot_x, plot_y))  # Store position
    wheel_math_indices.append(i)  # Initially, visual index = math index
    
    ax2.plot(plot_x, plot_y, 'o', color='green', markersize=10, zorder=4)
    
    label_x = plot_x * 1.1 
    label_y = plot_y * 1.1 
    ax2.text(label_x, label_y, wheel_labels[i], color='black', fontsize=10, 
             ha='center', va='center', weight='bold')

# Note: omega arrays are already reordered above to match visual positions

# Initialize wheel rotation arrows (will be updated during animation)
# Each arrow shows the direction of wheel rotation/thrust based on omega sign
# Arrows are positioned at each wheel location
wheel_arrows = []
arrow_scale = 0.25 * L  # Scale factor for arrow length (relative to L)

for i, (wx, wy) in enumerate(wheel_positions):
    # Initialize arrow at wheel location (will be updated with direction)
    # Use slightly thicker arrows for better visibility
    arrow = ax2.quiver(wx, wy, 0, 0, scale=1, scale_units='xy', 
                       angles='xy', width=0.01, zorder=5, alpha=0.85)
    wheel_arrows.append(arrow)

# Text annotation for parameters
text_params = ax2.text(-1.6*L, 1.4*L, '', fontsize=9, 
                       verticalalignment='top', 
                       bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))

# Legend for wheel arrows at the bottom of the plot
wheel_legend_text = ax2.text(0, -1.4*L, 'Wheel Arrows: Red = Positive omega, Blue = Negative omega', 
                             fontsize=9, ha='center', va='center',
                             bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.7))


def update(frame):
    """Update function for the animation."""
    alpha_vis = alpha_radians[frame]  # Visual angle: 0° = up
    alpha_deg = alpha_degrees[frame]
    
    # 1. Update Velocity Plot Marker (ax1)
    # Use reordered omega values for display
    x_data = [alpha_deg] * 3 
    y_data = [omega_w1_vis[frame], omega_w2_vis[frame], omega_w3_vis[frame]]
    line_marker.set_data(x_data, y_data)
    
    # 2. Update Direction Arrow (ax2)
    # Visual angle: 0° = up (towards blue dot), 90° = right, 180° = down, 270° = left
    # Convert visual angle directly to plot coordinates (no wheel rotation needed)
    # Visual 0° (up) = (0, 1), 90° (right) = (1, 0), 180° (down) = (0, -1), 270° (left) = (-1, 0)
    plot_x = V_robot * np.sin(alpha_vis)  # sin(0) = 0, sin(90°) = 1, sin(180°) = 0, sin(270°) = -1
    plot_y = V_robot * np.cos(alpha_vis)  # cos(0) = 1, cos(90°) = 0, cos(180°) = -1, cos(270°) = 0
    
    # Scale the plot vector (use same scale factor as initial arrow)
    direction_arrow_scale = 1.5  # Make arrow 1.5x longer
    plot_x *= L * direction_arrow_scale
    plot_y *= L * direction_arrow_scale
    
    direction_arrow.set_UVC(plot_x, plot_y) 

    # 3. Update Wheel Rotation Arrows (ax2)
    # Show the direction the MOTOR/WHEEL is spinning (rotation direction)
    # 
    # The arrow shows which way the wheel/motor is physically rotating.
    # This is the tangential direction of rotation around the wheel's mounting axis.
    # 
    # Key points:
    # - The wheel rotates around its mounting axis (pointing from center to wheel)
    # - The arrow shows the tangential direction of this rotation
    # - Positive omega → arrow points one way (red)
    # - Negative omega → arrow points opposite way (blue)
    # - The direction depends on the wheel's mounting angle, NOT just "right = positive"
    # 
    # The kinematics formula tells us: positive omega creates motion in direction (sin(θ), -cos(θ))
    # This IS the tangential rotation direction we want to show.
    # 
    # Map visual wheel order to math angle order:
    # Visual W1 (at ~60°) → math index 1 (30°)
    # Visual W2 (at ~-60°) → math index 2 (150°)
    # Visual W3 (at ~180°) → math index 0 (-90°)
    omega_wheels_vis = [omega_w1_vis[frame], omega_w2_vis[frame], omega_w3_vis[frame]]
    math_indices_for_vis = [1, 2, 0]  # Visual wheel i uses math angle at this index
    
    for i, (omega, math_idx) in enumerate(zip(omega_wheels_vis, math_indices_for_vis)):
        # Get wheel position (already in visual coordinates)
        wx, wy = wheel_positions[i]
        
        # Calculate the wheel's rotation/thrust direction directly from visual position
        # The wheel is mounted perpendicular to the radial direction (from center to wheel)
        # When the wheel rotates, it exerts force in the TANGENTIAL direction
        # 
        # Radial direction (from center to wheel): (wx, wy) normalized
        # Tangential direction (perpendicular, for wheel rotation): (-wy, wx) or (wy, -wx)
        # 
        # For positive omega, we use one tangential direction
        # For negative omega, we use the opposite direction
        # 
        # Calculate radial vector magnitude for normalization
        radial_mag = np.sqrt(wx**2 + wy**2)
        
        if abs(omega) > 0.01 and radial_mag > 1e-6:
            # Calculate tangential direction directly from visual wheel position
            # The wheel is mounted at (wx, wy), pointing outward from center
            # The wheel rotates around the radial axis, creating thrust in the tangential direction
            # 
            # Radial unit vector: (wx, wy) / ||(wx, wy)||
            # Tangential unit vector (perpendicular): rotate radial 90° counter-clockwise
            # In 2D: rotate (x, y) by 90° CCW → (-y, x)
            # So tangential = (-radial_y, radial_x) = (-wy/||r||, wx/||r||)
            # 
            # Normalize radial vector
            radial_x = wx / radial_mag
            radial_y = wy / radial_mag
            
            # Tangential direction: perpendicular to radial
            # Two possible directions: rotate radial 90° CCW or 90° CW
            # We need to choose based on omega sign to match expected behavior
            # 
            # For positive omega: use one tangential direction
            # For negative omega: use the opposite tangential direction
            # 
            # Test shows: W1 (top-right, omega negative) should point left/up-left
            #             W2 (top-left, omega positive) should point right/up-right
            # 
            # Tangential 1: (-radial_y, radial_x) - rotate 90° CCW
            # Tangential 2: (radial_y, -radial_x) - rotate 90° CW
            # 
            # For positive omega, use tangential 2 (radial_y, -radial_x)
            # For negative omega, use tangential 1 (-radial_y, radial_x)
            # This matches expected behavior:
            #   W1 (top-right, omega negative) → points left/up-left
            #   W2 (top-left, omega positive) → points right/up-right
            if omega >= 0:
                tang_x = radial_y   # Rotate 90° CW
                tang_y = -radial_x
            else:
                tang_x = -radial_y  # Rotate 90° CCW
                tang_y = radial_x
            
            # The tangential direction is already normalized (since radial is normalized)
            thrust_dir_x_vis = tang_x
            thrust_dir_y_vis = tang_y
        else:
            thrust_dir_x_vis = 0
            thrust_dir_y_vis = 0
        
        # Calculate arrow length based on omega magnitude
        # Arrow length should be proportional to omega magnitude
        # Use a moderate scale factor to make magnitude differences noticeable but not excessive
        # For omega in rad/s, scale to a reasonable visual length
        # Reduced by half to keep arrows within plot boundaries
        omega_scale_factor = 0.02 * L  # Scale factor: 1 rad/s = 0.02*L length (reduced from 0.04)
        arrow_len = abs(omega) * omega_scale_factor
        
        # Minimum arrow length for very small omegas (but not for zero)
        if abs(omega) > 0.1:
            min_arrow_len = 0.02 * L  # Minimum for visibility (reduced proportionally)
            arrow_len = max(arrow_len, min_arrow_len)
        elif abs(omega) < 0.01:
            arrow_len = 0  # No arrow for zero omega
        else:
            # Very small omega, show tiny arrow
            arrow_len = 0.01 * L  # Reduced proportionally
        
        # Set arrow direction (thrust direction times length)
        arrow_dx = thrust_dir_x_vis * arrow_len
        arrow_dy = thrust_dir_y_vis * arrow_len
        
        # Color coding for visual feedback:
        # Red = positive omega (motor/wheel spinning in positive direction)
        # Blue = negative omega (motor/wheel spinning in negative/opposite direction)
        # Gray = zero or very small omega (motor not spinning)
        if abs(omega) < 0.01:
            wheel_arrows[i].set_color('lightgray')
            wheel_arrows[i].set_alpha(0.3)
        elif omega > 0:
            wheel_arrows[i].set_color('red')
            wheel_arrows[i].set_alpha(0.9)
        else:
            wheel_arrows[i].set_color('blue')
            wheel_arrows[i].set_alpha(0.9)
        
        # Update arrow vector to show wheel rotation direction
        wheel_arrows[i].set_UVC(arrow_dx, arrow_dy)

    # 4. Update Text Parameters (ax2)
    w1_val = omega_w1_vis[frame]
    w2_val = omega_w2_vis[frame]
    w3_val = omega_w3_vis[frame]
    
    # Text reflects the current state
    text_content = (
        f"--- Inputs ---\n"
        f"Robot Orient: {ROBOT_ORIENT:.1f}\u00B0\n"
        f"Driving Direction: {alpha_deg:.0f}\u00B0\n"
        f"Driving Speed: {V_robot:.1f} m/s\n"
        f"--- Outputs ---\n"
        f"$\\omega_1$ (-90\u00B0): {w1_val:.2f} rad/s\n"
        f"$\\omega_2$ (30\u00B0): {w2_val:.2f} rad/s\n"
        f"$\\omega_3$ (150\u00B0): {w3_val:.2f} rad/s"
    )
    
    text_params.set_text(text_content)
    
    return line_marker, direction_arrow, text_params, *wheel_arrows

# Create the animation
ani = FuncAnimation(fig, update, frames=len(alpha_degrees), interval=50, blit=False)

# Show the plot(s)
plt.tight_layout()
plt.show()