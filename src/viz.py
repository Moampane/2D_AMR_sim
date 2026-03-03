import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import pickle
from pathlib import Path
from utils import Pose, Landmark


class Visualizer:
    """
    Visualizer for Kalman Filter trajectory estimation results.
    """

    def __init__(
        self,
        output_path: Path,
    ):
        """
        Initialize the visualizer class.
        """
        self.output_path = output_path
        gt_log_path = output_path / "gt_data.csv"
        sensor_log_path = output_path / "sensor_data.csv"
        env_info_path = output_path / "env_data.csv"
        kf_log_path = output_path / "kf_data.csv"
        
        self.gt_log = pd.read_csv(gt_log_path)
        self.sensor_log = pd.read_csv(sensor_log_path)
        self.env_info = pd.read_csv(env_info_path)
        self.kf_log = pd.read_csv(kf_log_path)

    def plot_env(self):
        """
        Plot the environment features with no trajectories.
        """
        # set up axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_title(f"Environment Map")

        # Set up the plot boundaries
        world_min_x, world_max_x = self.env_info["min_x"].iloc[0], self.env_info["max_x"].iloc[0]
        world_min_y, world_max_y = self.env_info["min_y"].iloc[0], self.env_info["max_y"].iloc[0]
        ax.set_xlim(world_min_x - 1, world_max_x + 1)
        ax.set_ylim(world_min_y - 1, world_max_y + 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # set up env boundaries
        width = world_max_x - world_min_x
        height = world_max_y - world_min_y
        walls = patches.Rectangle(
            (world_min_x, world_min_y),
            width,
            height,
            linewidth=5,
            edgecolor="black",
            facecolor="none",
            alpha=1.0,
        )
        ax.add_patch(walls)

        # Plot obstacles
        num_obs = 2
        for i in range(num_obs):
            width = self.env_info[f"obs_{i}_max_x"].iloc[0] - self.env_info[f"obs_{i}_min_x"].iloc[0]
            height = self.env_info[f"obs_{i}_max_y"].iloc[0] - self.env_info[f"obs_{i}_min_y"].iloc[0]
            rect = patches.Rectangle(
                (self.env_info[f"obs_{i}_min_x"].iloc[0], self.env_info[f"obs_{i}_min_y"].iloc[0]),
                width,
                height,
                linewidth=2,
                edgecolor="black",
                facecolor="gray",
                alpha=0.5,
                label="Obstacle" if i == 0 else None,
            )
            ax.add_patch(rect)

        # Plot landmarks
        all_lm_cols = [col for col in self.env_info.columns if col.startswith("lm_")]
        num_landmarks = len(all_lm_cols) // 2
        for i in range(num_landmarks):
            # Plot pinging range circle
            circle = patches.Circle(
                (self.env_info[f"lm_{i}_x"].iloc[0], self.env_info[f"lm_{i}_y"].iloc[0]),
                self.env_info["pinger_range"].iloc[0],
                linewidth=1,
                edgecolor="red",
                facecolor="red",
                alpha=0.1,
                label="Pinging Range" if i == 0 else None,
            )
            ax.add_patch(circle)

            # plot floating point landmarks
            ax.plot(
                self.env_info[f"lm_{i}_x"].iloc[0],
                self.env_info[f"lm_{i}_y"].iloc[0],
                "r*",
                markersize=15,
                label="Landmark" if i == 0 else None,
            )
            ax.annotate(
                f"LM{i}",
                (self.env_info[f"lm_{i}_x"].iloc[0], self.env_info[f"lm_{i}_y"].iloc[0]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                color="red",
            )

        ax.legend(loc="lower right")
        return fig, ax
    
    def poses_from_odom(self):
        """
        Computes estimated pose by integrating odometry velocity over time.
        """
        # Get initial starting position from Ground Truth (GT)
        x = self.gt_log["x"].iloc[0]
        y = self.gt_log["y"].iloc[0]
        theta = self.gt_log["Theta"].iloc[0]
        
        poses = []
        # Use timestep from env_info (ensure it's a scalar)
        dt = self.env_info["timestep_size"].iloc[0]

        # Iterate through sensor logs
        for row in self.sensor_log.itertuples():
            # Map CSV columns to variables
            # Using getattr or dot notation based on your CSV headers
            vx = row.Odometry_x_vel
            vy = row.Odometry_y_vel
            w = row.Odometry_ang_vel

            # Handle potential NaN/empty values in the first row
            if np.isnan(vx): vx, vy, w = 0.0, 0.0, 0.0

            # Dead reckoning integration
            x += vx * dt
            y += vy * dt
            theta += w * dt

            # 4. Wrap theta to [-pi, pi]
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            poses.append({"Time": row.Time, "x": x, "y": y, "theta": theta})

        return pd.DataFrame(poses)

    def poses_from_gt(self):
        """
        Extracts Time, x, y, and theta from the ground truth log.
        Converts time from decaseconds to seconds and renames Theta.
        """
        # Get relevant columns
        df_poses = self.gt_log[['Time', 'x', 'y', 'Theta']].copy()

        # Rename 'Theta' to 'theta' to match your required output
        df_poses = df_poses.rename(columns={'Theta': 'theta'})

        return df_poses
    
    def poses_from_kf(self):
        # Get relevant columns
        return self.kf_log[['x', 'y', 'theta']].copy()

    def poses_from_gps(self):
        """
        Adjusted to read from CSV columns: Time, GPS_x, GPS_y
        Returns a DataFrame: Time | x | y | theta
        """
        # Filter the sensor log to only rows where GPS_x is not NaN
        gps_data = self.sensor_log.dropna(subset=['GPS_x', 'GPS_y'])

        # Create the poses DataFrame 
        poses = pd.DataFrame({
            "Time": gps_data["Time"],
            "x": gps_data["GPS_x"],
            "y": gps_data["GPS_y"],
            "theta": np.nan  # GPS doesn't measure heading
        })

        return poses.reset_index(drop=True)

    def plot_single_trajectory(
        self,
        label,
        pose_table: pd.DataFrame,
        color="blue",
        alpha=1.0,
        scatter=False,
    ):
        """
        Given a DataFrame of robot pose data with the following columns:
        Time | x | y | theta
        Plot the trajectory as a quiver on an xy grid with theta as arrow angle.
        """
        # Get current axes or create new ones
        if plt.get_fignums():
            ax = plt.gca()
        else:
            fig, ax = self.plot_env()

        # Plot trajectory path
        ax.plot(
            pose_table["x"],
            pose_table["y"],
            "-",
            color=color,
            linewidth=2,
            label=label,
            alpha=alpha,
        )

        # Plot arrows showing heading at intervals
        # Show arrows every N points to avoid clutter
        skip = max(1, len(pose_table) // 20)

        for idx in range(0, len(pose_table), skip):
            row = pose_table.iloc[idx]
            dx = 0.5 * np.cos(row["theta"])
            dy = 0.5 * np.sin(row["theta"])

            ax.arrow(
                row["x"],
                row["y"],
                dx,
                dy,
                head_width=0.3,
                head_length=0.2,
                fc=color,
                ec=color,
                alpha=alpha * 0.25,
            )
        if scatter:
            ax.scatter(
                pose_table["x"],
                pose_table["y"],
                marker="*",
                color=color,
                linewidth=2,
                alpha=alpha,
            )

        # Mark start and end positions
        start = pose_table.iloc[0]
        end = pose_table.iloc[-1]

        ax.plot(
            start["x"],
            start["y"],
            "o",
            color=color,
            markersize=10,
            alpha=alpha,
        )
        ax.plot(
            end["x"],
            end["y"],
            "s",
            color=color,
            markersize=10,
            alpha=alpha,
        )

        ax.legend(loc="lower right")
        return ax

    def draw_all(self):
        """
        Docstring for draw_all

        :param self: Description
        """
        self.plot_env()
        self.plot_single_trajectory(
            "Ground Truth",
            self.poses_from_gt(),
            "green",
        )
        self.plot_single_trajectory(
            "Dead Reckoning",
            self.poses_from_odom(),
            "red",
        )
        self.plot_single_trajectory(
            "GPS Only",
            self.poses_from_gps(),
            "orange",
            scatter=True,
        )
        self.plot_single_trajectory(
            "Prediction",
            self.poses_from_kf(),
            "blue",
        )
        plt.savefig(self.output_path / "dataset_viz.png")
        print("Finished plotting at path: ")
        print(self.output_path / "dataset_viz.png")

    def animate_trajectories(
        self,
        fps=30,
        speedup=3.0,
        linger_seconds=5.0,
    ):
        """
        Create an animated GIF showing trajectories being drawn over time.

        Args:
            fps: Frames per second for the animation
            save_path: Where to save the GIF
            speedup: Speed multiplier (2.0 = 2x faster, 0.5 = half speed)
            linger_seconds: How long to hold on final frame with end markers
        """
        # Get all trajectory data
        gt_poses = self.poses_from_gt()
        odom_poses = self.poses_from_odom()
        gps_poses = self.poses_from_gps()

        # Find the maximum number of frames needed
        max_frames = max(len(gt_poses), len(odom_poses))

        # Apply speedup by sampling fewer frames
        frame_skip = int(speedup)
        frame_indices = list(range(0, max_frames, max(1, frame_skip)))

        # Add linger frames at the end (repeat last frame)
        linger_frames = int(linger_seconds * fps)
        frame_indices.extend([frame_indices[-1]] * linger_frames)

        # Initialize the plot
        fig, ax = self.plot_env()

        # Initialize line objects for each trajectory
        (gt_line,) = ax.plot(
            [], [], "-", color="green", linewidth=2, label="Ground Truth", alpha=0.8
        )
        (odom_line,) = ax.plot(
            [], [], "-", color="red", linewidth=2, label="Dead Reckoning", alpha=0.8
        )
        (gps_line,) = ax.plot([], [], "-", color="orange", linewidth=2, alpha=0.8)
        gps_scatter = ax.scatter(
            [],
            [],
            c="orange",
            s=50,
            marker="x",
            label="GPS Measurements",
            alpha=0.6,
            zorder=5,
        )

        # Initialize end marker objects (hidden initially)
        gt_end = ax.plot([], [], "s", color="green", markersize=10, alpha=0)[0]
        odom_end = ax.plot([], [], "s", color="red", markersize=10, alpha=0)[0]
        gps_end = ax.plot([], [], "s", color="orange", markersize=10, alpha=0)[0]

        # Add time display
        time_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.legend(loc="lower right")

        def init():
            """Initialize animation"""
            gt_line.set_data([], [])
            odom_line.set_data([], [])
            gps_line.set_data([], [])
            gps_scatter.set_offsets(np.empty((0, 2)))
            gt_end.set_data([], [])
            odom_end.set_data([], [])
            gps_end.set_data([], [])
            time_text.set_text("")
            return (
                gt_line,
                odom_line,
                gps_line,
                gps_scatter,
                gt_end,
                odom_end,
                time_text,
            )

        def animate(frame_idx):
            """Update function for each frame"""
            actual_frame = (
                frame_indices[frame_idx]
                if frame_idx < len(frame_indices)
                else frame_indices[-1]
            )
            is_final_frame = frame_idx >= len(frame_indices) - linger_frames

            # Update ground truth
            if actual_frame < len(gt_poses):
                gt_data = gt_poses.iloc[: actual_frame + 1]
                gt_line.set_data(gt_data["x"], gt_data["y"])
                current_time = gt_data.iloc[-1]["Time"]
                time_text.set_text(f"Time: {current_time:.1f}s")

                # Show end marker on final frames
                if is_final_frame:
                    gt_end.set_data([gt_data.iloc[-1]["x"]], [gt_data.iloc[-1]["y"]])
                    gt_end.set_alpha(0.8)

            # Update dead reckoning
            if actual_frame < len(odom_poses):
                odom_data = odom_poses.iloc[: actual_frame + 1]
                odom_line.set_data(odom_data["x"], odom_data["y"])

                # Show end marker on final frames
                if is_final_frame:
                    odom_end.set_data(
                        [odom_data.iloc[-1]["x"]], [odom_data.iloc[-1]["y"]]
                    )
                    odom_end.set_alpha(0.8)

            # Update GPS measurements synchronized by time
            # Find GPS measurements up to the current time
            if actual_frame < len(gt_poses):
                current_time = gt_poses.iloc[actual_frame]["Time"]
                gps_up_to_now = gps_poses[gps_poses["Time"] <= current_time]
                if len(gps_up_to_now) > 0:
                    gps_line.set_data(gps_up_to_now["x"], gps_up_to_now["y"])
                    gps_scatter.set_offsets(gps_up_to_now[["x", "y"]].values)

            return gt_line, odom_line, gps_scatter, gt_end, odom_end, time_text

        # Create animation
        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(frame_indices),
            interval=1000 / fps,  # milliseconds between frames
            blit=True,
            repeat=True,
        )

        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(self.output_path / "trajectory_animation.gif", writer=writer)
        plt.close(fig)
        print("Finished animating at path: ")
        print(self.output_path / "trajectory_animation.gif")
        return anim