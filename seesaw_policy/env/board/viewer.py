"""Adjustable seesaw viewer for the student; no teacher runner."""

import math
import time
from types import MethodType
import numpy as np
import torch
import viser
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.viewer_config import ViewerConfig
from seesaw_policy.env.boarding.parameters import ANGLES_DEG
from .asset import arena_position
from .board_profiles import effective_board_config


def set_play_board_geometry(runtime, length, width, angle_deg):
    values = dict(length=float(length), width=float(width), angle_deg=float(angle_deg))
    bounds = dict(length=(1.5, 3.6), width=(0.2, 0.6), angle_deg=(1.0, 35.0))
    for name, value in values.items():
        low, high = bounds[name]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    runtime.play_board_geometry = values


def track_board_geometry_changes(scene):
    # MJLab tracks baked appearance fields, but geom_size is also baked into
    # mesh handles. Include it so reset widths rebuild visible mesh variants.
    scene._baked_appearance_fields = scene._baked_appearance_fields | {"geom_size"}
    scene._baked_appearance_fingerprint = scene._appearance_fingerprint()


class SeesawViewer(ViserPlayViewer):
    def setup(self):
        super().setup()
        # max_extra_envs controls neighboring-world overlays, while mjviser's
        # batched meshes have a separate selection switch. These worlds use
        # the same map coordinates and must not be rendered on top of each other.
        self._scene.show_only_selected = True
        for handle in self._server.gui._gui_input_handle_from_uuid.values():
            if (
                isinstance(handle, viser.GuiCheckboxHandle)
                and handle.label == "Hide others"
            ):
                handle.value = True
        track_board_geometry_changes(self._scene)
        self.pause()
        self._pause_button.label = "Play"
        self._pause_button.icon = viser.Icon.PLAYER_PLAY
        geometry = self.env.task_env
        runtime = geometry.seesaw_runtime
        with self._server.gui.add_folder("Seesaw inspection"):
            self._server.gui.add_markdown(
                "Starts **paused**; no training. Select a level and click **Reset**. "
                "**Play** runs the selected checkpoint. Only seesaw episodes are created."
            )
            choice = self._server.gui.add_dropdown(
                "Board level",
                options=tuple(f"L{i}" for i in range(10)),
                initial_value="L0",
            )
            custom = self._server.gui.add_checkbox(
                "Override board geometry", initial_value=False
            )
            selected = int(self._scene.env_idx)
            length = self._server.gui.add_slider(
                "Board length (m)",
                min=1.5,
                max=3.6,
                step=0.05,
                initial_value=float(
                    np.clip(float(runtime.board_half_length[selected]) * 2, 1.5, 3.6)
                ),
            )
            width = self._server.gui.add_slider(
                "Board width (cm)",
                min=20.0,
                max=60.0,
                step=1.0,
                initial_value=float(
                    np.clip(float(runtime.board_half_width[selected]) * 200, 20.0, 60.0)
                ),
            )
            angle = self._server.gui.add_slider(
                "Maximum tilt (deg)",
                min=1.0,
                max=35.0,
                step=0.5,
                initial_value=float(
                    np.clip(
                        float(runtime.board_max_angle[selected]) * 180 / math.pi,
                        1.0,
                        35.0,
                    )
                ),
            )
            self._server.gui.add_markdown(
                "Enable **Override board geometry**, adjust the sliders, then click **Apply and reset**. "
                "Disable it and reset to restore checkpoint randomization."
            )
            reset = self._server.gui.add_button("Apply and reset")
            overview = self._server.gui.add_button("Overview of ten boards")
            focus = self._server.gui.add_button("Focus selected board")
            self._server.gui.add_markdown(
                "Default level tilt (custom settings override this):\n"
                + "\n".join(f"- L{i}: {ANGLES_DEG[i]:.2f}° stop" for i in range(10))
            )
            current = self._server.gui.add_markdown("")
            dr_config = effective_board_config(
                runtime.config.get("domain_rand", {}), runtime.training_iteration
            )
            if dr_config.get("enabled", False):
                ramp = dr_config.get("seesaw3_5")
                if ramp:
                    self._server.gui.add_markdown(
                        f"{ramp['base_fraction']:.0%} of resets use the base ranges; "
                        "the remaining resets use the current ramp ranges below."
                    )
                self._server.gui.add_markdown(
                    "Independent uniform samples on each reset:\n"
                    + "\n".join(
                        f"- {name}: {dr_config[key][0]}–{dr_config[key][1]} {unit}"
                        for name, key, unit in (
                            ("Width", "width_range_m", "m"),
                            ("Mass", "mass_range_kg", "kg"),
                            ("Hinge damping", "damping_range", "N m s/rad"),
                            ("Sliding friction", "friction_range", ""),
                        )
                    )
                )

        def focus_camera(client, level):
            x, y, _ = arena_position(
                level, runtime.config.get("arena_offset", (0.0, 0.0, 0.0))
            )
            offset = self._scene._scene_offset
            client.camera.position = np.array((x + 4.0, y - 5.0, 3.5)) + offset
            client.camera.look_at = np.array((x, y, 0.5)) + offset

        @reset.on_click
        def reset_level(_):
            level = int(choice.value[1:])
            with self._sim_lock:
                runtime.play_levels.fill_(level)
                if custom.value:
                    set_play_board_geometry(
                        runtime, length.value, width.value / 100.0, angle.value
                    )
                else:
                    runtime.play_board_geometry = None
                self.env.reset()
                self.policy.reset()
                self._scene.update(self.env.unwrapped.sim.data)
                self._scene.request_update()
            for client in self._server.get_clients().values():
                focus_camera(client, level)

        @focus.on_click
        def focus_level(_):
            for client in self._server.get_clients().values():
                focus_camera(client, int(choice.value[1:]))

        @overview.on_click
        def show_overview(_):
            for client in self._server.get_clients().values():
                client.camera.position = (
                    np.array(runtime.config.get("arena_offset", (0.0, 0.0, 0.0)))
                    + np.array((16.0, -20.0, 23.0))
                    + self._scene._scene_offset
                )
                client.camera.look_at = (
                    np.array(runtime.config.get("arena_offset", (0.0, 0.0, 0.0)))
                    + np.array((16.0, 2.5, 0.3))
                    + self._scene._scene_offset
                )

        labels = []
        for level in range(10):
            x, y, _ = arena_position(
                level, runtime.config.get("arena_offset", (0.0, 0.0, 0.0))
            )
            handle = self._server.scene.add_label(
                f"/seesaw_labels/L{level}",
                text=f"L{level} | {ANGLES_DEG[level]:.1f}°",
                position=(x, y + 0.8, 1.3),
            )
            labels.append((handle, np.array((x, y + 0.8, 1.3))))
        original_update = self._scene.update
        last_panel_update = 0.0

        def update_with_labels(scene, *args, **kwargs):
            nonlocal last_panel_update
            result = original_update(*args, **kwargs)
            now = time.perf_counter()
            if now - last_panel_update < 0.1:
                return result
            last_panel_update = now
            index = self._scene.env_idx
            geoms = runtime.board_randomization.board_geoms
            sizes = runtime.env.sim.model.geom_size[index, geoms].cpu().numpy() * 2
            limits = (
                runtime.env.sim.model.jnt_range[
                    index, runtime.board_randomization.joint_ids, 1
                ]
                .cpu()
                .numpy()
                * 180
                / math.pi
            )
            for level, (handle, position) in enumerate(labels):
                handle.position = position + scene._scene_offset
                handle.text = f"L{level} | {sizes[level, 0]:.2f} m × {sizes[level, 1]*100:.0f} cm | {limits[level]:.1f}°"
            current.content = (
                f"Current: **L{int(runtime.levels[index])}**, "
                f"width **{float(runtime.board_half_width[index])*2:.3f} m**, "
                f"length **{float(runtime.board_half_length[index])*2:.3f} m**, "
                f"maximum tilt **{float(runtime.board_max_angle[index])*180/math.pi:.2f}°**, "
                f"current angle **{float(runtime.board_angle[index])*180/math.pi:.2f}°**, "
                f"mass **{float(runtime.board_mass[index]):.2f} kg**, "
                f"damping **{float(runtime.board_damping[index]):.3f}**, "
                f"friction **{float(runtime.board_friction[index]):.3f}**"
            )
            return result

        self._scene.update = MethodType(update_with_labels, self._scene)
        # A paused native viewer otherwise initially displays mesh handles at
        # their construction poses until the first reset/physics step.
        self._scene.update(self.env.unwrapped.sim.data)
        self._scene.request_update()
