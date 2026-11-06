### fix(keyframe): `multi_char.ip_adapter_scale_per_slot` reaches the two-person draw

The field was documented (default 0.7), parsed and hashed, but never reached the renderer: both two-person paths (scene-lock Pass B and the plain regional render) drew at `keyframe.ip_adapter_scale`, 0.65. Setting it forced a redraw that drew the same image. It now reaches both paths. Unset, the new default, falls back to `keyframe.ip_adapter_scale`, so a job that does not set it draws and hashes exactly as before and no stored still redraws. A job that sets it draws at that scale, and its stills redraw.
