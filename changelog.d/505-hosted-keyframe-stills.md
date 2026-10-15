**Feat: hosted keyframes are full-step 1280x720 InstantID stills.**

The still the movie keeps is no longer an 8-step square. `KeyframeParams` and
`KeyframeConfig` (standard and final) are 30 steps, guidance 5.0, 1280x720, InstantID.
Draft stays the 4-step opt-in. One person takes InstantID even when scene lock is on.
Two people stay on masked IP-Adapter. A `lora_only` still is refused before it is saved.
The bake manifest drops Wan2.2-I2V and Wan2.2-Lightning. The image-to-video code path
stays; this change does not rebuild the image.
