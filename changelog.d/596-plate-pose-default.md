### feat(keyframe): a two-person scene-lock plate draws without the pose skeleton by default

Under the fixed standing skeleton, a two-person scene-lock plate posed the pair for the camera instead of drawing the shot's action, and Pass B kept that pose through canny. On vivijure-cf#1281 (18 jobs, one board, 6 seeds), 2 of 6 stills held the action with the right people at today's default, against 5 of 6 with no skeleton. The plate count guard caught every third person.

So `multi_char.pose_conditioning` now defaults to unset, which means each path's own default:
- A scene-lock plate for two-plus people draws without the pose ControlNet while the plate count guard runs.
- With `plate_count_retries` 0, it keeps the pose, so pose off with the guard off is never a default.
- The plain regional path (scene lock off) keeps its pose.
- An explicit `true` or `false` sets both paths, so a job can turn the plate's pose back on.

Reuse hash: only a two-plus-person, scene-lock, guarded still with the pose unset gets a new hash and redraws. Every other stored still keeps its hash, held by tests against values computed on the previous code.
