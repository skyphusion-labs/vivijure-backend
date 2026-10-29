**Fix: anime `final` no longer auto-enables GFPGAN (#526).**

`FinishConfig.for_tier(final)` still turns GFPGAN on for photoreal.
Anime reads `style` from the finish or keyframe override.
No restorer pick keeps restore off.
An explicit restorer still wins.
