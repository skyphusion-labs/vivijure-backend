**Feat: the scene-lock identity crop padding is a knob, `keyframe.face_crop_pad`.**

Pass B gives each person's IP-Adapter a crop of their cast portrait padded by this fraction of the face box on each side. The default stays 0.35, the crop shipped before, so nothing changes until it is set. Measured at seed 110501 on 1.1.6 (vivijure-cf#1143): the two-shot lost Mara's hair on the scene-lock path, whose 0.35 crop holds little hair, and kept it on the path that takes the whole portrait. The knob lets one image be swept live. It is clamped to 0..2 and is part of the keyframe reuse hash.
