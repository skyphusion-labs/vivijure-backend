**Fix: the keyframe identity prompt carries each slot's character description (vivijure-cf#1147).**

The prompt was style, scene and LoRA triggers, so a shot that named only the character lost the hair and wardrobe the cast member is defined with. The description now follows the triggers on the identity pass and is left out of the plate pass.
