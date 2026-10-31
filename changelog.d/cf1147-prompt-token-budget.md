**Fix: character descriptions fit the 77 token limit (vivijure-cf#1147).**

The description came before the location-lock clause and pushed it out of the prompt on long shots, and on a two-person shot the first character took the whole budget. The lock clause now comes first, and what is left is split evenly across the shot's slots, cut at clause boundaries.
