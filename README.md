# Burger's equation DL solver

Physics differential equations are a great opportunity to play with DL frameworks' auto-differentiation features and develop a poorly ambitious solver from scratch. On the pros side *it's fun, interpolation comes for free, and it's a great pretext for practicing the frameworks* — the cons are it's sillily unefficient and practically useless. Still enjoyable though.

* `1d` for 1D case. Actually usable.
* `3d` for 3D case. Don't use this. It will generate ~35PiB of data with just `n = 100` and smoke your hardware for a laugh.