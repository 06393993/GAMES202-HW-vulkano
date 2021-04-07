# GAMES202-HW-vulkano

The goal of this project is to port the homework of [GAMES202](http://games-cn.org/games202/) to
the Rust and Vulkan platform on Windows. IMGUI is used to draw UI controls.

## Run

1. Follow [the setup of vulkano](https://github.com/vulkano-rs/vulkano#setup) on Windows. This
   project uses vulkano to work with Vulkan.
1. `cargo run`.

## Control

* WASD will move the camera forward, backward, left, and right respectively.
* Hold the middle button of the mouse and move the mouse will change the angle of the camera.

## TODO

1. Better UI
    * Display and edit the camera parameters.
    * Display and edit the TRSTransform for the model.
    * Display and edit the point light TRSTransform and material parameters.
1. Support for window resizing.
1. Support for multiple objects, multiple models.
1. Support for saving and loading a scene.
