##### Functions in `onnx-to-trt-inference.cpp` #####

### 1. `process_img`
- Reads an image from the given path and returns a normalized grayscale OpenCV Mat.
  
### 2. `get_input_shape_from_image`
- Returns the shape of the input image as a 4D vector (NCHW).

### 3. `CustomLogger`
- Custom Logger class for TensorRT, primarily for debugging purposes.
  
### 4. `measure_trt_performance`
- Measures the performance of TensorRT inference engine using the given TRT plan and image tensor.
  
### 5. `gen_trt_plan`
- Generates a TensorRT plan given the ONNX model path, plan file path, precision mode, and optional dynamic shape.



##### How to Build ####

1. ***Create a Build Directory***
    ```
    mkdir build
    cd build
    ```

2. ***Run CMake Command***
    - **Windows PowerShell:**
        ```
        cmake -DCMAKE_PREFIX_PATH=<put your path for torch cmake on device here> .. ; cmake --build . --config Release
        e.g. cmake -DCMAKE_PREFIX_PATH=C:\Users\Gemini\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\share\cmake .. ; cmake --build . --config Release
        ```
    - **Windows CMD:**
        ```
        cmake -DCMAKE_PREFIX_PATH=<put your path for torch cmake on device here> .. && cmake --build . --config Release
        e.g. cmake -DCMAKE_PREFIX_PATH=C:\Users\Gemini\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\share\cmake .. && cmake --build . --config Release
        ```
  
3. ***Run the Executable***
    - Executable will be generated in the `Release` folder within the `build` directory.

---

That should cover how to use and build the code using CMakeLists.txt. Let me know if you have more questions!