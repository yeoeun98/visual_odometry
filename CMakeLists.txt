cmake_minimum_required(VERSION 2.8)
project( mono-vo )
find_package( OpenCV 3.0 REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB viso
    "src/*.h"
    "src/*.cpp"
)
add_executable( visual_odometry ${viso} )
target_link_libraries( visual_odometry ${OpenCV_LIBS} )
