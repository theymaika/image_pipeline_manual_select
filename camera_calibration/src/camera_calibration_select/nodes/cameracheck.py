#!/usr/bin/env python
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2009, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of the Willow Garage nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import rclpy
from camera_calibration_select.camera_checker import CameraCheckerNode


def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-p", "--pattern", default="chessboard", help="specify calibration pattern type [default: %default]")
    parser.add_option("-s", "--size", default="8x6", help="specify chessboard size (inner corners only) as rows x columns [default: %default]. For ChArUco boards, this is the total size of the board not the number of inner corners. Must be rows x columns.")
    parser.add_option("-q", "--square", default=".108", help="specify chessboard square size in meters [default: %default]")
    parser.add_option("--approximate",
                      type="float", default=0.0,
                      help="allow specified slop (in seconds) when pairing images from unsynchronized stereo cameras")
    parser.add_option("--queue_size", type="int", default=1, help="size of the queue for synchronizing images")
    parser.add_option("-o", "--error_output_filename", default="", help="output file name for reprojection error data")
    parser.add_option("-t", "--camera_type", default="pinhole", help="specify camera model type (pinhole, fisheye) [default: %default]")
    #ChAruco options
    parser.add_option("-m", "--charuco_marker_size",
                     action="append", default=[],
                     help="ArUco marker size (meters); only valid with `-p charuco`")
    parser.add_option("-d", "--aruco_dict",
                     action="append", default=[],
                     help="ArUco marker dictionary; only valid with `-p charuco`; one of 'aruco_orig', '4x4_250', " +
                     "'5x5_250', '6x6_250', '7x7_250'")

    options, _ = parser.parse_args(rclpy.utilities.remove_ros_args())

    size = tuple([int(c) for c in options.size.split('x')])
    dim = float(options.square)
    approximate = float(options.approximate)
    queue_size = int(options.queue_size)
    
    rclpy.init()
    node = CameraCheckerNode("cameracheck", size, dim, approximate, queue_size, options.error_output_filename, options.pattern, options.camera_type, options.charuco_marker_size, options.aruco_dict)
    node.spin()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
