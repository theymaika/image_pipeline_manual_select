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

import cv2
import cv_bridge
import functools
import message_filters
import numpy
import rclpy
from rclpy.node import Node
import sensor_msgs.msg
import sensor_msgs.srv
import threading
from time import sleep

from camera_calibration_select.calibrator import MonoCalibrator, StereoCalibrator, ChessboardInfo, MonoDrawable, StereoDrawable, Patterns, CAMERA_MODEL
from message_filters import ApproximateTimeSynchronizer

try:
    from queue import Queue
except ImportError:
    from Queue import Queue


def mean(seq):
    return sum(seq) / len(seq)


def lmin(seq1, seq2):
    """ Pairwise minimum of two sequences """
    return [min(a, b) for (a, b) in zip(seq1, seq2)]


def lmax(seq1, seq2):
    """ Pairwise maximum of two sequences """
    return [max(a, b) for (a, b) in zip(seq1, seq2)]


class SpinThread(threading.Thread):
    """
    Thread that spins the ros node, while imshow runs in the main thread
    """

    def __init__(self, node):
        threading.Thread.__init__(self)
        self.node = node

    def run(self):
        rclpy.spin(self.node)


class ConsumerThread(threading.Thread):
    def __init__(self, queue, function):
        threading.Thread.__init__(self)
        self.queue = queue
        self.function = function

    def run(self):
        while rclpy.ok():
            m = self.queue.get()
            # if self.queue.empty():
            #     continue
            self.function(m)


class BufferQueue(Queue):
    """Slight modification of the standard Queue that discards the oldest item
    when adding an item and the queue is full.
    """

    def put(self, item, *args, **kwargs):
        # The base implementation, for reference:
        # https://github.com/python/cpython/blob/2.7/Lib/Queue.py#L107
        # https://github.com/python/cpython/blob/3.8/Lib/queue.py#L121
        with self.mutex:
            if self.maxsize > 0 and self._qsize() == self.maxsize:
                self._get()
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class CameraCheckerNode(Node):
    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    def __init__(self, name, chess_size, dim, approximate=0, queue_size=1, error_output_file="camera", pattern="chessboard", camera_type='pinhole', charuco_marker_size=[], aruco_dict=[]):
        super().__init__(name)

        if pattern == 'charuco':
            if not charuco_marker_size:
                raise ValueError(
                    "Charuco pattern requires --charuco_marker_size to be specified.")
            if not aruco_dict:
                raise ValueError(
                    "Charuco pattern requires --aruco_dict to be specified.")
        self.board = ChessboardInfo(pattern,  n_cols=chess_size[0], n_rows=chess_size[1], dim=dim, marker_size=float(
            charuco_marker_size[0]), aruco_dict=aruco_dict[0])

        self.linearity_rms_errors: list[float] = []
        self.reproj_rms_errors: list[float] = []
        self._last_display = None
        self.current_image_height = 0
        self.current_image_width = 0
        self.enough_data = False
        self.error_output_filename = error_output_file

        if pattern == "chessboard":
            pattern_type = Patterns.Chessboard
        elif pattern == "charuco":
            pattern_type = Patterns.ChArUco
        elif pattern == "acircles":
            pattern_type = Patterns.ACircles
        elif pattern == "circles":
            pattern_type = Patterns.Circles
        else:
            raise ValueError(f"Unsupported pattern type: {pattern}")

        if camera_type == "pinhole":
            self.camera_model = CAMERA_MODEL.PINHOLE
        elif camera_type == "fisheye":
            self.camera_model = CAMERA_MODEL.FISHEYE
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")

        # make sure n_cols is not smaller than n_rows, otherwise error computation will be off
        if self.board.n_cols < self.board.n_rows:
            self.board.n_cols, self.board.n_rows = self.board.n_rows, self.board.n_cols

        image_topic = "monocular/image_rect"
        camera_topic = "monocular/camera_info"

        tosync_mono = [
            (image_topic, sensor_msgs.msg.Image),
            (camera_topic, sensor_msgs.msg.CameraInfo),
        ]

        if approximate <= 0:
            sync = message_filters.TimeSynchronizer
        else:
            sync = functools.partial(
                ApproximateTimeSynchronizer, slop=approximate)

        tsm = sync([message_filters.Subscriber(self, type, topic)
                   for (topic, type) in tosync_mono], 10)
        tsm.registerCallback(self.queue_monocular)

        left_topic = "stereo/left/image_rect"
        left_camera_topic = "stereo/left/camera_info"
        right_topic = "stereo/right/image_rect"
        right_camera_topic = "stereo/right/camera_info"

        tosync_stereo = [
            (left_topic, sensor_msgs.msg.Image),
            (left_camera_topic, sensor_msgs.msg.CameraInfo),
            (right_topic, sensor_msgs.msg.Image),
            (right_camera_topic, sensor_msgs.msg.CameraInfo)
        ]

        tss = sync([message_filters.Subscriber(self, type, topic)
                   for (topic, type) in tosync_stereo], 10)
        tss.registerCallback(self.queue_stereo)

        self.br = cv_bridge.CvBridge()

        self.q_mono = BufferQueue(queue_size)
        self.q_stereo = BufferQueue(queue_size)

        mth = ConsumerThread(self.q_mono, self.handle_monocular)
        mth.daemon = True
        mth.start()

        sth = ConsumerThread(self.q_stereo, self.handle_stereo)
        sth.daemon = True
        sth.start()

        self._queue_display = BufferQueue(maxsize=1)
        self.initWindow()

        self.mc = MonoCalibrator([self.board], pattern=pattern_type)
        self.sc = StereoCalibrator([self.board], pattern=pattern_type)

    def spin(self):
        rclpy_spin_thread = SpinThread(self)
        rclpy_spin_thread.start()

        while rclpy.ok():
            if self._queue_display.qsize() > 0:
                self.image = self._queue_display.get()
                cv2.imshow("display", self.image)
            else:
                sleep(0.1)
            k = cv2.waitKey(6) & 0xFF
            if k in [27, ord('q')]:
                return

    def queue_monocular(self, msg, cmsg):
        self.q_mono.put((msg, cmsg))

    def queue_stereo(self, lmsg, lcmsg, rmsg, rcmsg):
        self.q_stereo.put((lmsg, lcmsg, rmsg, rcmsg))

    def mkgray(self, msg):
        return self.mc.mkgray(msg)

    def image_corners(self, im):
        (ok, corners, ids, b) = self.mc.get_corners(im)
        if ok:
            return corners, ids
        else:
            return None, None

    def handle_monocular(self, msg):

        (image, camera) = msg
        gray = self.mkgray(image)
        # not using downsample_and_detect because it gives unaccurate reprojections
        # scrib_mono, resized_corners, downsampled_corners, ids, board, (
        #     x_scale, y_scale) = self.mc.downsample_and_detect(gray)
        corners, ids = self.image_corners(gray)
        self.current_image_width = gray.shape[1]
        self.current_image_height = gray.shape[0]
        current_timestamp = image.header.stamp
        if corners is not None:
            # Comuptes the RMS error between a detected point on a row and the line defined by the leftmost and the rightmost detected points on the same row and  averages the RMS error across all rows.  This is a measure of how well the detected corners fit a chessboard pattern.
            linearity_rms = self.mc.linear_error(corners, ids, self.board)

            # Add in reprojection check
            image_points = corners
            object_points = self.mc.mk_object_points(
                [self.board], use_board_size=True)[0]
            dist_coeffs = numpy.zeros((4, 1))
            camera_matrix = numpy.array([[camera.p[0], camera.p[1], camera.p[2]],
                                         [camera.p[4], camera.p[5], camera.p[6]],
                                         [camera.p[8], camera.p[9], camera.p[10]]])
            print("image points matrix size : {}, object points matrix size : {}, camera matrix size : {}".format(
                image_points.shape, object_points.shape, camera_matrix.shape))
            print("corner ids detected: {}".format(ids.flatten()))
            if self.mc.pattern == Patterns.Chessboard:
                ok, rot, trans = cv2.solvePnP(
                    object_points, image_points, camera_matrix, dist_coeffs)
            elif self.mc.pattern == Patterns.ChArUco:
                rvec = numpy.array([[0.0],
                                    [0.0],
                                    [0.0]], dtype=numpy.float32)

                tvec = numpy.array([[0.0],
                                    [0.0],
                                    [0.0]], dtype=numpy.float32)
                ok, rot, trans = cv2.aruco.estimatePoseCharucoBoard(
                    corners, ids, self.board.charuco_board, camera_matrix, dist_coeffs, rvec, tvec)
            # Convert rotation into a 3x3 Rotation Matrix
            rot3x3, _ = cv2.Rodrigues(rot)
            # Reproject model points into image
            object_points_world = numpy.asmatrix(
                rot3x3) * numpy.asmatrix(object_points.squeeze().T) + numpy.asmatrix(trans)
            reprojected_h = camera_matrix * object_points_world
            reprojected = (reprojected_h[0:2, :] / reprojected_h[2, :])
            #filter out points in the reprojection that are not part of the detected corners
            filtered_reprojected = [reprojected[:,i] for i in ids]
            
            if self.mc.pattern != Patterns.ChArUco:
                reprojection_errors = image_points.squeeze().T - filtered_reprojected
            else:
                #charuco board are represented as columns x rows, so we need to swap x and y in the reprojection to compute the error correctly
                swapped_reprojected = filtered_reprojected.copy()
                swapped_reprojected[[0,1]] = filtered_reprojected[[1,0]]
                reprojection_errors = image_points.squeeze().T - swapped_reprojected.T

            reprojection_rms = numpy.sqrt(numpy.sum(numpy.array(
                reprojection_errors) ** 2) / numpy.product(reprojection_errors.shape))

            # Print the results
            print("Linearity RMS Error: %.3f Pixels      Reprojection RMS Error: %.3f Pixels" % (
                linearity_rms, reprojection_rms))

            if linearity_rms is not None and reprojection_rms is not None and current_timestamp not in [t[0] for t in self.linearity_rms_errors]:
                self.linearity_rms_errors.append(
                    (current_timestamp, linearity_rms))
                self.reproj_rms_errors.append(
                    (current_timestamp, reprojection_rms))

            if len(self.linearity_rms_errors) >= 20:
                self.enough_data = True
        else:
            print(f'{current_timestamp} : no chessboard')
            linearity_rms = None
            reprojection_rms = None

        scrib = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        drawable = MonoDrawable()
        # draw chessboard on image for display
        if corners is not None:
            if self.board.pattern == 'chessboard':
                cv2.drawChessboardCorners(
                    scrib, (self.board.n_cols, self.board.n_rows), corners, True)
            elif self.board.pattern == 'charuco':
                cv2.aruco.drawDetectedCornersCharuco(scrib, corners, ids)

        drawable.scrib = scrib
        drawable.linear_error = linearity_rms
        drawable.reproj_error = reprojection_rms

        self.redraw_monocular(drawable)

    @classmethod
    def putText(cls, img, text, org, color=(0, 0, 0)):
        cv2.putText(img, text, org, cls.FONT_FACE, cls.FONT_SCALE,
                    color, thickness=cls.FONT_THICKNESS)

    @classmethod
    def getTextSize(cls, text):
        return cv2.getTextSize(text, cls.FONT_FACE, cls.FONT_SCALE, cls.FONT_THICKNESS)[0]

    def text_height(self, start_height, i):
        return start_height + 40 + i * 30

    def redraw_monocular(self, drawable):
        height = drawable.scrib.shape[0]
        width = drawable.scrib.shape[1]
        display = numpy.zeros(
            (height, width + 100, 3), dtype=numpy.uint8)
        image_top_edge = (display.shape[0] - height) // 2

        display[image_top_edge:image_top_edge +
                height, 0:width, :] = drawable.scrib
        display[:, width:width+100, :].fill(255)
        self.buttons(display)

        self.putText(display, "lin.err",
                     (width, self.text_height(image_top_edge, 0)))
        linerror = drawable.linear_error
        if linerror is None or linerror < 0:
            msg = "?"
        else:
            msg = "%.4f" % linerror
        self.putText(
            display, msg, (width, self.text_height(image_top_edge, 1)))
        self.putText(display, "reproj.err",
                     (width, self.text_height(image_top_edge, 2)))
        reprojerror = drawable.reproj_error
        if reprojerror is None or reprojerror < 0:
            msg = "?"
        else:
            msg = "%.4f" % reprojerror
        self.putText(
            display, msg, (width, self.text_height(image_top_edge, 3)))

        self._last_display = display
        self._queue_display.put(display)

    def handle_stereo(self, msg):

        (lmsg, lcmsg, rmsg, rcmsg) = msg
        lgray = self.mkgray(lmsg)
        rgray = self.mkgray(rmsg)

        L, _ = self.image_corners(lgray)
        R, _ = self.image_corners(rgray)
        if L is not None and R is not None:
            epipolar = self.sc.epipolar_error(L, R)

            dimension = self.sc.chessboard_size(
                L, R, self.board, msg=(lcmsg, rcmsg))

            print("epipolar error: %f pixels   dimension: %f m" %
                  (epipolar, dimension))
        else:
            print("no chessboard")

    def initWindow(self):
        print("Initializing display window...")
        cv2.namedWindow("display", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('display', self.on_mouse_movement)

    main_button_height = 100
    main_button_width = 100
    save_start = 200

    def on_mouse_movement(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.current_image_width < x:
            if self.save_start <= y <= self.save_start + self.main_button_height and self.enough_data:
                self.save_errors()
                self.buttons(self._last_display)
                self._queue_display.put(self._last_display)
            if self._last_display is not None:
                print("Pixel value at click:", self._last_display[y, x])

    def save_errors(self):
        if not self.linearity_rms_errors or not self.reproj_rms_errors:
            print("No error data to save.")
            return

        # Save linearity RMS errors
        with open(f"{self.error_output_filename}_linear_rms_errors.txt", "w") as f:
            for timestamp, error in self.linearity_rms_errors:
                f.write(f"{timestamp.sec}.{timestamp.nanosec}, {error}\n")

        # Save reprojection RMS errors
        with open(f"{self.error_output_filename}_reprojection_rms_errors.txt", "w") as f:
            for timestamp, error in self.reproj_rms_errors:
                f.write(f"{timestamp.sec}.{timestamp.nanosec}, {error}\n")

        print(
            f"Errors saved to {self.error_output_filename}_linear_rms_errors.txt and {self.error_output_filename}_reprojection_rms_errors.txt")

    def button(self, dst, label, enable):
        dst.fill(255)
        size = (dst.shape[1], dst.shape[0])
        if enable:
            color = (155, 155, 80)
        else:
            color = (224, 224, 224)
        cv2.circle(dst, (size[0] // 2, size[1] // 2),
                   min(size) // 2, color, -1)
        (w, h) = self.getTextSize(label)
        self.putText(
            dst, label, ((size[0] - w) // 2, (size[1] + h) // 2), (255, 255, 255))

    def buttons(self, display):
        x = self.current_image_width
        # Add Save button
        self.button(display[self.save_start:self.save_start+self.main_button_height,
                    x:x+self.main_button_width], "Save", self.enough_data)
