import sys
import os
from flask import Flask, jsonify, request
import base64
import io

import trimesh
import numpy as np
from estimater import *
from datareader import *
import cv2
import logging

app = Flask(__name__)
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext() #what is this used for?
estimators = {}

def encode_ndarray(data):
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def decode_ndarray(data):
    decoded_image = base64.b64decode(data)
    buffer = io.BytesIO(decoded_image)
    return np.load(buffer)

class PoseEstimator:
    def __init__(self, mesh_file):
        set_seed(0)
        mesh = trimesh.load(mesh_file)
        mesh.apply_scale(0.001)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        # scorer = ScorePredictor()
        # refiner = PoseRefinePredictor()
        # glctx = dr.RasterizeCudaContext() #what is this used for?
        # print(mesh)
        self.est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir='./debug_pose', debug=1, glctx=glctx)
        
    def pose_estimation(self, K, image, depth, segmentation, register=True, est_refine_iter=5, track_refine_iter=2, visualize=False):
        '''
        use 3D models, segmentations, and rgbd to estimate the 6dof pose of the object
        input:
            model: 3d model of the object
            segmentation: segmentation of the object
            rgbd: rgbd image
        return
            pose
        '''
        #TODO Hongchao
        if register:
            pose = self.est.register(K, rgb=image, depth=depth, ob_mask=segmentation, iteration=est_refine_iter)
        else:
            pose = self.est.track_one(rgb=image, depth=depth, K=K, ob_mask=segmentation, iteration=track_refine_iter)
        center_pose = pose@np.linalg.inv(self.to_origin)
        if visualize:
            vis = draw_xyz_axis(image, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
            # cv2.imshow('1', vis[...,::-1])
            # cv2.waitKey(1)
            return center_pose, vis
        return center_pose

@app.route('/init_estimator', methods=['POST'])
def init_estimator():
    global estimators
    mesh_file = request.json.get('mesh_file')
    label = request.json.get('label')
    estimators[label] = PoseEstimator(mesh_file)
    return jsonify({"status": "success"})

@app.route('/pose_estimation', methods=['POST'])
def pose_estimation():
    label = request.json.get('label')
    K = decode_ndarray(request.json.get('K'))
    image = decode_ndarray(request.json.get('image'))
    depth = decode_ndarray(request.json.get('depth'))
    segmentation = decode_ndarray(request.json.get('segmentation'))
    register = request.json.get('register')
    logging.info(K, image[0,0], depth[0,0], segmentation[0,0])
    pose, vis = estimators[label].pose_estimation(K, image, depth / 1e3, segmentation, register=register, visualize=True)
    results = {"pose": encode_ndarray(pose), "vis": encode_ndarray(vis)}
    return results




    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)