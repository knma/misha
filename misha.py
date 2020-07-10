import sys, os

import numpy as np
import argparse, json
from types import SimpleNamespace as SN

import bpy
from mathutils import Matrix, Vector, Quaternion, Euler


class Misha():
  def __init__(self, config):
    self.config = config
    self.kintree = {
       -1 : (-1,'root'),
        0 : (-1, 'Pelvis'),
        1 : (0, 'L_Hip'),
        2 : (0, 'R_Hip'),
        3 : (0, 'Spine1'),
        4 : (1, 'L_Knee'),
        5 : (2, 'R_Knee'),
        6 : (3, 'Spine2'),
        7 : (4, 'L_Ankle'),
        8 : (5, 'R_Ankle'),
        9 : (6, 'Spine3'),
        10 : (7, 'L_Foot'),
        11 : (8, 'R_Foot'),
        12 : (9, 'Neck'),
        13 : (9, 'L_Collar'),
        14 : (9, 'R_Collar'),
        15 : (12, 'Head'),
        16 : (13, 'L_Shoulder'),
        17 : (14, 'R_Shoulder'),
        18 : (16, 'L_Elbow'),
        19 : (17, 'R_Elbow'),
        20 : (18, 'L_Wrist'),
        21 : (19, 'R_Wrist'),
        22 : (20, 'L_Hand'),
        23 : (21, 'R_Hand')
    }
    self.n_bones = 24

  def start(self):
    motion = np.load(self.config['motion'])
    self.motion = motion = SN(**motion)

    self.gender = str(motion.gender)[:1]
    self.n_frames = min(len(motion.poses), self.config['n_frames_limit'])

    print(list(vars(motion).keys()))
    print(f'frames {self.n_frames}')
    print(f'gender {self.gender}')
    print(f'mocap_framerate {motion.mocap_framerate}')

    bpy.context.scene.frame_end = self.n_frames-1
    # bpy.context.scene.render.fps = int(motion.mocap_framerate)

    bpy.ops.import_scene.fbx(
        filepath=os.path.join('model', f'basicModel_{self.gender}_lbs_10_207_0_v1.0.2.fbx'),
        axis_forward='Y', axis_up='Z', global_scale=100
    )

    self.obname = f'{self.gender}_avg'
    self.ob = bpy.data.objects[self.obname]
    self.ob.data.use_auto_smooth = False
    self.shape = motion.betas[:10]

    self.ob.data.shape_keys.animation_data_clear()
    self.arm_ob = bpy.data.objects['Armature']
    self.arm_ob.rotation_euler = Euler((np.radians(0), 0, 0), 'XYZ')
    self.arm_ob.animation_data_clear()

    self.ob.select_set(True)
    bpy.context.view_layer.objects.active = self.ob
    for k in self.ob.data.shape_keys.key_blocks.keys():
        self.ob.data.shape_keys.key_blocks[k].slider_min = -10
        self.ob.data.shape_keys.key_blocks[k].slider_max = 10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_min = -10
        bpy.data.shape_keys["Key"].key_blocks[k].slider_max = 10

    self.arm_ob.select_set(True)
    bpy.context.view_layer.objects.active = self.arm_ob
    self.deselect()
    self.ob.select_set(True)
    bpy.context.view_layer.objects.active = self.ob

    for i in range(self.n_frames):
      pose = motion.poses[i]
      print(pose.shape)
      pose = np.reshape(pose, (-1,3))
      pose = pose[:self.n_bones]

      trans = motion.trans[i]
      trans = list(trans)
      self.apply_pose_shape(pose, trans, self.shape, frame=i)

  def deselect(self):
      for o in bpy.data.objects.values():
          o.select_set(False)
      bpy.context.view_layer.objects.active = None

  def get_bname(self, i, obname='f_avg'):
      return obname+'_'+ self.kintree[i][1]

  def rodrigues(self, rotvec):
      theta = np.linalg.norm(rotvec)
      r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
      cost = np.cos(theta)
      mat = np.asarray([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]])
      return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

  def rodrigues2bshapes(self, pose, mat_pose=False):
      if mat_pose:
          mat_rots = np.zeros((self.n_bones, 3, 3))
          mat_rots[1:] = pose[1:]
      else:
          rod_rots = np.asarray(pose).reshape(self.n_bones, 3)
          mat_rots = [self.rodrigues(rod_rot) for rod_rot in rod_rots]
      bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
      return(mat_rots, bshapes)

  def apply_pose_shape(self, pose, trans, shape, frame=None, mat_pose=False):
      mrots, bsh = self.rodrigues2bshapes(pose, mat_pose)

      for ibone, mrot in enumerate(mrots):
          bname = self.get_bname(ibone)
          bone = self.arm_ob.pose.bones[self.get_bname(ibone, obname=f'{self.gender}_avg')]
          if ibone == 0:
            bone.location = trans
          bone.rotation_quaternion = Matrix(mrot).to_quaternion()
          if frame is not None:
            bone.keyframe_insert('rotation_quaternion', frame=frame)
            if ibone == 0:
              bone.keyframe_insert('location', frame=frame)

      for ibshape, bshape in enumerate(bsh):
          self.ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].value = bshape
          if frame is not None:
              self.ob.data.shape_keys.key_blocks['Pose%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)

      for ibshape, shape_elem in enumerate(shape):
          self.ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].value = shape_elem
          if frame is not None:
              self.ob.data.shape_keys.key_blocks['Shape%03d' % ibshape].keyframe_insert('value', index=-1, frame=frame)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Misha models')
  parser.add_argument('--config', nargs='*', type=str)
  if '--' in sys.argv:
      args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

  with open(args.config[0], 'r') as config_file:
      config = json.load(config_file)
      print(config)

  misha = Misha(config)
  misha.start()
