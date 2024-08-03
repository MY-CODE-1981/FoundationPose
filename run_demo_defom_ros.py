# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import * 
import argparse
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ros_utils import *
import ros_numpy
from FastSAM.fastsam import FastSAM, FastSAMPrompt 
from std_msgs.msg import String
import clip
logger = logging.getLogger('my-logger')
logger.propagate = False

class FoundationPoseRos:
  def __init__(self, args):
    self.args = args

    cam_info = rospy.wait_for_message(args.camera_info_topic, CameraInfo)
    self.K, self.w, self.h = set_intrinsic(cam_info)
    # self.K[1, 1] = self.K[0, 0]
    self.bridge = CvBridge()

    result_topic = "foundationpose_result"
    self.result_topic = rospy.Publisher(result_topic , Image, queue_size=10)

    result_topic = "fastsam_result"
    self.fastsam_result_topic = rospy.Publisher(result_topic , Image, queue_size=10)
    self.clip_setup()
    self.fast_sam_setup()


    self.init_status = False
    self.seg_status = False
    # rgb = message_filters.Subscriber(args.rgb_topic, Image)
    # depth = message_filters.Subscriber(args.depth_topic, Image)
    # text = message_filters.Subscriber(args.text_topic, String)
    # ts = message_filters.ApproximateTimeSynchronizer([rgb, depth], 1, 1)
    # ts.registerCallback(self.foundation_pose)

    # ts = message_filters.ApproximateTimeSynchronizer([rgb ,text, depth], 1, 1, allow_headerless = True)
    # ts.registerCallback(self.fast_sam_inference)

    self.color_flag = False
    self.depth_flag = False
    self.text_flag = False
    self.color = []
    self.depth = []
    self.text = []
    
    self.color_sub = rospy.Subscriber(args.rgb_topic, Image, self.color_callback)

    self.depth_sub = rospy.Subscriber(args.depth_topic, Image, self.depth_callback)
    
    self.text_sub = rospy.Subscriber(args.text_topic, String, self.text_callback)

    self.process_rate = 10
    rospy.Timer(rospy.Duration(1.0 / self.process_rate), self.fast_sam_inference)

    rospy.Timer(rospy.Duration(1.0 / self.process_rate), self.foundation_pose)
    
    self.count_first = False

  def clip_setup(self):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.clip_model, preprocess = clip.load("ViT-B/32", device=device)


  def fast_sam_setup(self):
    self.fastsam_model = FastSAM(self.args.fast_sam_model)
    folder_path = "./demo_data"
    self.folder_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    text = clip.tokenize(self.folder_names).to(device)
    self.model_name_features = self.clip_model.encode_text(text)

  def color_callback(self, msg):
    if self.color_flag == False:
        print("color_callback")
        self.color = msg # self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.color_flag = True

  def depth_callback(self, msg):
    if self.depth_flag == False:
        print("depth_callback")
        # self.depth = msg # self.bridge.imgmsg_to_cv2(msg, "16UC1")
        self.depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        self.depth_flag = True
              
  def text_callback(self, msg):
    if self.text_flag == False:
      print("text_callback")
      self.text = msg
      self.text_flag = True

  def fast_sam_inference(self, event):
    print("callback outside")
    if self.color_flag and self.depth_flag and self.text_flag:
      print("callback_inside")
      print("fast sam call")
      text = self.text.data
      # self.seg_status = False
      # self.init_status = False
      if text == "stop":
        return
      text_tokens = clip.tokenize([text]).to(device)
      text_features = self.clip_model.encode_text(text_tokens)
      similarity = self.model_name_features@text_features.T
      max_index = torch.argmax(similarity)
      folder = self.folder_names[max_index]
      print("selected model is ", folder)
      if self.count_first is not True:
        self.foundation_pose_setup(folder)
        self.count_first = True

      input = ros_numpy.numpify(self.color).astype(np.uint8)
      # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
      # depth =  ros_numpy.numpify(self.depth)/1000.0


      everything_results = self.fastsam_model(input,device=self.args.device,retina_masks=self.args.retina,imgsz=self.args.imgsz, conf=self.args.conf,iou=self.args.iou)
      prompt_process = FastSAMPrompt(input, everything_results, device=self.args.device)
      ann = prompt_process.text_prompt(text=text)
    
      mask= 255*ann[0].astype('uint8')
      msg = ros_numpy.msgify(Image,mask, encoding="mono8")
      self.fastsam_result_topic.publish(msg)
      self.seg = mask
      self.seg_status = True
      self.color_flag = False
      self.depth_flag = False
      self.text_flag = False


  def foundation_pose_setup(self, folder):
    self.mesh = trimesh.load(os.path.join(self.args.mesh_folder, folder, "mesh/textured_simple.obj"))
    self.debug = args.debug
    self.debug_dir = args.debug_dir
    os.system(f'rm -rf {self.debug_dir}/* && mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam')

    self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
    self.bbox = np.stack([-self.extents/2, self.extents/2], axis=0).reshape(2,3)

    self.scorer = ScorePredictor()
    self.refiner = PoseRefinePredictor()
    self.glctx = dr.RasterizeCudaContext()
    self.est = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, scorer=self.scorer, refiner=self.refiner, debug_dir=self.debug_dir, debug=self.debug, glctx=self.glctx)
    logging.info("estimator initialization done")

  def foundation_pose(self, event):#rgb, depth):
    if not self.seg_status:
      return
    print("foundation pose init")
    # depth =  ros_numpy.numpify(self.depth)/1000.0
    depth = self.depth.astype(np.float64)
    color = ros_numpy.numpify(self.color).astype(np.uint8)
    # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    seg = self.seg
    if not self.init_status:
      print("############################")
      print("0-0")
      mask = seg.astype(bool)
      start = time.time()
      pose = self.est.register(K=self.K, rgb=color, depth=depth, ob_mask=mask, iteration=self.args.est_refine_iter)
      time_taken = time.time() - start
      self.init_status = True

    else:
      try:
        print("############################")
        print("0-1")
        pose = self.est.track_one(rgb=color, depth=depth, K=self.K, iteration=args.track_refine_iter)
        center_pose = pose@np.linalg.inv(self.to_origin)
        vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
        msg = ros_numpy.msgify(Image, vis[...,::-1], encoding="bgr8")
        self.result_topic.publish(msg)
      except:
        print("error for pose estimation")




if __name__=='__main__':
  rospy.init_node('foundation_pose_node')

  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_folder', type=str, default=f'{code_dir}/demo_data/')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--text_topic', type=str, default="/text_prompt")
  # parser.add_argument('--rgb_topic', type=str, default="/camera/color/image_raw")
  # parser.add_argument('--depth_topic', type=str, default="/camera/aligned_depth_to_color/image_raw")
  parser.add_argument('--rgb_topic', type=str, default="/camera/infra1/image_raw")
  parser.add_argument('--depth_topic', type=str, default="/camera/infra1/depth_raw")
  parser.add_argument('--camera_info_topic', type=str, default="/camera/infra1/camera_info")
  parser.add_argument('--fast_sam_model', type=str, default=f'{code_dir}/FastSAM/weights/FastSAM-x.pt')
  parser.add_argument("--imgsz", type=int, default=1280, help="image size")
  parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
  parser.add_argument(
        "--conf", type=float, default=0.9, help="object confidence threshold"
    )
  device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
  parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
  parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )

  

  
  args = parser.parse_args()




  set_logging_format()
  set_seed(0)

  FoundationPoseRos(args=args)
  rospy.spin()



  